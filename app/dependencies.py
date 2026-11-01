"""依赖装配：单例（LLM 客户端 / 会话仓库 / Agent）。

路由层只通过 ``get_agent()`` / ``get_store()`` 拿对象，不关心具体实现，方便替换与测试。
"""

from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import Depends, Request

from app.agent.loop import DataAnalystAgent
from app.config import Settings, get_settings
from app.errors import AuthRequiredError, PermissionDeniedError, SessionNotFoundError
from app.llm.base import LLMClient, LLMError
from app.llm.mock_client import MockLLMClient
from app.llm.openai_client import OpenAICompatClient
from app.services.auth_service import AuthService
from app.services.quota_service import LOCAL_USER_KEY, QuotaService
from app.storage.repository import UserRecord
from app.storage.session_store import Session, SessionStore
from app.tools import registry  # 触发 pandas_tools 的工具注册

logger = logging.getLogger(__name__)


@lru_cache
def get_llm() -> LLMClient:
    settings = get_settings()
    if settings.llm_provider == "mock":
        logger.warning("LLM provider = mock，使用本地假模型（不会真实分析数据）")
        return MockLLMClient()
    try:
        return OpenAICompatClient(settings)
    except LLMError as exc:
        # 启动即失败优于等到用户提问才报错
        logger.error("LLM 初始化失败: %s", exc)
        raise


@lru_cache
def get_store() -> SessionStore:
    return SessionStore()


@lru_cache
def get_agent() -> DataAnalystAgent:
    return DataAnalystAgent(llm=get_llm(), store=get_store(), tools=registry)


@lru_cache
def get_auth_service() -> AuthService:
    return AuthService(get_settings())


@lru_cache
def get_quota_service() -> QuotaService:
    """进程级配额单例。

    **必须**是单例：分钟窗口与当前并发都是进程内状态，每次请求新建一个
    QuotaService 等于没有限流。（测试里要用 ``cache_clear`` 隔离，见 conftest。）
    """
    return QuotaService(get_settings())


def quota_user_key(user: UserRecord | None) -> str:
    """把当前用户翻译成配额键。

    认证关闭时返回**固定的本机键**，而不是跳过限流：
    本机模式同样可能被脚本疯狂调用，同样要保护 LLM 费用。
    """
    settings = get_settings()
    if not settings.auth_enabled or user is None:
        return LOCAL_USER_KEY
    return user.user_id


def read_token(request: Request) -> str | None:
    """从 Cookie 读登录令牌。"""
    settings = get_settings()
    return request.cookies.get(settings.auth_cookie_name)


def current_user(
    request: Request,
    auth: AuthService = Depends(get_auth_service),
) -> UserRecord | None:
    """当前登录用户；未登录或认证关闭时返回 None。

    注意这里**不抛异常** —— 公开接口（登录页、静态资源）需要能匿名访问。
    """
    if not auth.enabled:
        return None
    return auth.resolve_token(read_token(request))


def require_user(
    user: UserRecord | None = Depends(current_user),
) -> UserRecord:
    """要求已登录。认证关闭时放行（返回 None 交由调用方按「本机模式」处理）。"""
    settings = get_settings()
    if not settings.auth_enabled:
        return None  # type: ignore[return-value]
    if user is None:
        raise AuthRequiredError()
    return user


def require_admin(user: UserRecord = Depends(require_user)) -> UserRecord:
    """要求管理员。"""
    settings = get_settings()
    if not settings.auth_enabled:
        return None  # type: ignore[return-value]
    if not user.is_admin:
        raise PermissionDeniedError("该操作需要管理员权限")
    return user


def owner_filter(user: UserRecord | None) -> str | None:
    """把当前用户翻译成会话归属过滤条件。

    认证关闭时返回 None（不过滤，本机模式）；开启时返回真实 user_id。
    """
    settings = get_settings()
    if not settings.auth_enabled:
        return None
    return user.user_id if user else None


def ensure_owner(session: Session | None, user: UserRecord | None) -> Session:
    """校验会话归属，不通过一律按「不存在」抛 404。

    **这是全工程唯一的归属校验实现**。此前 files / export / sessions 各写了一份
    变体，唯独对话入口漏了校验 —— 于是任何人只要拿到别人的 session_id，就能
    借 ``/api/chat`` 在别人的数据上跑分析、还能用 ``/api/chat/reset`` 清空对方对话。
    重复实现正是那个漏洞的根因，所以这里收敛成一处。

    返回 session 便于调用方直接链式使用（``session = ensure_owner(...)``），
    省掉「校验过了还要再 assert 一次非 None」的重复判断。

    Args:
        session: 已从仓库取出的会话，可能为 None。
        user: 当前登录用户；认证关闭时由 ``require_user`` 返回 None。

    Raises:
        SessionNotFoundError: 会话不存在，或不属于当前用户。

    注意：归属不符时**必须**返回 404 而非 403 —— 403 等于告诉对方
    「这个 id 确实存在」，本身就是信息泄露。
    """
    settings = get_settings()
    if not settings.auth_enabled:
        # 本机模式：不区分归属，但仍要求会话真实存在
        if session is None:
            raise SessionNotFoundError()
        return session
    if session is None or session.owner_id != (user.user_id if user else None):
        raise SessionNotFoundError()
    return session


def describe_settings() -> dict[str, str]:
    settings: Settings = get_settings()
    return {
        "provider": settings.llm_provider,
        "model": settings.llm_model if settings.llm_provider == "openai" else "mock",
        "max_steps": str(settings.max_agent_steps),
    }
