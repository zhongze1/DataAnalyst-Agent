"""依赖装配：单例（LLM 客户端 / 会话仓库 / Agent）。

路由层只通过 ``get_agent()`` / ``get_store()`` 拿对象，不关心具体实现，方便替换与测试。
"""

from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import Depends, Request

from app.agent.loop import DataAnalystAgent
from app.config import Settings, get_settings
from app.errors import AuthRequiredError, PermissionDeniedError
from app.llm.base import LLMClient, LLMError
from app.llm.mock_client import MockLLMClient
from app.llm.openai_client import OpenAICompatClient
from app.services.auth_service import AuthService
from app.storage.repository import UserRecord
from app.storage.session_store import SessionStore
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


def describe_settings() -> dict[str, str]:
    settings: Settings = get_settings()
    return {
        "provider": settings.llm_provider,
        "model": settings.llm_model if settings.llm_provider == "openai" else "mock",
        "max_steps": str(settings.max_agent_steps),
    }
