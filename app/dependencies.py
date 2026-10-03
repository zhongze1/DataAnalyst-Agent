"""依赖装配：单例（LLM 客户端 / 会话仓库 / Agent）。

路由层只通过 ``get_agent()`` / ``get_store()`` 拿对象，不关心具体实现，方便替换与测试。
"""

from __future__ import annotations

import logging
from functools import lru_cache

from app.agent.loop import DataAnalystAgent
from app.config import Settings, get_settings
from app.llm.base import LLMClient, LLMError
from app.llm.mock_client import MockLLMClient
from app.llm.openai_client import OpenAICompatClient
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


def describe_settings() -> dict[str, str]:
    settings: Settings = get_settings()
    return {
        "provider": settings.llm_provider,
        "model": settings.llm_model if settings.llm_provider == "openai" else "mock",
        "max_steps": str(settings.max_agent_steps),
    }
