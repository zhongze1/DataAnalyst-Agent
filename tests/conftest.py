"""测试夹具。

三条硬性约束：
1. **离线**：所有用例用 MockLLMClient 及其子类，不触碰任何真实模型接口；
2. **隔离配置**：用 `Settings(_env_file=None)` 构造，测试结果不受开发者本机 .env 影响；
3. **不碰用户数据目录**：真实 `get_settings()` 在导入时会创建仓库内的 `data/uploads`，
   所以这里把所有直接导入 `get_settings` 的模块统一替换成测试配置。
"""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app.agent.loop
import app.config
import app.dependencies
import app.main
import app.storage.session_store
import app.tools.base
from app.agent.loop import DataAnalystAgent
from app.config import Settings
from app.dependencies import get_agent, get_store
from app.llm.mock_client import MockLLMClient
from app.main import app as fastapi_app
from app.storage.session_store import Session, SessionStore
from app.tools import registry as global_registry
from app.tools.registry import ToolRegistry
from stubs import SHEET_MAIN, SHEET_SUMMARY


# --------------------------------------------------------------------- 配置
@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """与 .env 完全隔离的测试配置。步数上限压到 6，Mock 剧本刚好 4 步跑完。"""
    return Settings(
        _env_file=None,
        llm_provider="mock",
        upload_dir=tmp_path / "uploads",
        max_agent_steps=6,
        max_tool_output_chars=2000,
        max_upload_bytes=10 * 1024 * 1024,
    )


@pytest.fixture(autouse=True)
def isolate_settings(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """把所有 `from app.config import get_settings` 的模块统一指向测试配置。

    注意：这些模块都是在 import 时就把函数对象绑到自己命名空间里的，
    所以必须逐个替换；替换后由 monkeypatch 自动还原，无需手动清 lru_cache。
    """
    for module in (
        app.config,
        app.main,
        app.tools.base,
        app.storage.session_store,
        app.agent.loop,
        app.dependencies,
    ):
        if hasattr(module, "get_settings"):
            monkeypatch.setattr(module, "get_settings", lambda: settings, raising=False)
    yield


@pytest.fixture(autouse=True)
def enable_info_logging() -> Iterator[None]:
    """让 INFO 日志在测试中**真正被构造**。

    不开的话，根 logger 默认 WARNING，各处 ``logger.info(...)`` 会被静默跳过 ——
    extra 键与 LogRecord 内置属性冲突、格式化参数错误这类缺陷就会一路逃过测试。
    """
    root = logging.getLogger()
    original_level, original_handlers = root.level, list(root.handlers)
    root.setLevel(logging.INFO)
    root.addHandler(logging.NullHandler())
    try:
        yield
    finally:
        root.setLevel(original_level)
        root.handlers = original_handlers


# --------------------------------------------------------------------- 数据
@pytest.fixture
def registry() -> ToolRegistry:
    """全局工具注册表（含 pandas_tools 注册的全部工具）。"""
    return global_registry


@pytest.fixture
def store(settings: Settings) -> SessionStore:
    return SessionStore(root=settings.upload_dir)


@pytest.fixture
def sample_xlsx() -> bytes:
    """一份「脏」样本：含缺失值、重复行、离群点、类别列、日期列。"""
    sales = [100.0, 120.5, 99.0, 980.0, 105.0, 110.0, 102.0, 108.0, 990.0, 101.0,
             99.5, 103.0, 107.0, 104.0, 106.0, 102.5, 101.5, 103.5, 105.5, 108.5]
    frame = pd.DataFrame({
        "订单号": [f"SO{i:03d}" for i in range(20)],
        "下单日期": pd.date_range("2025-01-01", periods=20, freq="D"),
        "区域": ["华东", "华北", "华南"] * 6 + ["西南", "东北"],
        "销售额": sales,
        "销量": [3, 5, 2, 8, 1, 4, 6, 2, 9, 3, 1, 5, 7, 2, 4, 6, 3, 5, 2, 8],
        "备注": ["企业采购"] * 10 + [None] * 10,
    })
    frame = pd.concat([frame, frame.tail(2)], ignore_index=True)  # 制造 2 行重复
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name=SHEET_MAIN, index=False)
        frame.head(5).to_excel(writer, sheet_name=SHEET_SUMMARY, index=False)
    return buffer.getvalue()


@pytest.fixture
def garbage_bytes() -> bytes:
    """不是 Excel 的内容，用于校验解析失败时的清理逻辑。"""
    return b"this is definitely not an excel file"


@pytest.fixture
def session(store: SessionStore, sample_xlsx: bytes) -> Session:
    return store.create("测试数据.xlsx", sample_xlsx)


# ----------------------------------------------------------------- 组装
@pytest.fixture
def make_agent(store: SessionStore, settings: Settings):
    """按需构造 Agent，可换 LLM 桩、可覆盖单次运行的配置。"""

    def _make(llm: Any = None, **overrides: Any) -> DataAnalystAgent:
        return DataAnalystAgent(
            llm=llm if llm is not None else MockLLMClient(),
            store=store,
            tools=global_registry,
            settings=settings.model_copy(update=overrides),
        )

    return _make


@pytest.fixture
def make_client(store: SessionStore, make_agent):
    """构造绑定了测试依赖的 TestClient，用例结束后自动还原。"""
    created: list[TestClient] = []

    def _make(llm: Any = None, **agent_overrides: Any) -> TestClient:
        fastapi_app.dependency_overrides[get_store] = lambda: store
        fastapi_app.dependency_overrides[get_agent] = lambda: make_agent(llm, **agent_overrides)
        client = TestClient(fastapi_app)
        created.append(client)
        return client

    try:
        yield _make
    finally:
        fastapi_app.dependency_overrides.clear()
        for client in created:
            client.close()


@pytest.fixture
def client(make_client) -> TestClient:
    """默认客户端：Mock LLM + 默认步数。"""
    # 刻意不使用 `with TestClient(app)`：那会触发 lifespan，
    # 进而调用真实 get_settings() 去创建仓库内的 data/uploads 目录。
    return make_client()
