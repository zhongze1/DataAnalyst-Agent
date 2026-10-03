"""工具层契约测试。

核心契约：**`registry.run()` 永不抛异常**，永远返回 `(ok: bool, text: str)`。
这条契约是 Agent Loop 不会因为一次工具失败而整体崩溃的根本保证。
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.tools.base import Tool, ToolContext, format_value
from app.tools.registry import ToolRegistry
from stubs import SHEET_MAIN


@pytest.fixture
def ctx(session, store) -> ToolContext:
    return ToolContext(
        session_id=session.session_id,
        file_path=str(session.path),
        file_name=session.filename,
        sheets=list(session.sheets),
        frames=session.frames,
    )


# -------------------------------------------------- 永不抛异常的核心契约
def test_unknown_tool_returns_error_tuple(registry) -> None:
    ok, text = registry.run("__不存在的工具__", {}, ToolContext(session_id="t"))
    assert ok is False
    assert "工具不存在" in text
    assert "可用工具" in text  # 必须告诉模型有哪些工具可用


def test_business_error_is_captured(registry, ctx: ToolContext) -> None:
    """ToolError 应被转成可纠正的提示，而不是抛出去。"""
    ok, text = registry.run("load_sheet", {"sheet": "不存在的工作表"}, ctx)
    assert ok is False
    assert text.startswith("[ToolError]")
    assert "可用工作表" in text  # 错误信息要含纠错所需信息


def test_unexpected_exception_is_contained() -> None:
    """非 ToolError 的意外异常也必须被兜住，否则整个 Loop 会挂。"""

    def boom(ctx: ToolContext) -> str:
        raise ValueError("unexpected boom")

    local = ToolRegistry()
    local.add(
        Tool(name="boom", description="d", parameters={"type": "object", "properties": {}}, func=boom)
    )
    ok, text = local.run("boom", {}, ToolContext(session_id="t"))
    assert ok is False
    assert "[RuntimeError]" in text
    assert "ValueError" in text


# ------------------------------------------------------------ 参数校验
def test_missing_required_argument(registry, ctx: ToolContext) -> None:
    ok, text = registry.run("load_sheet", {}, ctx)
    assert ok is False
    assert "缺少必填参数" in text and "sheet" in text


def test_unknown_argument_is_rejected(registry, ctx: ToolContext) -> None:
    ok, text = registry.run("preview_rows", {"不存在参数": 1}, ctx)
    assert ok is False
    assert "不支持参数" in text


def test_empty_string_counts_as_missing(registry, ctx: ToolContext) -> None:
    ok, text = registry.run("load_sheet", {"sheet": ""}, ctx)
    assert ok is False
    assert "缺少必填参数" in text


# ------------------------------------------------------------ 正常执行
def test_list_sheets_reports_loaded_state(registry, ctx: ToolContext) -> None:
    ok, text = registry.run("list_sheets", {}, ctx)
    assert ok is True
    assert "[Sheets]" in text and SHEET_MAIN in text


def test_health_check_surfaces_alerts(registry, ctx: ToolContext) -> None:
    """聚合工具必须能一次给出结构 + 缺失 + 告警。"""
    registry.run("load_sheet", {"sheet": SHEET_MAIN}, ctx)
    ok, text = registry.run("data_health_check", {}, ctx)
    assert ok is True
    assert "体检报告" in text
    assert "关注点" in text
    assert "备注" in text  # 样本里 备注 列缺失率 50%


def test_run_pandas_expression_and_statement(registry, ctx: ToolContext) -> None:
    registry.run("load_sheet", {"sheet": SHEET_MAIN}, ctx)
    ok_expr, text_expr = registry.run("run_pandas", {"code": "df.shape"}, ctx)
    assert ok_expr is True
    assert "22" in text_expr and "5" in text_expr  # 20 行 + 2 重复行, 5 列

    ok_stmt, text_stmt = registry.run(
        "run_pandas", {"code": "out = int(df['备注'].isna().sum())"}, ctx
    )
    assert ok_stmt is True
    assert text_stmt.strip() == "12"


@pytest.mark.parametrize(
    "code",
    [
        "import os",
        "open('/etc/passwd')",
        "eval('1+1')",
        "df.__class__",
        "__import__('os')",
    ],
)
def test_sandbox_blocks_dangerous_code(registry, ctx: ToolContext, code: str) -> None:
    registry.run("load_sheet", {"sheet": SHEET_MAIN}, ctx)
    ok, text = registry.run("run_pandas", {"code": code}, ctx)
    assert ok is False
    assert "[ToolError]" in text


# ------------------------------------------------------------ schema 合法性
def test_openai_schemas_are_well_formed(registry) -> None:
    schemas = registry.openai_schemas()
    assert schemas, "至少要注册一个工具"
    for schema in schemas:
        assert schema["type"] == "function"
        function = schema["function"]
        assert function["name"] and function["description"]
        params = function["parameters"]
        assert params["type"] == "object"
        assert isinstance(params.get("properties", {}), dict)
        for required in params.get("required", []):
            assert required in params["properties"]


def test_duplicate_tool_name_is_rejected() -> None:
    local = ToolRegistry()
    tool = Tool(name="dup", description="d", parameters={"type": "object"}, func=lambda ctx: "x")
    local.add(tool)
    with pytest.raises(ValueError, match="重复"):
        local.add(tool)


# ------------------------------------------------------------ 结果格式化
def test_format_value_handles_pandas_objects() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3]})
    text = format_value(frame)
    assert "[DataFrame]" in text and "shape=(3, 1)" in text

    series = pd.Series([1, 2, 3], name="a")
    assert "[Series]" in format_value(series)

    assert format_value({"k": "v"}) == "k: v"
    assert format_value("plain") == "plain"


def test_format_value_truncates(settings) -> None:
    settings.max_tool_output_chars = 50
    text = format_value("x" * 500)
    assert len(text) < 200
    assert "已截断" in text
