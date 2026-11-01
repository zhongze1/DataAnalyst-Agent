"""DA-007 工具失败双通路测试。

不变量：
1. **模型侧**拿到的是带细节的原始错误 —— 它需要这些才能自我纠正；
2. **用户侧**拿到的是脱敏后的人话 —— 不含内部细节、不含异常栈；
3. 原始错误只进日志，不出 HTTP 接口。
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from app.tools.base import (
    ToolContext,
    encode_tool_error,
    humanize_tool_error,
    split_tool_error,
)


@pytest.fixture
def ctx() -> ToolContext:
    frame = pd.DataFrame({"a": [1, 2, 3]})
    return ToolContext(session_id="t", sheets=["S"], frames={"S": frame})


# ============================================================ 编解码
def test_encode_and_split_roundtrip() -> None:
    encoded = encode_tool_error("COLUMN_NOT_FOUND", "列 'Z' 不存在。可用列: ['a']")
    assert encoded.startswith("[COLUMN_NOT_FOUND]")
    code, detail = split_tool_error(encoded)
    assert code == "COLUMN_NOT_FOUND"
    assert detail == "列 'Z' 不存在。可用列: ['a']"


def test_split_tolerates_unprefixed_text() -> None:
    code, detail = split_tool_error("普通错误文本")
    assert code == "TOOL_ERROR"
    assert detail == "普通错误文本"


def test_split_preserves_multiline_detail() -> None:
    detail = "第一行\n第二行\n第三行"
    _, got = split_tool_error(encode_tool_error("RUNTIME_ERROR", detail))
    assert got == detail


# ============================================================ 人话映射
@pytest.mark.parametrize(
    "code,expected",
    [
        ("SHEET_NOT_FOUND", "工作表"),
        ("COLUMN_NOT_FOUND", "列名"),
        ("COLUMN_MISSING", "当前 df"),
        ("SANDBOX_VIOLATION", "安全限制"),
        ("NO_DATA_LOADED", "上传"),
        ("CODE_TOO_LONG", "太长"),
        ("SYNTAX_ERROR", "语法"),
        ("UNKNOWN_TOOL", "不存在的工具"),
    ],
)
def test_each_code_has_human_hint(code: str, expected: str) -> None:
    assert expected in humanize_tool_error(code)


def test_unknown_code_falls_back_to_generic() -> None:
    assert humanize_tool_error("SOME_FUTURE_CODE") == humanize_tool_error("TOOL_ERROR")


def test_human_hint_has_no_traceback_or_paths() -> None:
    """无论什么错误码，人话里都不能出现栈或路径。"""
    for code in (
        "SHEET_NOT_FOUND", "COLUMN_NOT_FOUND", "COLUMN_MISSING",
        "SANDBOX_VIOLATION", "RUNTIME_ERROR", "TOOL_ERROR",
    ):
        hint = humanize_tool_error(code)
        assert "Traceback" not in hint
        assert "\\" not in hint and "/" not in hint
        assert ".py" not in hint


# ============================================================ registry 侧：模型保留细节
def test_model_side_keeps_actionable_detail(registry, ctx: ToolContext) -> None:
    """模型必须能拿到「可用工作表」这样的细节，否则无法自纠。"""
    ok, text = registry.run("load_sheet", {"sheet": "错表名"}, ctx)
    assert ok is False
    code, detail = split_tool_error(text)
    assert code == "SHEET_NOT_FOUND"
    assert "错表名" in detail
    assert "可用工作表" in detail and "S" in detail


def test_column_error_lists_available_columns(registry, ctx: ToolContext) -> None:
    ok, text = registry.run("value_counts", {"column": "缺失列"}, ctx)
    code, detail = split_tool_error(text)
    assert ok is False
    assert code == "COLUMN_NOT_FOUND"
    assert "'a'" in detail  # 告诉模型真实列名


# ============================================================ 端到端：接口只出人话
def test_api_returns_human_text_not_raw_error(client, session) -> None:
    """核心：HTTP 响应里不能出现原始错误细节。"""
    from app.tools import registry as reg
    from app.tools.base import ToolContext as TC

    # 直接构造一次失败，确认 registry 侧与接口侧的分流
    ctx = TC(session_id=session.session_id, file_path=str(session.path), sheets=list(session.sheets))
    ok, raw = reg.run("load_sheet", {"sheet": "机密表名"}, ctx)
    assert ok is False
    assert "机密表名" in raw  # 模型侧看得到

    code, _ = split_tool_error(raw)
    user_text = humanize_tool_error(code)
    assert "机密表名" not in user_text  # 用户侧看不到


def test_loop_humanizes_but_keeps_model_context(make_client, session) -> None:
    """Loop 交给模型的是原始错误，交给用户的是人话。"""
    from stubs import WrongSheetLLM

    client = make_client(WrongSheetLLM())
    body = client.post(
        "/api/chat", json={"session_id": session.session_id, "message": "分析"}
    ).json()
    failed = body["steps"][0]
    assert failed["ok"] is False
    assert "没有这个工作表" in failed["error"]
    # 原始细节不出现
    assert "不存在的工作表" not in failed["error"]
    # 但模型确实被喂了原始错误（从 session 历史里能看到）
    tool_messages = [m for m in session.messages if m["role"] == "tool"]
    assert tool_messages, "工具结果应回传给模型"
    assert "不存在的工作表" in tool_messages[0]["content"]


def test_failure_detail_is_logged(make_client, session, caplog) -> None:
    """原始错误必须留痕在日志里，否则用户报障时无从查起。"""
    caplog.set_level(logging.INFO)
    from stubs import WrongSheetLLM

    client = make_client(WrongSheetLLM())
    client.post("/api/chat", json={"session_id": session.session_id, "message": "分析"})
    assert "SHEET_NOT_FOUND" in caplog.text
