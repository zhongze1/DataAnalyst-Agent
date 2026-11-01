"""DA-008 工具调用审计落盘测试。

要锁定的：
1. 每次工具调用（含失败）都落一条 JSONL；
2. **只记参数形状不记内容** —— 与日志同一条隐私底线；
3. 审计写失败绝不能影响分析主流程；
4. 记录带 request_id / session_id，可与日志交叉回查。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from app.audit import AuditLogger, get_audit_logger
from app.context import request_id_var, session_id_var


@pytest.fixture
def ctx() -> "object":
    frame = pd.DataFrame({"a": [1, 2, 3]})
    from app.tools.base import ToolContext

    return ToolContext(session_id="sess-1", sheets=["S"], frames={"S": frame})


def _read_lines(audit_dir: Path) -> list[dict]:
    files = sorted(audit_dir.glob("*.jsonl"))
    if not files:
        return []
    return [
        json.loads(line)
        for line in files[-1].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ============================================================ 落盘
def test_successful_tool_call_is_recorded(registry, ctx, settings) -> None:
    registry.run("run_pandas", {"code": "df.shape"}, ctx)
    records = _read_lines(settings.audit_dir)
    assert len(records) == 1
    record = records[0]
    assert record["event"] == "tool_call"
    assert record["tool"] == "run_pandas"
    assert record["ok"] is True
    assert record["error_code"] is None
    assert record["result_chars"] > 0
    assert record["duration_ms"] >= 0


def test_failed_tool_call_records_error_code(registry, ctx, settings) -> None:
    """失败的调用更要留痕 —— 审计首先要能回答「谁失败了、为什么」。"""
    registry.run("load_sheet", {"sheet": "不存在"}, ctx)
    records = _read_lines(settings.audit_dir)
    assert len(records) == 1
    assert records[0]["ok"] is False
    assert records[0]["error_code"] == "SHEET_NOT_FOUND"


def test_unknown_tool_is_recorded(registry, ctx, settings) -> None:
    registry.run("__nope__", {}, ctx)
    records = _read_lines(settings.audit_dir)
    assert records[0]["error_code"] == "UNKNOWN_TOOL"
    assert records[0]["ok"] is False


# ============================================================ 隐私：只记形状
def test_arguments_content_is_never_written(registry, ctx, settings) -> None:
    """参数里可能有客户名、手机号甚至整段代码，审计只记形状。"""
    secret = "张三 13800138000"
    registry.run("run_pandas", {"code": f"out = '{secret}'"}, ctx)
    raw = (settings.audit_dir).glob("*.jsonl")
    content = next(raw).read_text(encoding="utf-8")
    assert secret not in content
    assert "张三" not in content
    assert "13800138000" not in content
    # 但保留了可回溯的形状
    record = _read_lines(settings.audit_dir)[0]
    assert record["args_summary"].startswith("code(str,")


# ============================================================ 上下文交叉回查
def test_record_carries_request_and_session_id(registry, ctx, settings) -> None:
    token_r = request_id_var.set("req-abc")
    token_s = session_id_var.set("sess-xyz")
    try:
        registry.run("run_pandas", {"code": "df.shape"}, ctx)
    finally:
        request_id_var.reset(token_r)
        session_id_var.reset(token_s)
    record = _read_lines(settings.audit_dir)[0]
    assert record["request_id"] == "req-abc"
    assert record["session_id"] == "sess-xyz"


# ============================================================ 开关与健壮性
def test_audit_can_be_disabled(registry, ctx, settings) -> None:
    settings.audit_enabled = False
    registry.run("run_pandas", {"code": "df.shape"}, ctx)
    assert _read_lines(settings.audit_dir) == []


def test_write_failure_does_not_break_tool(registry, ctx, settings, monkeypatch, caplog) -> None:
    """审计写失败绝不能连带分析失败 —— 它只是旁路记录。"""
    caplog.set_level(logging.WARNING)

    def _boom(*args, **kwargs):
        raise OSError("磁盘只读")

    monkeypatch.setattr(Path, "open", _boom)
    ok, out = registry.run("run_pandas", {"code": "df.shape"}, ctx)
    assert ok, f"工具本身不应受影响: {out}"
    assert "审计记录写入失败" in caplog.text


def test_logger_creates_daily_file(settings) -> None:
    logger = AuditLogger(audit_dir=settings.audit_dir / "nested")
    assert logger.dir.exists()
    logger.record({"event": "test"})
    files = list(logger.dir.glob("*.jsonl"))
    assert len(files) == 1
    assert len(files[0].name) == len("2026-01-01.jsonl")


def test_singleton_is_reused(settings) -> None:
    assert get_audit_logger() is get_audit_logger()


def test_non_serializable_extra_is_handled(settings) -> None:
    """不可序列化的字段不能让审计崩掉（default=str 兜底）。"""
    settings.audit_dir.mkdir(parents=True, exist_ok=True)
    logger = AuditLogger(audit_dir=settings.audit_dir)
    logger.record({"event": "test", "obj": object(), "n": 1})
    record = _read_lines(settings.audit_dir)[-1]
    assert isinstance(record["obj"], str)
    assert record["n"] == 1
