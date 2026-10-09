"""工具调用审计（DA-008）。

与日志的分工：
- **日志** 面向运维：可能被轮转、级别可调、格式随配置变；
- **审计** 面向合规：结构固定、长期留存、回答「谁在何时调了什么工具、拿到什么结果」。

隐私约束：与 DA-002 的日志一致，**只记参数形状不记内容**。
真要追查参数原文，靠 request_id 回查日志。
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.context import current_request_id, current_session_id
from app.logging_setup import describe_arguments

logger = logging.getLogger(__name__)

_write_lock = threading.Lock()


class AuditLogger:
    """按天切分的 JSONL 追加写入器。

    刻意做得很笨：一次 open、一次 write、一次 close。审计的可靠性优先于性能，
    工具调用频率本身很低（一次分析十几次）。
    """

    def __init__(self, audit_dir: Path | None = None) -> None:
        settings = get_settings()
        self._dir = Path(audit_dir or settings.audit_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def dir(self) -> Path:
        return self._dir

    def _today(self) -> Path:
        now = datetime.now(timezone.utc)
        return self._dir / f"{now:%Y-%m-%d}.jsonl"

    def record(self, event: dict[str, Any]) -> None:
        """追加一条审计记录。任何失败都只记日志，不影响主流程。"""
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            **event,
        }
        line = json.dumps(payload, ensure_ascii=False, default=str)
        target = self._today()
        try:
            with _write_lock:
                with target.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        except OSError:
            # 审计写失败绝不能连带分析失败
            logger.warning("审计记录写入失败: %s", target, exc_info=True)


_audit: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    global _audit
    if _audit is None:
        _audit = AuditLogger()
    return _audit


def record_tool_call(
    *,
    tool: str,
    arguments: dict[str, Any] | None,
    ok: bool,
    duration_ms: float,
    result_chars: int = 0,
    error_code: str | None = None,
) -> None:
    """记录一次工具调用。

    只落「形状」不落内容：``sheet(str,4)`` 而不是 ``sheet='销售明细'``。
    """
    if not get_settings().audit_enabled:
        return
    get_audit_logger().record(
        {
            "event": "tool_call",
            "request_id": current_request_id(),
            "session_id": current_session_id(),
            "tool": tool,
            "args_summary": describe_arguments(arguments or {}),
            "ok": ok,
            "duration_ms": round(duration_ms, 1),
            "result_chars": result_chars,
            "error_code": error_code,
        }
    )


def record_quota_event(
    *,
    event: str,
    dimension: str,
    user_key: str,
    limit: int | None = None,
    retry_after: int | None = None,
    usage: dict[str, Any] | None = None,
) -> None:
    """记录一次配额事件（DA-017）：拒绝或用量结算。

    与工具审计同样的隐私约束：只记**维度、上限、轮次**这类结构信息，
    ``user_key`` 是内部用户 id（本机模式为 ``"local"``），
    不记提示词、口令、Cookie 或任何业务数据原文。
    """
    if not get_settings().audit_enabled:
        return
    get_audit_logger().record(
        {
            "event": event,
            "request_id": current_request_id(),
            "session_id": current_session_id(),
            "user_key": user_key,
            "dimension": dimension,
            "limit": limit,
            "retry_after": retry_after,
            "usage": usage or None,
        }
    )
