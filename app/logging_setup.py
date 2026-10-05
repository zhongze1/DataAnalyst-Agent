"""结构化日志：JSON 输出 + 上下文自动注入 + 敏感参数脱敏。

三条规则：
1. 每条日志自动带上 request_id / session_id，不用调用方手工传递；
2. JSON 格式便于日志系统聚合，text 格式便于本地阅读（由 AGENT_LOG_FORMAT 切换）；
3. **绝不打业务数据**：工具参数只记名称与规模，不记内容。
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from app.context import current_request_id, current_session_id

# LogRecord 自带属性不是「额外字段」，不能混进 JSON payload
_RESERVED_ATTRS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {
    "asctime",
    "message",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """输出一行一个 JSON 对象。

    注意：request_id / session_id 是在**格式化时**从 ContextVar 现读的，
    而不是发日志时快照进 LogRecord。因此 handler 必须在请求上下文中**同步**
    调用 format()。若换成 QueueHandler 或后台格式化线程，上下文会错配，
    届时需要把 request_id 显式写进 extra。
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": current_request_id(),
        }
        session_id = current_session_id()
        if session_id != "-":
            payload["session_id"] = session_id

        # 调用方通过 extra={...} 传入的结构化字段
        for key, value in record.__dict__.items():
            if key in _RESERVED_ATTRS or key.startswith("_"):
                continue
            payload[key] = value

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """人读格式，末尾附上 request_id 便于串联因果。"""

    def __init__(self) -> None:
        super().__init__("%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        request_id = current_request_id()
        return f"{base} | req={request_id}" if request_id != "-" else base


def setup_logging(fmt: str = "text", level: str = "INFO") -> None:
    """配置根 logger。可重复调用（每次调用会重设格式），不会叠加 handler。"""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())

    # httpx 每发一次请求就打一行 INFO，只占篇幅没有信息量（我们自己的埋点
    # 记录了耗时、模型、token），这里压到 WARNING。
    for noisy in ("httpx", "httpcore", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def short_hash(text: str) -> str:
    """字符串参数的内容指纹（SHA256 前 8 位）。

    用途：判断「两次调用的参数是否相同」——这是事后追查最常见的需求，
    而读原文会泄露业务数据。8 位十六进制在单次会话的量级下足够区分。
    """
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:8]


def describe_arguments(arguments: dict[str, Any]) -> str:
    """把工具参数压成「名称(类型,规模,指纹)」，**丢弃内容**。

    工具参数里可能有客户名、手机号，甚至整段 pandas 代码；日志与审计只保留
    排障必需的形状信息：名称、类型、字符串长度、内容指纹。
    """
    if not arguments:
        return "-"
    parts: list[str] = []
    for key, value in arguments.items():
        if isinstance(value, str):
            parts.append(f"{key}(str,{len(value)},#{short_hash(value)})")
        elif isinstance(value, (list, tuple)):
            parts.append(f"{key}(list,{len(value)})")
        elif isinstance(value, dict):
            parts.append(f"{key}(dict,{len(value)})")
        else:
            parts.append(f"{key}({type(value).__name__})")
    return " ".join(parts)
