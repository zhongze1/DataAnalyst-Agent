"""HTTP 中间件：请求上下文与访问日志。

**必须是纯 ASGI 中间件**：``BaseHTTPMiddleware`` 会用 task + 内存流包装响应，
对 SSE 这类小片段高频的流式响应会产生缓冲，破坏 chunked 语义。
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any, MutableMapping

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.context import request_id_var, reset_request_context, set_session_id

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
_HEADER_BYTES = REQUEST_ID_HEADER.lower().encode()
# 客户端自带的 request_id 只允许这些字符：它会被写进响应头（HTTP 头必须是
# ASCII，中文会直接抛 UnicodeEncodeError → 500）、进日志、还会回显到前端。
# 与其信任上游，不如白名单校验，不合法就自己生成一个。
_REQUEST_ID_SAFE = re.compile(r"^[A-Za-z0-9._:/-]{1,128}$")


def _sanitize_request_id(raw: str | None) -> str:
    """校验客户端传来的 request_id；不合法（含非 ASCII / 超长 / 怪字符）则重新生成。"""
    if raw and _REQUEST_ID_SAFE.match(raw):
        return raw
    return uuid.uuid4().hex[:12]


class RequestContextMiddleware:
    """生成/透传 request_id，记录访问日志，并把 request_id 回写到响应头。

    客户端可以通过请求头自带 request_id（便于跨服务串联），没带则生成一个
    12 位短 ID。
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _sanitize_request_id(Headers(scope=scope).get(REQUEST_ID_HEADER))
        # 写进 scope：异常处理器运行时 ContextVar 已被重置，只能从 scope 取
        scope["request_id"] = request_id
        request_token = request_id_var.set(request_id)
        # 每个请求都从「无会话」开始，杜绝上一个请求的 session_id 串味
        session_token = set_session_id(None)
        started = time.perf_counter()
        state: MutableMapping[str, Any] = {"status": 500}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                state["status"] = message["status"]
                headers = message.setdefault("headers", [])
                headers.append((_HEADER_BYTES, request_id.encode("ascii")))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            logger.exception("未捕获异常 %s %s", scope.get("method"), scope.get("path"))
            raise
        else:
            duration_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "%s %s -> %d | %.1fms",
                scope.get("method"),
                scope.get("path"),
                state["status"],
                duration_ms,
                extra={
                    "method": scope.get("method"),
                    "path": scope.get("path"),
                    "status_code": state["status"],
                    "duration_ms": round(duration_ms, 1),
                },
            )
        finally:
            reset_request_context(request_token, session_token)

