"""HTTP 中间件：请求上下文与访问日志。

必须挂在最外层，才能保证「异常路径」和「静态资源」也带 request_id。
"""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import Request
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.context import request_id_var, reset_request_context, session_id_var, set_session_id

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """生成/透传 request_id，记录访问日志，并把 request_id 回写到响应头。

    客户端可以通过请求头自带 request_id（便于跨服务串联），
    没带则生成一个 12 位短 ID。
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:12]
        request_token = request_id_var.set(request_id)
        # 写进 scope：异常处理器运行时 ContextVar 已被重置，只能从 scope 取
        request.scope["request_id"] = request_id
        # 每个请求都从「无会话」开始，杜绝上一个请求的 session_id 串味
        session_token = set_session_id(None)
        started = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "未捕获异常 %s %s", request.method, request.url.path
            )
            raise
        else:
            duration_ms = (time.perf_counter() - started) * 1000
            response.headers[REQUEST_ID_HEADER] = request_id
            logger.info(
                "%s %s -> %d | %.1fms",
                request.method,
                request.url.path,
                response.status_code,
                duration_ms,
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration_ms, 1),
                },
            )
            return response
        finally:
            reset_request_context(request_token, session_token)
