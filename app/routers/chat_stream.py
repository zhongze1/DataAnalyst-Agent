"""SSE 流式对话接口。

与 ``/api/chat`` 的关系：两者共用同一个 :meth:`DataAnalystAgent.run`，
区别只是本路由传了 ``on_event`` 回调。非流式接口保持原样，供 SSE 失败时降级。

客户端断开时（用户点「停止生成」或关闭页面），生成器被 ASGI 服务器取消，
``finally`` 里会取消后端正在跑的 task，不留孤儿任务。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.agent.events import error_event, done_event
from app.agent.loop import DataAnalystAgent
from app.context import set_session_id
from app.dependencies import get_agent, get_store
from app.errors import SessionNotFoundError
from app.schemas import ChatRequest
from app.storage.session_store import SessionStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

# 队列的结束哨兵：不能用 None，因为 None 可能被当作合法载荷
_SENTINEL = object()


@router.post("/stream", response_class=StreamingResponse)
async def chat_stream(
    payload: ChatRequest,
    agent: DataAnalystAgent = Depends(get_agent),
    store: SessionStore = Depends(get_store),
) -> StreamingResponse:
    """以 SSE 推送：开始 → 工具步骤 → 答案增量 → 收尾快照。"""
    session = store.get(payload.session_id)
    if payload.session_id and session is None:
        raise SessionNotFoundError()
    set_session_id(session.session_id if session else None)

    queue: asyncio.Queue = asyncio.Queue()

    async def on_event(event) -> None:
        await queue.put(event)

    async def runner() -> None:
        try:
            result = await agent.run(session, payload.message, on_event=on_event)
            await queue.put(
                done_event(
                    answer=result.answer,
                    steps=[
                        {
                            "step": s.step,
                            "tool": s.tool,
                            "arguments": s.arguments,
                            "ok": s.ok,
                            "result": s.result,
                            "error": s.error,
                        }
                        for s in result.steps
                    ],
                    stop_reason=result.stop_reason,
                    usage=result.usage or None,
                )
            )
        except asyncio.CancelledError:
            logger.info("客户端断开，已取消分析 | session=%s", session.session_id if session else "-")
            raise
        except Exception as exc:  # noqa: BLE001 - 兜底后转成 SSE 错误事件
            logger.exception("流式分析失败")
            await queue.put(error_event("INTERNAL_ERROR", "分析过程出错，请稍后重试"))
            _ = exc
        finally:
            await queue.put(_SENTINEL)

    task = asyncio.create_task(runner())

    async def event_stream() -> AsyncIterator[bytes]:
        try:
            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    break
                yield item.to_sse()
            # 任务异常时（LLM 故障等）也要让前端拿到明确的终止事件
            if task.done() and task.exception() is not None:
                yield error_event("LLM_SERVICE_ERROR", "分析失败，请稍后重试").to_sse()
        finally:
            # 客户端提前断开（点「停止生成」）时，取消后端任务
            if not task.done():
                task.cancel()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            # 让反向代理（nginx）不要缓冲，否则流式会被攒成一坨
            "X-Accel-Buffering": "no",
        },
    )

