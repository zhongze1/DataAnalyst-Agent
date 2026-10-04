"""对话接口：把用户问题交给 Agent Loop，返回答案 + 工具调用轨迹。"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.agent.loop import DataAnalystAgent
from app.context import set_session_id
from app.dependencies import get_agent, get_store
from app.schemas import ChatRequest, ChatResponse, ToolStep
from app.storage.session_store import SessionStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    agent: DataAnalystAgent = Depends(get_agent),
    store: SessionStore = Depends(get_store),
) -> ChatResponse:
    """单 Agent Loop 的入口：一次提问 = 一次完整的「思考 → 调工具 → 观察 → 回答」。"""
    session = store.get(payload.session_id)
    if payload.session_id and session is None:
        raise HTTPException(status_code=404, detail="会话不存在或已过期，请重新上传文件。")

    # 绑定会话上下文：本次请求内所有日志都会自动带上 session_id
    set_session_id(session.session_id if session else None)

    result = await agent.run(session, payload.message)
    return ChatResponse(
        session_id=session.session_id if session else (payload.session_id or ""),
        answer=result.answer,
        steps=[
            ToolStep(
                step=step.step,
                tool=step.tool,
                arguments=step.arguments,
                ok=step.ok,
                result=step.result,
                error=step.error,
            )
            for step in result.steps
        ],
        stop_reason=result.stop_reason,  # type: ignore[arg-type]
        usage=result.usage or None,
    )


@router.post("/reset")
async def reset_conversation(
    session_id: str,
    store: SessionStore = Depends(get_store),
) -> dict[str, bool]:
    """清空对话历史（保留已上传的数据），开始一轮新的分析。"""
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在。")
    store.reset_messages(session)
    return {"reset": True}
