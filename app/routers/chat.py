"""对话接口：把用户问题交给 Agent Loop，返回答案 + 工具调用轨迹。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.agent.loop import DataAnalystAgent
from app.context import set_session_id
from app.dependencies import (
    ensure_owner,
    get_agent,
    get_quota_service,
    get_store,
    quota_user_key,
    require_user,
)
from app.schemas import ERROR_RESPONSES, ChatRequest, ChatResponse, ToolStep
from app.services.quota_service import QuotaService
from app.storage.repository import UserRecord
from app.storage.session_store import SessionStore

router = APIRouter(prefix="/api/chat", tags=["chat"], responses=ERROR_RESPONSES)


@router.post("", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    agent: DataAnalystAgent = Depends(get_agent),
    user: UserRecord = Depends(require_user),
    store: SessionStore = Depends(get_store),
    quota: QuotaService = Depends(get_quota_service),
) -> ChatResponse:
    """单 Agent Loop 的入口：一次提问 = 一次完整的「思考 → 调工具 → 观察 → 回答」。

    会话归属由 :func:`~app.dependencies.ensure_owner` 统一校验：不是自己的会话
    一律 404。对话是**在别人数据上执行任意工具**的入口，漏掉校验等于把整个
    数据面开放给任何知道 session_id 的人。
    """
    # session_id 可省略（此时 session=None，Agent 会提示「请先上传文件」）；
    # 一旦给了就必须是自己的，否则 404。
    session = ensure_owner(store.get(payload.session_id), user) if payload.session_id else None

    # 绑定会话上下文：本次请求内所有日志都会自动带上 session_id
    set_session_id(session.session_id if session else None)

    # 配额预检必须在真正调用模型之前：被拒绝的请求不能产生任何 LLM 费用。
    # 顺序是「会话校验 → 预检占用 → 执行 → 结算 → 释放」，无效请求不吃额度。
    reservation = quota.reserve(
        user_key=quota_user_key(user),
        session_id=session.session_id if session else None,
    )
    try:
        result = await agent.run(session, payload.message)
    finally:
        # 正常结束、抛异常、被取消都要释放，否则并发会被永久锁死
        quota.release(reservation)

    # 结算失败不影响响应（settle 内部只记日志）
    quota.settle(reservation, result.usage)
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
    user: UserRecord = Depends(require_user),
    store: SessionStore = Depends(get_store),
) -> dict[str, bool]:
    """清空对话历史（保留已上传的数据），开始一轮新的分析。

    同样必须校验归属：否则任何人都能清空别人的对话历史。
    """
    session = ensure_owner(store.get(session_id), user)
    store.reset_messages(session)
    return {"reset": True}
