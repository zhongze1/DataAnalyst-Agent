"""历史会话接口。

与 ``/api/files/*`` 的分工：那边管「上传的文件」，这里管「会话与对话」。
删除接口仍在 files.py（前端与既有测试都依赖它），此处不重复实现。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from app.dependencies import ensure_owner, get_store, owner_filter, require_user
from app.schemas import SessionDetail, SessionSummary
from app.security.redact import redact_text, redact_value
from app.storage.repository import UserRecord
from app.storage.session_store import SessionStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("", response_model=list[SessionSummary])
async def list_sessions(
    user: UserRecord = Depends(require_user),
    store: SessionStore = Depends(get_store),
) -> list[SessionSummary]:
    """历史会话列表，按最近活跃倒序。不含完整对话，只带摘要。

    只返回当前用户自己的会话 —— 认证不是装饰，归属过滤必须落在查询上。
    """
    summaries: list[SessionSummary] = []
    for session in store.list_sessions(owner_id=owner_filter(user)):
        summaries.append(
            SessionSummary(
                session_id=session.session_id,
                filename=session.filename,
                sheets=list(session.sheets),
                created_at=session.created_at,
                updated_at=session.updated_at,
                turn_count=session.turn_count,
                size_bytes=session.size_bytes,
                # DA-015：preview 来自用户提问原文（可能含手机号等），对外必须脱敏
                preview=redact_text(session.last_question()),
            )
        )
    logger.info("列出会话 %d 个", len(summaries))
    return summaries


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session_detail(
    session_id: str,
    user: UserRecord = Depends(require_user),
    store: SessionStore = Depends(get_store),
) -> SessionDetail:
    """会话详情：含完整消息序列，用于刷新后恢复界面。

    归属校验统一走 :func:`~app.dependencies.ensure_owner`（不是自己的会话按 404）。
    """
    session = ensure_owner(store.get(session_id), user)
    records = store.repo.get_messages(session_id)
    return SessionDetail(
        session_id=session.session_id,
        filename=session.filename,
        sheets=list(session.sheets),
        created_at=session.created_at,
        updated_at=session.updated_at,
        turn_count=session.turn_count,
        size_bytes=session.size_bytes,
        # DA-015：内存里的 messages 是**喂给模型的原文**（tool 输出含明文手机号等）。
        # 导出链路已经脱敏，详情接口若原样返回，等于开了一个绕过脱敏的后门 ——
        # 所以这里在出口统一过一遍，与界面/导出保持同一口径。
        messages=[redact_value(m) for m in session.messages],
        last_step_count=len(records[-1].steps) if records and records[-1].steps else 0,
    )

