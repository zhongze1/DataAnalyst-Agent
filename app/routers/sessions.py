"""历史会话接口。

与 ``/api/files/*`` 的分工：那边管「上传的文件」，这里管「会话与对话」。
删除接口仍在 files.py（前端与既有测试都依赖它），此处不重复实现。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from app.dependencies import get_store
from app.errors import SessionNotFoundError
from app.schemas import SessionDetail, SessionSummary
from app.storage.session_store import SessionStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("", response_model=list[SessionSummary])
async def list_sessions(store: SessionStore = Depends(get_store)) -> list[SessionSummary]:
    """历史会话列表，按最近活跃倒序。不含完整对话，只带摘要。"""
    summaries: list[SessionSummary] = []
    for session in store.list_sessions():
        summaries.append(
            SessionSummary(
                session_id=session.session_id,
                filename=session.filename,
                sheets=list(session.sheets),
                created_at=session.created_at,
                updated_at=session.updated_at,
                turn_count=session.turn_count,
                size_bytes=session.size_bytes,
                preview=session.last_question(),
            )
        )
    logger.info("列出会话 %d 个", len(summaries))
    return summaries


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session_detail(
    session_id: str,
    store: SessionStore = Depends(get_store),
) -> SessionDetail:
    """会话详情：含完整消息序列，用于刷新后恢复界面。"""
    session = store.get(session_id)
    if session is None:
        raise SessionNotFoundError()
    records = store.repo.get_messages(session_id)
    return SessionDetail(
        session_id=session.session_id,
        filename=session.filename,
        sheets=list(session.sheets),
        created_at=session.created_at,
        updated_at=session.updated_at,
        turn_count=session.turn_count,
        size_bytes=session.size_bytes,
        messages=session.messages,
        last_step_count=len(records[-1].steps) if records and records[-1].steps else 0,
    )

