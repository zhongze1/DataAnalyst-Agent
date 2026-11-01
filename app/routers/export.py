"""报告导出接口（W2-2）。

```
GET /api/sessions/{sid}/export?format=md|xlsx&last_turn=false
```

归属校验复用 W2-1.5 的规则：不是自己的会话按「不存在」返回 404。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from urllib.parse import quote

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response

from app.dependencies import ensure_owner, get_store, require_user
from app.errors import BadRequestError
from app.services.export_service import (
    build_markdown,
    build_xlsx,
    collect_turns,
)
from app.storage.artifacts import list_artifacts
from app.storage.repository import UserRecord
from app.storage.session_store import SessionStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sessions", tags=["export"])

EXPORT_FORMATS = {"md", "xlsx"}


@router.get("/{session_id}/export")
async def export_session(
    session_id: str,
    format: str = Query("md", description="md | xlsx"),
    last_turn: bool = Query(False, description="只导出最后一轮结论"),
    user: UserRecord = Depends(require_user),
    store: SessionStore = Depends(get_store),
) -> Response:
    """导出分析报告（结论 + 执行轨迹 + 数据明细）。"""
    fmt = (format or "md").lower()
    if fmt not in EXPORT_FORMATS:
        raise BadRequestError(f"不支持的导出格式：{format}。可用：{', '.join(sorted(EXPORT_FORMATS))}")

    session = ensure_owner(store.get(session_id), user)

    # 必须用带 steps 的版本：轨迹快照只在数据库里，内存副本不带（见 messages_for_export）
    messages = store.messages_for_export(session)
    turns = collect_turns(messages)
    artifacts = list_artifacts(session.path)
    generated_at = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")

    common = {
        "filename": session.filename,
        "session_id": session.session_id,
        "turns": turns,
        "artifacts": artifacts,
        "session_path": session.path,
        "generated_at": generated_at,
        "only_last_turn": last_turn,
    }
    bundle = build_markdown(**common) if fmt == "md" else build_xlsx(**common)

    logger.info(
        "导出报告 | 会话=%s | 格式=%s | 轮次=%d | 产物=%d",
        session.session_id, fmt, len(turns), len(artifacts),
    )
    # 中文文件名走 RFC 5987 编码，否则部分浏览器会乱码
    quoted = quote(bundle.filename)
    return Response(
        content=bundle.content,
        media_type=bundle.media_type,
        headers={
            "Content-Disposition": (
                f"attachment; filename=report{'.md' if fmt == 'md' else '.xlsx'}; "
                f"filename*=UTF-8''{quoted}"
            ),
            "X-Artifact-Count": str(len(artifacts)),
        },
    )