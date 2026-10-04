"""文件相关接口：上传 Excel、查询文件信息。"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.context import set_session_id
from app.dependencies import get_store
from app.schemas import FileInfo, UploadResponse
from app.storage.session_store import SessionStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/files", tags=["files"])


@router.post("/upload", response_model=UploadResponse)
async def upload_excel(
    file: UploadFile = File(..., description="Excel 文件（.xlsx/.xlsm）"),
    store: SessionStore = Depends(get_store),
) -> UploadResponse:
    """上传 Excel，返回 session_id 与工作表列表。后续提问都带这个 session_id。"""
    content = await file.read()
    try:
        session = store.create(file.filename or "upload.xlsx", content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    set_session_id(session.session_id)
    logger.info(
        "文件上传成功 | %s | %.1f KB | 工作表 %d 个",
        session.filename,
        session.size_bytes / 1024,
        len(session.sheets),
        # 注意：extra 的键不能撞 LogRecord 内置属性（filename/module/message 等），
        # 否则 logging 会抛 KeyError 让整个请求 500
        extra={"file_name": session.filename, "size_bytes": session.size_bytes},
    )

    return UploadResponse(
        file=FileInfo(
            session_id=session.session_id,
            filename=session.filename,
            size_bytes=session.size_bytes,
            sheets=session.sheets,
        )
    )


@router.get("/{session_id}", response_model=FileInfo)
async def get_file_info(
    session_id: str,
    store: SessionStore = Depends(get_store),
) -> FileInfo:
    """查询某个会话的文件信息（刷新页面后可用来恢复上下文）。"""
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在或已过期，请重新上传文件。")
    return FileInfo(
        session_id=session.session_id,
        filename=session.filename,
        size_bytes=session.size_bytes,
        sheets=session.sheets,
    )


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    store: SessionStore = Depends(get_store),
) -> dict[str, bool]:
    session = store.delete(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在。")
    return {"deleted": True}
