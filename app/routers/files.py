"""文件相关接口：上传 Excel、查询/删除会话。"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, UploadFile

from app.context import set_session_id
from app.dependencies import get_store
from app.errors import SessionNotFoundError
from app.schemas import ERROR_RESPONSES, FileInfo, UploadResponse
from app.storage.session_store import SessionStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/files", tags=["files"], responses=ERROR_RESPONSES)


@router.post("/upload", response_model=UploadResponse)
async def upload_excel(
    file: UploadFile = File(..., description="Excel 文件（.xlsx/.xlsm）"),
    store: SessionStore = Depends(get_store),
) -> UploadResponse:
    """上传 Excel，返回 session_id 与工作表列表。后续提问都带这个 session_id。

    校验失败时抛 AppError 子类，由全局处理器转成统一错误响应体；
    文件解析失败的具体原因只写日志，不返回给客户端。
    """
    content = await file.read()
    # 抛出的 FileTypeError / FileEmptyError / FileTooLargeError / FileParseError
    # 会被全局处理器接住，这里不需要 try/except
    session = store.create(file.filename or "upload.xlsx", content)

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
        raise SessionNotFoundError()
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
    session = store.get(session_id)
    if session is None:
        raise SessionNotFoundError("会话不存在")
    store.delete(session_id)
    logger.info("会话已删除 | %s", session_id, extra={"deleted_session_id": session_id})
    return {"deleted": True}
