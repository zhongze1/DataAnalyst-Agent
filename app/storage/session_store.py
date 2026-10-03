"""会话与文件存储。

每个会话对应磁盘目录 ``data/uploads/<session_id>/``：
    - ``<原文件名>``  上传的 Excel 原文
    - ``meta.json``    会话元信息（文件名、工作表列表、创建时间）

DataFrame 按需加载并缓存在内存 ``Session.frames`` 中，进程重启后仍可继续提问。
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import UPLOAD_DIR, get_settings

logger = logging.getLogger(__name__)

META_NAME = "meta.json"


@dataclass
class Session:
    """一次分析会话：一份文件 + 一段对话历史。"""

    session_id: str
    filename: str
    path: Path
    sheets: list[str] = field(default_factory=list)
    created_at: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size if self.path.exists() else 0

    def to_meta(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "filename": self.filename,
            "sheets": self.sheets,
            "created_at": self.created_at,
        }


class SessionStore:
    """会话仓库。"""

    def __init__(self, root: Path | None = None) -> None:
        self._root = Path(root or get_settings().upload_dir or UPLOAD_DIR)
        self._root.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, Session] = {}
        self.load_all()

    # ------------------------------------------------------------- 查询
    def get(self, session_id: str | None) -> Session | None:
        if not session_id:
            return None
        return self._sessions.get(session_id)

    def list_sessions(self) -> list[Session]:
        return sorted(self._sessions.values(), key=lambda s: s.created_at, reverse=True)

    # ------------------------------------------------------------- 写入
    def create(self, filename: str, content: bytes) -> Session:
        """落盘一份上传文件并建立会话。"""
        settings = get_settings()
        suffix = Path(filename).suffix.lower()
        if suffix not in settings.allowed_suffixes:
            raise ValueError(
                f"暂不支持的文件类型 {suffix!r}，仅支持: {', '.join(settings.allowed_suffixes)}"
            )
        if not content:
            raise ValueError("上传内容为空。")
        if len(content) > settings.max_upload_bytes:
            limit_mb = settings.max_upload_bytes // (1024 * 1024)
            raise ValueError(f"文件超过 {limit_mb}MB 限制。")

        session_id = uuid.uuid4().hex[:12]
        folder = self._root / session_id
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / Path(filename).name
        target.write_bytes(content)

        try:
            sheets = self.list_sheet_names(target)
        except Exception as exc:  # 解析失败则清理目录，避免留下脏会话
            target.unlink(missing_ok=True)
            folder.rmdir()
            raise ValueError(f"无法解析 Excel 文件: {exc}") from exc

        session = Session(
            session_id=session_id,
            filename=target.name,
            path=target,
            sheets=sheets,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        (folder / META_NAME).write_text(
            json.dumps(session.to_meta(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._sessions[session_id] = session
        logger.info("新会话 %s：%s（%s）", session_id, session.filename, sheets)
        return session

    def reset_messages(self, session: Session) -> None:
        session.messages.clear()

    def delete(self, session_id: str) -> bool:
        session = self._sessions.pop(session_id, None)
        if not session:
            return False
        folder = self._root / session_id
        for child in folder.glob("*"):
            child.unlink(missing_ok=True)
        folder.rmdir()
        return True

    # ------------------------------------------------------------- 磁盘
    def load_all(self) -> None:
        """启动时从磁盘恢复会话索引（只恢复文件与工作表，不恢复对话历史）。"""
        if not self._root.exists():
            return
        for folder in self._root.iterdir():
            meta_file = folder / META_NAME
            if not folder.is_dir() or not meta_file.exists():
                continue
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                candidates = [
                    p for p in folder.iterdir()
                    if p.is_file() and p.name != META_NAME
                ]
                if not candidates:
                    continue
                path = candidates[0]
                self._sessions[meta["session_id"]] = Session(
                    session_id=meta["session_id"],
                    filename=meta.get("filename", path.name),
                    path=path,
                    sheets=meta.get("sheets", []),
                    created_at=meta.get("created_at", ""),
                )
            except Exception:  # pragma: no cover - 损坏的会话直接跳过
                logger.warning("跳过损坏会话目录: %s", folder)

    # ------------------------------------------------------------- 工具方法
    @staticmethod
    def list_sheet_names(path: Path) -> list[str]:
        with pd.ExcelFile(path) as excel:
            return list(excel.sheet_names)

    @staticmethod
    def read_frame(path: str | Path, sheet: str) -> pd.DataFrame:
        """读取单个工作表为 DataFrame（由工具层按需调用）。"""
        return pd.read_excel(path, sheet_name=sheet)
