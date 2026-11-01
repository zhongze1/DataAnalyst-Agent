"""会话仓库：**内存只是缓存，SQLite 才是真相**。

改造前：``self._sessions`` 既是索引也是数据，进程一停对话全丢。
改造后：
    - 元数据与消息 → :class:`SessionRepository`（持久化）
    - ``Session`` 对象与 DataFrame → 进程内缓存，按需从 repository 重建

上传的 Excel 仍留在磁盘，因为 pandas 需要真实文件路径。
"""

from __future__ import annotations

import logging
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import UPLOAD_DIR, get_settings
from app.errors import (
    FileEmptyError,
    FileParseError,
    FileTooLargeError,
    FileTypeError,
)
from app.storage.repository import MessageRecord, SessionRecord, SessionRepository
from app.storage.sqlite_repo import SqliteSessionRepository, utc_now

logger = logging.getLogger(__name__)

META_NAME = "meta.json"  # 仅用于旧数据迁移，新代码不再写


@dataclass
class Session:
    """一次分析会话的内存视图。

    ``frames`` 是 DataFrame 缓存，**不持久化** —— 可从 :attr:`path` 重建，
    属于缓存而非数据。``messages`` 与 :attr:`path` 一起构成「当前工作集」。
    """

    session_id: str
    filename: str
    path: Path
    sheets: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    turn_count: int = 0
    owner_id: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def size_bytes(self) -> int:
        try:
            return self.path.stat().st_size
        except OSError:
            return 0

    def last_question(self, limit: int = 40) -> str:
        """最后一条用户提问的摘要，用于历史列表展示。"""
        for message in reversed(self.messages):
            if message.get("role") == "user":
                text = str(message.get("content", "")).strip().replace("\n", " ")
                return text[:limit]
        return ""


class SessionStore:
    """会话仓库：元数据与消息持久化，文件与 DataFrame 走内存缓存。

    读路径：内存缓存 → repository → 重建 Session。
    写路径：repository（持久）+ 更新内存缓存。
    """

    def __init__(
        self,
        root: Path | None = None,
        repo: SessionRepository | None = None,
    ) -> None:
        self._root = Path(root or get_settings().upload_dir or UPLOAD_DIR)
        self._root.mkdir(parents=True, exist_ok=True)
        self._repo = repo if repo is not None else SqliteSessionRepository(get_settings().db_path)
        self._cache: dict[str, Session] = {}
        self._warm_cache()

    @property
    def repo(self) -> SessionRepository:
        return self._repo

    def _warm_cache(self) -> None:
        """启动时把会话索引进内存（不预热 DataFrame 与全部消息）。"""
        for record in self._repo.list_sessions(limit=10_000):
            if not Path(record.file_path).exists():
                logger.warning("会话文件缺失，跳过: %s", record.file_path)
                continue
            self._cache[record.session_id] = _session_from_record(record)
        logger.info("已从数据库恢复 %d 个会话", len(self._cache))

    # ------------------------------------------------------------- 查询
    def get(self, session_id: str | None) -> Session | None:
        """取会话。**必须保持对象身份**。

        Agent Loop 直接往 ``session.messages`` 上追加内容，调用方（如测试
        fixture）也持有同一个对象。如果这里在缓存命中时「顺手」新建对象替换
        缓存，写入就会落到一个没人引用的对象上 —— 曾经踩过这个坑。
        因此：缓存命中就复用同一实例，只在需要时把消息**灌进**它。
        """
        if not session_id:
            return None
        session = self._cache.get(session_id)
        if session is None:
            record = self._repo.get_session(session_id)
            if record is None:
                return None
            if not Path(record.file_path).exists():
                logger.warning("会话文件缺失: %s", record.file_path)
                return None
            session = _session_from_record(record)
            self._cache[session_id] = session
        if not session.messages:
            # 消息可能是空的（新会话，或进程重启后首次访问）—— 从库里补齐
            loaded = _restore_messages(self._repo.get_messages(session_id))
            if loaded:
                session.messages = loaded
        return session

    def list_sessions(self, owner_id: str | None = None) -> list[Session]:
        """历史会话列表，按最近活跃倒序。走数据库，不扫目录。

        ``owner_id`` 为 None 表示不过滤（认证关闭的本机模式）；开启认证后
        调用方必须传当前用户 id，否则会看到别人的会话。
        """
        records = self._repo.list_sessions(owner_id=owner_id)
        previews = self._repo.last_user_messages([r.session_id for r in records])
        sessions: list[Session] = []
        for record in records:
            # 用副本，列表页的摘要不能污染缓存里的真实会话对象
            session = _session_from_record(record)
            preview = previews.get(record.session_id, "")
            if preview:
                session.messages = [{"role": "user", "content": preview}]
            sessions.append(session)
        return sessions

    # ------------------------------------------------------------- 写入
    def create(self, filename: str, content: bytes, owner_id: str | None = None) -> Session:
        """落盘一份上传文件并在数据库建立会话。

        ``owner_id`` 在认证开启时由路由层从当前登录用户传入。
        """
        settings = get_settings()
        suffix = Path(filename).suffix.lower()
        if suffix not in settings.allowed_suffixes:
            raise FileTypeError(
                f"暂不支持的文件类型 {suffix or '（无后缀）'}，"
                f"仅支持: {', '.join(settings.allowed_suffixes)}"
            )
        if not content:
            raise FileEmptyError()
        if len(content) > settings.max_upload_bytes:
            raise FileTooLargeError(settings.max_upload_bytes)

        session_id = uuid.uuid4().hex[:12]
        folder = self._root / session_id
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / Path(filename).name
        target.write_bytes(content)

        try:
            sheets = self.list_sheet_names(target)
        except Exception as exc:
            # 解析失败则清理目录，不留半截会话。
            # 真实原因（openpyxl 报错细节）只进日志，不进响应体。
            target.unlink(missing_ok=True)
            folder.rmdir()
            logger.warning("Excel 解析失败 | 文件名=%s | 原因=%s", filename, exc, exc_info=True)
            raise FileParseError(cause=exc) from exc

        now = utc_now()
        record = SessionRecord(
            session_id=session_id,
            filename=target.name,
            file_path=str(target),
            sheets=sheets,
            created_at=now,
            updated_at=now,
            turn_count=0,
            owner_id=owner_id,
        )
        try:
            self._repo.upsert_session(record)
        except Exception:
            # 入库失败要把文件收回去，否则磁盘上留孤儿文件
            target.unlink(missing_ok=True)
            folder.rmdir()
            logger.exception("会话入库失败")
            raise
        session = _session_from_record(record)
        self._cache[session_id] = session
        logger.info("新会话 %s：%s（%s）", session_id, session.filename, sheets)
        return session

    def save_turn(self, session: Session, steps: list[dict[str, Any]] | None = None) -> None:
        """一轮对话结束后持久化：消息全量替换 + 会话元数据更新。

        整体替换而非增量追加：消息最多几十条，代价可忽略，而增量追踪
        迟早会因为中途异常导致顺序漂移。
        """
        if not session.messages:
            return
        turn_count = sum(1 for m in session.messages if m.get("role") == "user")
        self._repo.replace_messages(
            session.session_id, _to_records(session, steps)
        )
        session.turn_count = turn_count
        session.updated_at = utc_now()
        self._repo.upsert_session(_to_record(session))

    def messages_for_export(self, session: Session) -> list[dict[str, Any]]:
        """取**带工具轨迹**的消息序列，供导出报告使用。

        必须从数据库读而不是用 ``session.messages``：轨迹快照（``steps``）只存在
        数据库的 ``steps_json`` 列里，内存副本为了不污染模型请求是不带的。
        早前导出直接读内存副本，于是「执行轨迹」永远是空的 —— 报告号称有轨迹，
        实际只有结论和数据明细，而且没有任何报错，属于静默缺功能。
        """
        return _records_to_export_messages(self._repo.get_messages(session.session_id))

    def claim_orphan_sessions(self, owner_id: str) -> int:
        """把无主会话归给指定用户，并**同步刷新内存缓存**。

        必须走 store 而不是直接调 repo：``get()`` 命中缓存时不会重读数据库，
        只改库会导致缓存里的 owner_id 一直是旧值（认证过滤随之失效）。
        """
        claimed = self._repo.claim_orphan_sessions(owner_id)
        if claimed:
            for session in self._cache.values():
                if not session.owner_id:
                    session.owner_id = owner_id
        return claimed

    def reset_messages(self, session: Session) -> None:
        """清空对话历史（保留已上传的数据与文件）。"""
        session.messages.clear()
        self._repo.replace_messages(session.session_id, [])
        session.turn_count = 0
        session.updated_at = utc_now()
        self._repo.upsert_session(_to_record(session))

    def delete(self, session_id: str) -> bool:
        session = self._cache.pop(session_id, None)
        record = self._repo.get_session(session_id)
        path = Path(record.file_path) if record else (session.path if session else None)
        deleted = self._repo.delete_session(session_id)
        if path is not None:
            folder = path.parent
            try:
                # 只删 uploads 下的一级子目录，避免误伤用户手动放的文件
                if folder.is_dir() and folder.parent == self._root:
                    # 会话目录里可能有 artifacts/ 子目录，unlink 对它无效，
                    # 会抛 IsADirectoryError 并让整个清理被 except 吞掉
                    for child in folder.glob("*"):
                        if child.is_dir():
                            shutil.rmtree(child, ignore_errors=True)
                        else:
                            child.unlink(missing_ok=True)
                    folder.rmdir()
            except OSError:
                logger.warning("删除会话文件失败: %s", folder, exc_info=True)
        return deleted or session is not None

    # ------------------------------------------------------------- 工具方法
    @staticmethod
    def list_sheet_names(path: Path) -> list[str]:
        with pd.ExcelFile(path) as excel:
            return list(excel.sheet_names)

    @staticmethod
    def read_frame(path: str | Path, sheet: str) -> pd.DataFrame:
        """读取单个工作表为 DataFrame（由工具层按需调用）。"""
        return pd.read_excel(path, sheet_name=sheet)


# --------------------------------------------------------------------- 转换
def _to_record(session: Session) -> SessionRecord:
    return SessionRecord(
        session_id=session.session_id,
        filename=session.filename,
        file_path=str(session.path),
        sheets=list(session.sheets),
        created_at=session.created_at,
        updated_at=session.updated_at,
        turn_count=session.turn_count,
        owner_id=session.owner_id,
    )


def _session_from_record(record: SessionRecord) -> Session:
    return Session(
        session_id=record.session_id,
        filename=record.filename,
        path=Path(record.file_path),
        sheets=list(record.sheets),
        created_at=record.created_at,
        updated_at=record.updated_at,
        turn_count=record.turn_count,
        owner_id=record.owner_id,
    )


def _to_records(session: Session, steps: list[dict[str, Any]] | None) -> list[MessageRecord]:
    """消息 → 待落库记录；轨迹快照挂在最后一条 assistant 上。

    ``payload`` 存完整消息：assistant 的 tool_calls、tool 的 tool_call_id
    都在里面，缺了会导致恢复会话后模型接口报错。
    """
    last_assistant = max(
        (i for i, m in enumerate(session.messages) if m.get("role") == "assistant"),
        default=-1,
    )
    records: list[MessageRecord] = []
    for index, message in enumerate(session.messages):
        records.append(
            MessageRecord(
                message_id=uuid.uuid4().hex,
                session_id=session.session_id,
                role=str(message.get("role", "user")),
                payload=dict(message),
                steps=steps if (index == last_assistant and steps) else None,
                created_at=utc_now(),
            )
        )
    return records


def _restore_messages(records: list[MessageRecord]) -> list[dict[str, Any]]:
    """数据库记录 → Agent Loop 需要的消息列表（无损）。"""
    return [dict(record.payload) for record in records if record.payload]


def _records_to_export_messages(records: list[MessageRecord]) -> list[dict[str, Any]]:
    """数据库记录 → 导出用消息列表（**带 steps**）。

    与 :func:`_restore_messages` 的唯一区别：把 ``steps_json`` 里的轨迹快照
    合回 ``payload``。

    为什么单独做一个函数而不是直接改 ``_restore_messages``：``steps`` 不是
    OpenAI 消息格式的字段，混进 ``session.messages`` 就会被原样发给模型，
    严格的厂商会直接报 400。所以「喂给模型的」和「给人看的」分成两条路径，
    只在导出这条路径上补 steps。
    """
    messages: list[dict[str, Any]] = []
    for record in records:
        if not record.payload:
            continue
        message = dict(record.payload)
        if record.steps:
            message["steps"] = record.steps
        messages.append(message)
    return messages
