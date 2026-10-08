"""SQLite 实现 —— **整个项目里唯一允许出现 SQL 的地方**。

若将来换 PostgreSQL，只需照抄本文件的结构实现
:class:`~app.storage.repository.SessionRepository`，业务代码不动。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.storage.repository import MessageRecord, SessionRecord

logger = logging.getLogger(__name__)

# 建表语句刻意保持在 SQLite / PostgreSQL 的共同子集内：
# - 主键由应用层给（TEXT），不用 AUTOINCREMENT / IDENTITY
# - 时间与 JSON 都用 TEXT，避开方言差异
# - 不用 INSERT OR IGNORE（SQLite 专有），统一用标准 ON CONFLICT
_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    filename   TEXT NOT NULL,
    file_path  TEXT NOT NULL,
    sheets     TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    turn_count INTEGER NOT NULL DEFAULT 0,
    owner_id   TEXT
)
"""

_MESSAGES_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    payload    TEXT NOT NULL,
    steps_json TEXT,
    created_at TEXT NOT NULL
)
"""

_INDEX_SCHEMA = """
CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages (session_id, created_at)
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class SqliteSessionRepository:
    """SQLite 会话仓库。

    并发说明：FastAPI 的同步路由会跑在线程池里，因此连接必须
    ``check_same_thread=False``，同时用一把进程内锁串行化写入。
    单进程下这足够；v1 明确不跑多 worker，所以无需更重的跨进程协调。
    """

    def __init__(self, db_path: Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self._path), check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        self._configure()
        self._ensure_schema()

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------- 基础设施
    def _configure(self) -> None:
        """SQLite 专有的运行参数（属于配置，不是 schema，可安全替换）。"""
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")      # 读写不互斥
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA busy_timeout=5000")     # 锁等待而非立即报错

    def _ensure_schema(self) -> None:
        with self._lock:
            self._conn.execute(_SCHEMA)
            self._conn.execute(_MESSAGES_SCHEMA)
            self._conn.execute(_INDEX_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ----------------------------------------------------------------- 会话
    def upsert_session(self, record: SessionRecord) -> None:
        sql = """
            INSERT INTO sessions
                (session_id, filename, file_path, sheets,
                 created_at, updated_at, turn_count, owner_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (session_id) DO UPDATE SET
                filename   = excluded.filename,
                file_path  = excluded.file_path,
                sheets     = excluded.sheets,
                updated_at = excluded.updated_at,
                turn_count = excluded.turn_count,
                owner_id   = excluded.owner_id
        """
        with self._lock:
            self._conn.execute(
                sql,
                (
                    record.session_id,
                    record.filename,
                    record.file_path,
                    json.dumps(record.sheets, ensure_ascii=False),
                    record.created_at,
                    record.updated_at,
                    int(record.turn_count),
                    record.owner_id,
                ),
            )

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return _row_to_session(row) if row else None

    def list_sessions(
        self, limit: int = 200, owner_id: str | None = None
    ) -> list[SessionRecord]:
        """列出会话。

        ``owner_id`` 为 None 表示**不过滤**（认证关闭时用）；给出用户 id 时
        只返回该用户的会话。认证开启后调用方总会传真实 id，因此 None 不会
        造成「看到别人会话」的风险。
        """
        sql = "SELECT * FROM sessions"
        params: tuple = ()
        if owner_id is not None:
            sql += " WHERE owner_id = ?"
            params = (owner_id,)
        sql += " ORDER BY updated_at DESC, created_at DESC LIMIT ?"
        with self._lock:
            rows = self._conn.execute(sql, (*params, int(limit))).fetchall()
        return [_row_to_session(row) for row in rows]

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )
            # 级联删消息。V1 不启用 FOREIGN KEY 级联，这里显式删除，
            # 避免依赖 PRAGMA 设置（换库时行为一致）
            self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        return cursor.rowcount > 0

    def count_sessions(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()
        return int(row["n"]) if row else 0

    def claim_orphan_sessions(self, owner_id: str) -> int:
        """把 owner_id 为空的分析会话归给指定用户，返回迁移条数。

        只在首次创建管理员后执行一次：这批会话产生于「无登录」时代，
        本来就是部署者一个人的历史数据，不该被丢弃。
        """
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE sessions SET owner_id = ?"
                " WHERE owner_id IS NULL OR owner_id = ''",
                (owner_id,),
            )
        return cursor.rowcount

    # ----------------------------------------------------------------- 消息
    def replace_messages(self, session_id: str, messages: list[MessageRecord]) -> None:
        """整体替换。序列最多几十条，代价可忽略，比增量追踪更不容易漂移。"""
        delete_sql = "DELETE FROM messages WHERE session_id = ?"
        insert_sql = """
            INSERT INTO messages
                (message_id, session_id, role, content, payload, steps_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                self._conn.execute(delete_sql, (session_id,))
                for message in messages:
                    self._conn.execute(
                        insert_sql,
                        (
                            message.message_id or uuid.uuid4().hex,
                            session_id,
                            message.role,
                            message.content,
                            json.dumps(message.payload, ensure_ascii=False, default=str),
                            json.dumps(message.steps, ensure_ascii=False, default=str)
                            if message.steps
                            else None,
                            message.created_at or utc_now(),
                        ),
                    )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def get_messages(self, session_id: str, limit: int | None = None) -> list[MessageRecord]:
        sql = (
            "SELECT * FROM messages WHERE session_id = ?"
            " ORDER BY created_at, rowid"
        )
        params: tuple = (session_id,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (session_id, int(limit))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_message(row) for row in rows]

    def last_user_messages(self, session_ids: list[str]) -> dict[str, str]:
        """一次查询取多个会话的最后一条用户提问（历史列表用）。"""
        if not session_ids:
            return {}
        placeholders = ", ".join("?" for _ in session_ids)
        sql = f"""
            SELECT session_id, content FROM messages
            WHERE role = 'user' AND session_id IN ({placeholders})
            ORDER BY session_id, created_at, rowid
        """
        with self._lock:
            rows = self._conn.execute(sql, tuple(session_ids)).fetchall()
        # 同一会话可能有多条 user 消息，后一条覆盖前一条即为「最后一条」
        return {row["session_id"]: row["content"] for row in rows}


# --------------------------------------------------------------------- 行映射
def _loads(raw: Any, fallback: Any) -> Any:
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return fallback


def _row_to_session(row: sqlite3.Row) -> SessionRecord:
    sheets = _loads(row["sheets"], [])
    return SessionRecord(
        session_id=row["session_id"],
        filename=row["filename"],
        file_path=row["file_path"],
        sheets=sheets if isinstance(sheets, list) else [],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        turn_count=int(row["turn_count"]),
        owner_id=row["owner_id"],
    )


def _row_to_message(row: sqlite3.Row) -> MessageRecord:
    return MessageRecord(
        message_id=row["message_id"],
        session_id=row["session_id"],
        role=row["role"],
        # content 由 payload 自动派生，不从 content 列反填 ——
        # 派生是单向的，避免两个来源不一致时静默出错
        payload=_loads(row["payload"], {}),
        steps=_loads(row["steps_json"], None),
        created_at=row["created_at"],
    )
