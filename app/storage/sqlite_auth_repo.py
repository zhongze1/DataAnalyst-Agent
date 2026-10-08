"""认证数据访问层：users 与 auth_sessions。

与 :mod:`app.storage.sqlite_repo` 共用同一个数据库文件但各自持有连接 ——
SQLite 在 WAL 模式下支持多连接并发，加上 busy_timeout，单进程下无争用。

建表语句只用 SQLite / PostgreSQL 的共同子集：不用 AUTOINCREMENT、不用
INSERT OR IGNORE，时间与布尔都用 TEXT / INTEGER。username 走 UNIQUE 约束
而非「先查再插」——并发下后者会插出重名用户。
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

from app.storage.repository import AuthSessionRecord, UserRecord
from app.storage.sqlite_repo import utc_now

logger = logging.getLogger(__name__)

_USERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id       TEXT PRIMARY KEY,
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    display_name  TEXT,
    role          TEXT NOT NULL DEFAULT 'analyst',
    disabled      INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL
)
"""

_AUTH_SESSIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS auth_sessions (
    token_hash TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    user_agent TEXT
)
"""

_AUTH_INDEX_SCHEMA = """
CREATE INDEX IF NOT EXISTS idx_auth_sessions_user
    ON auth_sessions (user_id)
"""


class SqliteAuthRepository:
    """用户与登录会话的 DAO。"""

    def __init__(self, db_path: Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self._path), check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(_USERS_SCHEMA)
        self._conn.execute(_AUTH_SESSIONS_SCHEMA)
        self._conn.execute(_AUTH_INDEX_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ----------------------------------------------------------------- 用户
    def create_user(self, record: UserRecord) -> UserRecord:
        sql = """
            INSERT INTO users
                (user_id, username, password_hash, display_name,
                 role, disabled, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self._lock:
            self._conn.execute(
                sql,
                (
                    record.user_id,
                    record.username,
                    record.password_hash,
                    record.display_name or record.username,
                    record.role,
                    1 if record.disabled else 0,
                    record.created_at,
                ),
            )
        return record

    def get_user(self, user_id: str) -> UserRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
        return _row_to_user(row) if row else None

    def get_user_by_username(self, username: str) -> UserRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()
        return _row_to_user(row) if row else None

    def list_users(self) -> list[UserRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM users ORDER BY created_at, username"
            ).fetchall()
        return [_row_to_user(row) for row in rows]

    def count_users(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()
        return int(row["n"]) if row else 0

    def update_password(self, user_id: str, password_hash: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE users SET password_hash = ? WHERE user_id = ?",
                (password_hash, user_id),
            )

    def set_disabled(self, user_id: str, disabled: bool) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE users SET disabled = ? WHERE user_id = ?",
                (1 if disabled else 0, user_id),
            )

    # ------------------------------------------------------------- 登录会话
    def create_auth_session(self, record: AuthSessionRecord) -> None:
        sql = """
            INSERT INTO auth_sessions
                (token_hash, user_id, created_at, expires_at, user_agent)
            VALUES (?, ?, ?, ?, ?)
        """
        with self._lock:
            self._conn.execute(
                sql,
                (
                    record.token_hash,
                    record.user_id,
                    record.created_at,
                    record.expires_at,
                    record.user_agent,
                ),
            )

    def get_auth_session(self, token_hash: str) -> AuthSessionRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM auth_sessions WHERE token_hash = ?", (token_hash,)
            ).fetchone()
        return _row_to_auth_session(row) if row else None

    def delete_auth_session(self, token_hash: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM auth_sessions WHERE token_hash = ?", (token_hash,)
            )

    def delete_user_sessions(self, user_id: str, *, keep: str | None = None) -> int:
        """踢下线。``keep`` 保留当前这一条（改密码时保留当前会话）。"""
        sql = "DELETE FROM auth_sessions WHERE user_id = ?"
        params: tuple = (user_id,)
        if keep:
            sql += " AND token_hash != ?"
            params = (user_id, keep)
        with self._lock:
            cursor = self._conn.execute(sql, params)
        return cursor.rowcount

    def purge_expired_sessions(self) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM auth_sessions WHERE expires_at < ?", (utc_now(),)
            )
        return cursor.rowcount


# --------------------------------------------------------------------- 行映射
def _row_to_user(row: sqlite3.Row) -> UserRecord:
    return UserRecord(
        user_id=row["user_id"],
        username=row["username"],
        password_hash=row["password_hash"],
        display_name=row["display_name"] or row["username"],
        role=row["role"],
        disabled=bool(row["disabled"]),
        created_at=row["created_at"],
    )


def _row_to_auth_session(row: sqlite3.Row) -> AuthSessionRecord:
    return AuthSessionRecord(
        token_hash=row["token_hash"],
        user_id=row["user_id"],
        created_at=row["created_at"],
        expires_at=row["expires_at"],
        user_agent=row["user_agent"] or "",
    )