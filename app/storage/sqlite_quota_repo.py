"""配额与用量的 SQLite 实现（DA-017）。

与 :mod:`app.storage.sqlite_repo` 同级：**SQL 只出现在 DAO 层**，
业务代码只依赖 :class:`~app.storage.repository.QuotaRepository` 协议。

并发说明与 :class:`~app.storage.sqlite_repo.SqliteSessionRepository` 一致：
``check_same_thread=False`` + 进程内锁串行化写；WAL 模式下读写互不阻塞。
但请注意：本表的**语义**要求「先自增、后判断」，所以自增与判断必须由
调用方在同一段**没有 await** 的同步代码里完成（见 ``QuotaService.reserve``）。
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

from app.storage.repository import DailyUsageRecord
from app.storage.sqlite_repo import utc_now

logger = logging.getLogger(__name__)

# 建表语句保持在 SQLite / PostgreSQL 的共同子集内（同 sqlite_repo 的约定）：
# 联合主键由应用层给、计数用 INTEGER、时间用 ISO8601 文本
_SCHEMA = """
CREATE TABLE IF NOT EXISTS quota_daily_usage (
    user_key             TEXT NOT NULL,
    day                  TEXT NOT NULL,
    requests             INTEGER NOT NULL DEFAULT 0,
    prompt_tokens        INTEGER NOT NULL DEFAULT 0,
    completion_tokens    INTEGER NOT NULL DEFAULT 0,
    total_tokens         INTEGER NOT NULL DEFAULT 0,
    usage_reported_calls INTEGER NOT NULL DEFAULT 0,
    usage_missing_calls  INTEGER NOT NULL DEFAULT 0,
    updated_at           TEXT NOT NULL,
    PRIMARY KEY (user_key, day)
)
"""


class SqliteQuotaRepository:
    """每日请求计数 + token 用量的仓库。"""

    def __init__(self, db_path: Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self._path), check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        self._configure()
        self._conn.execute(_SCHEMA)

    @property
    def path(self) -> Path:
        return self._path

    def _configure(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ 读
    def get_daily(self, user_key: str, day: str) -> DailyUsageRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM quota_daily_usage WHERE user_key = ? AND day = ?",
                (user_key, day),
            ).fetchone()
        return _row_to_record(row) if row else None

    # ------------------------------------------------------------------ 写
    def increment_requests(self, user_key: str, day: str) -> int:
        """当日请求数 +1，返回自增后的值。

        ``ON CONFLICT ... DO UPDATE SET requests = requests + 1`` 是**原子**的：
        自增在数据库里完成，不存在「两个调用方读到同一个旧值」的窗口。
        """
        now = utc_now()
        sql = """
            INSERT INTO quota_daily_usage (user_key, day, requests, updated_at)
            VALUES (?, ?, 1, ?)
            ON CONFLICT (user_key, day) DO UPDATE SET
                requests   = requests + 1,
                updated_at = excluded.updated_at
        """
        with self._lock:
            self._conn.execute(sql, (user_key, day, now))
            row = self._conn.execute(
                "SELECT requests FROM quota_daily_usage WHERE user_key = ? AND day = ?",
                (user_key, day),
            ).fetchone()
        return int(row["requests"]) if row else 0

    def add_tokens(
        self,
        user_key: str,
        day: str,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        reported_calls: int = 0,
        missing_calls: int = 0,
    ) -> None:
        """累加 token 用量。全部按增量相加，重复结算不会覆盖历史。"""
        now = utc_now()
        sql = """
            INSERT INTO quota_daily_usage
                (user_key, day, requests, prompt_tokens, completion_tokens,
                 total_tokens, usage_reported_calls, usage_missing_calls, updated_at)
            VALUES (?, ?, 0, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (user_key, day) DO UPDATE SET
                prompt_tokens        = prompt_tokens + excluded.prompt_tokens,
                completion_tokens    = completion_tokens + excluded.completion_tokens,
                total_tokens         = total_tokens + excluded.total_tokens,
                usage_reported_calls = usage_reported_calls + excluded.usage_reported_calls,
                usage_missing_calls  = usage_missing_calls + excluded.usage_missing_calls,
                updated_at           = excluded.updated_at
        """
        with self._lock:
            self._conn.execute(
                sql,
                (
                    user_key,
                    day,
                    max(0, int(prompt_tokens)),
                    max(0, int(completion_tokens)),
                    max(0, int(total_tokens)),
                    max(0, int(reported_calls)),
                    max(0, int(missing_calls)),
                    now,
                ),
            )


def _row_to_record(row: sqlite3.Row) -> DailyUsageRecord:
    return DailyUsageRecord(
        user_key=row["user_key"],
        day=row["day"],
        requests=int(row["requests"]),
        prompt_tokens=int(row["prompt_tokens"]),
        completion_tokens=int(row["completion_tokens"]),
        total_tokens=int(row["total_tokens"]),
        usage_reported_calls=int(row["usage_reported_calls"]),
        usage_missing_calls=int(row["usage_missing_calls"]),
        updated_at=row["updated_at"],
    )
