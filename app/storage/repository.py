"""会话持久化的抽象层。

**分层纪律**：SQL 只允许出现在 ``sqlite_repo.py``（DAO）里。业务代码
（router / agent / 前端）只依赖 :class:`SessionRepository` 协议与这里的
记录类型。将来换 PostgreSQL 只需新增一个实现，接口与业务代码零改动。

**SQL 可移植性约定**（为 SQLite ⇄ PostgreSQL 互换留出空间）：
- 主键由应用层生成 UUID，不用 ``AUTOINCREMENT`` / ``IDENTITY``（方言不同）
- upsert 一律用标准 ``INSERT ... ON CONFLICT ... DO UPDATE``（SQLite 3.24+ 与 PG 都支持）
- 布尔用 INTEGER 0/1，时间用 ISO8601 文本，JSON 用 TEXT 序列化
- 避开任何单库专有语法（``INSERT OR IGNORE``、``PRAGMA``、``JSONB`` 等）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class SessionRecord:
    """会话的持久化元数据（不含 DataFrame —— 那是缓存，不是数据）。"""

    session_id: str
    filename: str
    file_path: str
    sheets: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    turn_count: int = 0
    # 为 W2-1.5 用户体系预留：当前恒为 None，届时填 user_id，无需改表
    owner_id: str | None = None

    @property
    def size_bytes(self) -> int:
        from pathlib import Path

        try:
            return Path(self.file_path).stat().st_size
        except OSError:
            return 0


@dataclass
class MessageRecord:
    """一条对话消息。

    ``payload`` 是**完整消息的 JSON**，恢复会话时原样回灌给模型。
    这一点很关键：OpenAI 格式里 assistant 消息带 ``tool_calls``、tool 消息
    带 ``tool_call_id``，缺任何一个，重放时模型接口就会报错。
    只存 role + content 会丢掉这两个字段，因此 payload 是无损的真相。

    ``content`` 单独冗余一列，是为将来做全文检索留的索引位。
    """

    message_id: str
    session_id: str
    role: str
    payload: dict[str, Any] = field(default_factory=dict)
    # assistant 消息的工具轨迹快照：历史列表要显示「上次用了几步工具」，
    # 有快照就不必重放整段对话去数
    steps: list[dict[str, Any]] | None = None
    created_at: str = ""
    # content 恒为 payload 的镜像，不需要调用方填写（见 __post_init__）
    content: str = field(default="", init=False)

    def __post_init__(self) -> None:
        """从 payload 派生 content。

        之前 content 是与 payload 并列的可选位置参数，调用方很容易只传
        payload 而漏掉 content，于是静默存进空字符串 —— 自己的测试就踩了
        这个坑。派生之后这个错误就不可能发生了。
        """
        raw = self.payload.get("content") if isinstance(self.payload, dict) else None
        self.content = "" if raw is None else str(raw)


@dataclass
class DailyUsageRecord:
    """某个用户在某一天（配额时区下的自然日）的用量。

    既承载**硬配额**（``requests``），也承载**统计**（token 用量）。
    两者放在一行是为了避免两次写入，且它们的生命周期完全一致。
    """

    user_key: str
    day: str
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # 回报了 usage 的模型调用轮次
    usage_reported_calls: int = 0
    # 未回报 usage 的轮次。厂商没给就不给，**不估算、不伪装成精确值**
    usage_missing_calls: int = 0
    updated_at: str = ""


@dataclass
class UserRecord:
    """用户。``password_hash`` 用 :mod:`app.security.passwords` 的编码格式。"""

    user_id: str
    username: str
    password_hash: str
    display_name: str = ""
    role: str = "analyst"
    disabled: bool = False
    created_at: str = ""

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


@dataclass
class AuthSessionRecord:
    """登录会话。

    ``token_hash`` 是 Cookie 令牌的 SHA-256，**不是明文**；数据库泄露时
    攻击者拿不到可直接使用的凭据。
    """

    token_hash: str
    user_id: str
    created_at: str
    expires_at: str
    user_agent: str = ""


@runtime_checkable
class AuthRepository(Protocol):
    """用户与登录会话的持久化接口。SQL 同样只允许出现在 DAO 层。"""

    # ---------------------------------------------------------- 用户
    def create_user(self, record: UserRecord) -> UserRecord: ...
    def get_user(self, user_id: str) -> UserRecord | None: ...
    def get_user_by_username(self, username: str) -> UserRecord | None: ...
    def list_users(self) -> list[UserRecord]: ...
    def count_users(self) -> int: ...
    def update_password(self, user_id: str, password_hash: str) -> None: ...
    def set_disabled(self, user_id: str, disabled: bool) -> None: ...

    # ---------------------------------------------------------- 登录会话
    def create_auth_session(self, record: AuthSessionRecord) -> None: ...
    def get_auth_session(self, token_hash: str) -> AuthSessionRecord | None: ...
    def delete_auth_session(self, token_hash: str) -> None: ...
    def delete_user_sessions(self, user_id: str, *, keep: str | None = None) -> int: ...
    def purge_expired_sessions(self) -> int: ...


@runtime_checkable
class SessionRepository(Protocol):
    """会话与消息的持久化接口。

    同步接口是有意的取舍：SQLite 的同步调用在微秒级，换 PostgreSQL 时
    可以用 psycopg2 的同步连接 + 连接池，接口不变；若要上 asyncpg
    才需要把协议改成 async —— 那是一次有意识的重构，不该被提前引入。
    """

    # ---------------------------------------------------------- 会话
    def upsert_session(self, record: SessionRecord) -> None:
        """写入或更新会话元数据（按 session_id 覆盖）。"""
        ...

    def get_session(self, session_id: str) -> SessionRecord | None: ...

    def list_sessions(self, limit: int = 200) -> list[SessionRecord]:
        """按最近活跃时间倒序返回。"""
        ...

    def delete_session(self, session_id: str) -> bool: ...

    def count_sessions(self) -> int: ...

    def claim_orphan_sessions(self, owner_id: str) -> int:
        """把 owner_id 为空的分析会话归给指定用户，返回迁移条数。"""
        ...

    # ---------------------------------------------------------- 消息
    def replace_messages(self, session_id: str, messages: list[MessageRecord]) -> None:
        """整体替换某会话的消息序列（幂等，避免增量追踪的漂移）。"""
        ...

    def get_messages(self, session_id: str, limit: int | None = None) -> list[MessageRecord]:
        """按时间正序返回消息。"""
        ...

    def last_user_messages(self, session_ids: list[str]) -> dict[str, str]:
        """批量取每个会话最后一条用户提问的摘要（历史列表用，避免 N+1 查询）。"""
        ...


@runtime_checkable
class QuotaRepository(Protocol):
    """配额与用量的持久化接口（DA-017）。

    只负责「每日计数」这一类的读写；分钟窗口与当前并发属于进程内状态，
    不进这里 —— 它们是短期、可容忍重启清零的量。
    """

    def get_daily(self, user_key: str, day: str) -> DailyUsageRecord | None:
        """取某用户某自然日的用量记录，没有则返回 None。"""
        ...

    def increment_requests(self, user_key: str, day: str) -> int:
        """当日请求数 +1 并返回**自增后**的值。

        必须是单向自增：调用方据此判断是否超限，
        若先读后写，两个并发请求会同时读到同一个旧值。
        """
        ...

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
        """累加 token 用量与「是否回报 usage」的轮次统计。"""
        ...
