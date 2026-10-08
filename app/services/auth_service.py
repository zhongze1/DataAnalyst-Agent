"""认证服务：登录、登出、当前用户解析、初始管理员引导。

分层：路由只调这里，不直接碰 repository；口令哈希在
:mod:`app.security.passwords`，SQL 在 DAO 层。
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone

from app.config import Settings
from app.errors import AuthError, InvalidCredentialsError
from app.security.passwords import (
    generate_token,
    hash_password,
    needs_rehash,
    token_fingerprint,
    verify_password,
)
from app.storage.repository import AuthRepository, AuthSessionRecord, UserRecord
from app.storage.sqlite_auth_repo import SqliteAuthRepository

logger = logging.getLogger(__name__)

ROLE_ADMIN = "admin"
ROLE_ANALYST = "analyst"
VALID_ROLES = {ROLE_ADMIN, ROLE_ANALYST}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.isoformat(timespec="milliseconds")


# 用户不存在时也走一次同参数的哈希运算，用于拉平耗时、堵住用户枚举
_DUMMY_HASH = hash_password("timing-equalizer", iterations=1000)


class AuthService:
    """认证相关的全部业务逻辑。"""

    def __init__(self, settings: Settings, repo: AuthRepository | None = None) -> None:
        self._settings = settings
        self._repo = repo if repo is not None else SqliteAuthRepository(settings.db_path)

    @property
    def repo(self) -> AuthRepository:
        return self._repo

    @property
    def enabled(self) -> bool:
        return self._settings.auth_enabled

    # ------------------------------------------------------------- 引导
    def bootstrap_admin(self) -> UserRecord | None:
        """首次启动时按环境变量创建管理员。幂等。"""
        if not self._settings.auth_enabled:
            return None
        if self._repo.count_users() > 0:
            return None
        username = (self._settings.auth_admin_username or "").strip()
        password = self._settings.auth_admin_password or ""
        if not username or not password:
            logger.warning(
                "auth_enabled=true 但未配置 auth_admin_username / auth_admin_password，"
                "无法创建初始管理员（当前处于「无人可登录」状态）"
            )
            return None

        admin = self._repo.create_user(
            UserRecord(
                user_id=uuid.uuid4().hex[:12],
                username=username,
                password_hash=hash_password(
                    password, iterations=self._settings.auth_password_iterations
                ),
                display_name=username,
                role=ROLE_ADMIN,
                created_at=_iso(_now()),
            )
        )
        logger.info("已创建初始管理员: %s", username)
        self._repo.purge_expired_sessions()
        return admin

    # ------------------------------------------------------------- 登录
    def authenticate(self, username: str, password: str) -> UserRecord:
        """校验用户名口令。

        用户不存在与口令错误**抛出同一个异常**，避免攻击者据此枚举用户名。
        """
        record = self._repo.get_user_by_username((username or "").strip())
        if record is None:
            # 仍然走一次哈希运算，让「用户不存在」与「口令错误」的耗时接近
            verify_password(password or "", _DUMMY_HASH)
            logger.warning("登录失败：用户名不存在")
            raise InvalidCredentialsError()

        if not verify_password(password or "", record.password_hash):
            logger.warning("登录失败：口令不匹配 | 用户=%s", record.username)
            raise InvalidCredentialsError()

        if record.disabled:
            logger.warning("登录被拒：账号已停用 | 用户=%s", record.username)
            raise AuthError("该账号已被停用，请联系管理员")

        if needs_rehash(record.password_hash, iterations=self._settings.auth_password_iterations):
            # 迭代数标准提升过：静默重算一次，下次登录就是新参数
            try:
                self._repo.update_password(
                    record.user_id,
                    hash_password(password, iterations=self._settings.auth_password_iterations),
                )
                logger.info("口令哈希已按当前标准重算 | 用户=%s", record.username)
            except Exception:  # 重算失败不影响本次登录
                logger.warning("口令哈希重算失败", exc_info=True)

        logger.info("登录成功 | 用户=%s | 角色=%s", record.username, record.role)
        return record
    # ------------------------------------------------------------- 令牌
    def issue_token(self, user: UserRecord, user_agent: str = "") -> str:
        """签发会话令牌，返回**明文**（只进 Cookie，不落库）。"""
        token = generate_token()
        expires = _now() + timedelta(hours=self._settings.auth_session_ttl_hours)
        self._repo.create_auth_session(
            AuthSessionRecord(
                token_hash=token_fingerprint(token),
                user_id=user.user_id,
                created_at=_iso(_now()),
                expires_at=_iso(expires),
                user_agent=(user_agent or "")[:200],
            )
        )
        return token

    def resolve_token(self, token: str | None) -> UserRecord | None:
        """令牌 -> 用户。过期、已注销、用户被停用都返回 None。"""
        if not token:
            return None
        record = self._repo.get_auth_session(token_fingerprint(token))
        if record is None:
            return None
        if record.expires_at <= _iso(_now()):
            self._repo.delete_auth_session(record.token_hash)
            return None
        user = self._repo.get_user(record.user_id)
        if user is None or user.disabled:
            return None
        return user

    def revoke_token(self, token: str | None) -> None:
        """登出：真的销毁服务端会话，不是只清 Cookie。"""
        if not token:
            return
        self._repo.delete_auth_session(token_fingerprint(token))
        logger.info("已登出")

    # ------------------------------------------------------------- 口令
    def change_password(self, user: UserRecord, old: str, new: str) -> None:
        if not verify_password(old or "", user.password_hash):
            logger.warning("改密码失败：原口令不正确 | 用户=%s", user.username)
            raise InvalidCredentialsError()
        self._repo.update_password(
            user.user_id,
            hash_password(new, iterations=self._settings.auth_password_iterations),
        )
        logger.info("口令已修改 | 用户=%s", user.username)

    def revoke_all_sessions(self, user_id: str, *, keep: str | None = None) -> int:
        return self._repo.delete_user_sessions(user_id, keep=keep)

    # ------------------------------------------------------------- 用户管理
    def create_user(
        self, username: str, password: str, *, role: str = ROLE_ANALYST
    ) -> UserRecord:
        name = (username or "").strip()
        if not name:
            raise AuthError("用户名不能为空")
        if role not in VALID_ROLES:
            raise AuthError(f"角色不合法: {role}")
        if self._repo.get_user_by_username(name) is not None:
            raise AuthError(f"用户名 {name} 已存在")
        record = self._repo.create_user(
            UserRecord(
                user_id=uuid.uuid4().hex[:12],
                username=name,
                password_hash=hash_password(
                    password, iterations=self._settings.auth_password_iterations
                ),
                display_name=name,
                role=role,
                created_at=_iso(_now()),
            )
        )
        logger.info("已创建用户: %s（角色 %s）", name, role)
        return record

    def list_users(self) -> list[UserRecord]:
        return self._repo.list_users()