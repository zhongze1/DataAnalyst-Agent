"""密码哈希与令牌生成。

选 PBKDF2-HMAC-SHA256 的理由：标准库自带、OWASP 认可参数、无需编译型依赖
（argon2-cffi 在 Windows 上要装 Rust 工具链）。将来要换 argon2 只需替换
:func:`hash_password` 与 :func:`verify_password` 两个函数。

编码格式（自描述，便于将来平滑升级算法）::

    pbkdf2_sha256$<iterations>$<salt_b64>$<hash_b64>
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import secrets

logger = logging.getLogger(__name__)

ALGORITHM = "pbkdf2_sha256"
# OWASP 对 PBKDF2-HMAC-SHA256 的推荐值
DEFAULT_ITERATIONS = 600_000
SALT_BYTES = 16
DK_LEN = 32

# 明文口令的最小长度。太短的口令即使哈希算法再强也扛不住爆破。
MIN_PASSWORD_LENGTH = 8


class PasswordTooShort(ValueError):
    """口令不满足最小长度。"""


def validate_password_strength(password: str) -> None:
    if len(password or "") < MIN_PASSWORD_LENGTH:
        raise PasswordTooShort(f"口令至少需要 {MIN_PASSWORD_LENGTH} 个字符")


def hash_password(password: str, *, iterations: int = DEFAULT_ITERATIONS) -> str:
    """生成自描述的哈希串。**明文口令永不入日志。**"""
    validate_password_strength(password)
    salt = secrets.token_bytes(SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations, DK_LEN)
    return f"{ALGORITHM}${iterations}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    """校验口令。算法/参数不匹配时返回 False 而不是抛异常。

    使用 :func:`hmac.compare_digest` 做常数时间比较，避免计时侧信道。
    """
    try:
        algorithm, iterations_raw, salt_b64, hash_b64 = encoded.split("$", 3)
        if algorithm != ALGORITHM:
            logger.warning("口令使用了不支持的算法: %s", algorithm)
            return False
        iterations = int(iterations_raw)
        salt = _unb64(salt_b64)
        expected = _unb64(hash_b64)
    except (ValueError, TypeError):
        # 存量数据格式异常：当作校验失败，但必须留痕
        logger.warning("口令哈希格式无法解析")
        return False

    actual = hashlib.pbkdf2_hmac(
        "sha256", (password or "").encode("utf-8"), salt, iterations, DK_LEN
    )
    return hmac.compare_digest(actual, expected)


def needs_rehash(encoded: str, *, iterations: int = DEFAULT_ITERATIONS) -> bool:
    """存量口令的迭代数是否已低于当前标准 —— 是则应在下次登录时重算。"""
    try:
        algorithm, iterations_raw, _, _ = encoded.split("$", 3)
    except (ValueError, TypeError):
        return True
    return algorithm != ALGORITHM or int(iterations_raw) < iterations


# ------------------------------------------------------------------ 令牌
def generate_token() -> str:
    """生成会话令牌。

    用 ``secrets`` 而非 ``uuid4``：uuid4 只有 122 bit 且来自非密码学安全的
    发生器，``token_urlsafe(32)`` 给 256 bit，适合当会话凭据。
    """
    return secrets.token_urlsafe(32)


def token_fingerprint(token: str) -> str:
    """令牌的存储形式：只存哈希。

    数据库泄露时，攻击者拿到的是不可逆摘要而非可直接使用的 Cookie。
    这条不做的话，整个认证形同虚设。
    """
    return hashlib.sha256((token or "").encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ base64
def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.b64decode(text + padding)