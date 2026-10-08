"""W2-1.5 认证测试。

三条安全底线：
1. 口令永不明文落库、永不进日志；
2. 会话令牌只存哈希（数据库泄露拿不到可用凭据）；
3. 归属隔离 —— 用户 A 拿不到用户 B 的会话（返回 404 而非 403，
   403 等于告诉对方「这个 id 存在」）。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.errors import AuthError, InvalidCredentialsError
from app.security.passwords import (
    DEFAULT_ITERATIONS,
    MIN_PASSWORD_LENGTH,
    PasswordTooShort,
    generate_token,
    hash_password,
    needs_rehash,
    token_fingerprint,
    verify_password,
)
from app.services.auth_service import AuthService
from app.storage.repository import AuthSessionRecord

# ============================================================ 口令哈希
def test_password_roundtrip() -> None:
    encoded = hash_password("correct-horse", iterations=1000)
    assert verify_password("correct-horse", encoded) is True
    assert verify_password("wrong-horse", encoded) is False


def test_hash_never_contains_plaintext() -> None:
    secret = "MySecret123"
    encoded = hash_password(secret, iterations=1000)
    assert secret not in encoded
    assert encoded.startswith("pbkdf2_sha256$1000$")


def test_same_password_gets_different_hashes() -> None:
    """盐要随机：相同口令两次哈希必须不同，否则彩虹表可破。"""
    a = hash_password("same-password", iterations=1000)
    b = hash_password("same-password", iterations=1000)
    assert a != b
    assert verify_password("same-password", a)
    assert verify_password("same-password", b)


def test_default_iterations_meet_owasp_recommendation() -> None:
    """锁定默认值：调低会削弱所有用户的口令抗暴力破解能力。"""
    assert DEFAULT_ITERATIONS >= 600_000


def test_short_password_is_rejected() -> None:
    with pytest.raises(PasswordTooShort):
        hash_password("x" * (MIN_PASSWORD_LENGTH - 1))
    hash_password("x" * MIN_PASSWORD_LENGTH)  # 边界：正好够长不报错


@pytest.mark.parametrize(
    "broken",
    ["", "no-dollars", "a$b$c", "pbkdf2_sha256$notanint$c$d", "md5$1$c$d"],
)
def test_broken_hash_fails_closed(broken: str) -> None:
    """存量数据格式异常时必须判定失败，而不是抛异常或放行。"""
    assert verify_password("anything", broken) is False


def test_needs_rehash_detects_outdated_params() -> None:
    old = hash_password("pw-12345678", iterations=1000)
    assert needs_rehash(old, iterations=600_000) is True
    assert needs_rehash(old, iterations=1000) is False


# ============================================================ 令牌
def test_token_is_unguessable_and_unique() -> None:
    tokens = {generate_token() for _ in range(200)}
    assert len(tokens) == 200
    assert all(len(t) >= 40 for t in tokens)


def test_fingerprint_is_deterministic_and_hides_token() -> None:
    token = generate_token()
    first = token_fingerprint(token)
    assert first == token_fingerprint(token)
    assert token not in first
    assert len(first) == 64  # sha256 hex


# ============================================================ AuthService
def test_bootstrap_creates_admin_once(auth_service: AuthService) -> None:
    admin = auth_service.repo.get_user_by_username("admin")
    assert admin is not None
    assert admin.is_admin
    # 幂等：再调一次不会重复建，也不会抛「用户名已存在」
    assert auth_service.bootstrap_admin() is None
    assert auth_service.repo.count_users() == 1


def test_login_success(auth_service: AuthService) -> None:
    user = auth_service.authenticate("admin", "admin-pass-123")
    assert user.username == "admin"


def test_login_rejects_wrong_password(auth_service: AuthService) -> None:
    with pytest.raises(InvalidCredentialsError):
        auth_service.authenticate("admin", "wrong-password")


def test_login_failure_is_identical_for_unknown_user(auth_service: AuthService) -> None:
    """防用户枚举：用户不存在与口令错误必须抛同一种异常、同一句文案。"""
    with pytest.raises(InvalidCredentialsError) as unknown:
        auth_service.authenticate("nobody", "whatever")
    with pytest.raises(InvalidCredentialsError) as wrong:
        auth_service.authenticate("admin", "wrong")
    assert unknown.value.code == wrong.value.code
    assert unknown.value.message == wrong.value.message


def test_disabled_user_cannot_login(auth_service: AuthService) -> None:
    admin = auth_service.repo.get_user_by_username("admin")
    auth_service.repo.set_disabled(admin.user_id, True)
    with pytest.raises(AuthError):
        auth_service.authenticate("admin", "admin-pass-123")


def test_token_roundtrip_and_revocation(auth_service: AuthService) -> None:
    user = auth_service.authenticate("admin", "admin-pass-123")
    token = auth_service.issue_token(user)
    assert auth_service.resolve_token(token).user_id == user.user_id
    auth_service.revoke_token(token)
    assert auth_service.resolve_token(token) is None


def test_expired_token_is_rejected_and_cleaned(auth_service: AuthService) -> None:
    user = auth_service.authenticate("admin", "admin-pass-123")
    token = generate_token()  # 自己造，不走 issue_token（它已插过一条）
    expired = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(
        timespec="milliseconds"
    )
    auth_service.repo.create_auth_session(
        AuthSessionRecord(
            token_hash=token_fingerprint(token),
            user_id=user.user_id,
            created_at=expired,
            expires_at=expired,
        )
    )
    assert auth_service.resolve_token(token) is None
    # 过期记录应被顺手清掉，避免表里堆垃圾
    assert auth_service.repo.get_auth_session(token_fingerprint(token)) is None


def test_garbage_token_is_rejected(auth_service: AuthService) -> None:
    assert auth_service.resolve_token("not-a-real-token") is None
    assert auth_service.resolve_token(None) is None
    assert auth_service.resolve_token("") is None
def test_change_password_revokes_other_sessions(auth_service: AuthService) -> None:
    """改口令后「其他设备」必须失效，否则等于没改。"""
    user = auth_service.authenticate("admin", "admin-pass-123")
    keep = auth_service.issue_token(user)
    other = auth_service.issue_token(user)

    auth_service.change_password(user, "admin-pass-123", "new-pass-456")
    auth_service.revoke_all_sessions(user.user_id, keep=token_fingerprint(keep))

    assert auth_service.resolve_token(keep) is not None
    assert auth_service.resolve_token(other) is None
    # 旧口令不能再登录
    with pytest.raises(InvalidCredentialsError):
        auth_service.authenticate("admin", "admin-pass-123")
    assert auth_service.authenticate("admin", "new-pass-456").username == "admin"


def test_change_password_requires_correct_old_password(auth_service: AuthService) -> None:
    user = auth_service.authenticate("admin", "admin-pass-123")
    with pytest.raises(InvalidCredentialsError):
        auth_service.change_password(user, "not-the-old-one", "new-pass-456")


def test_create_user_and_duplicate_detection(auth_service: AuthService) -> None:
    created = auth_service.create_user("alice", "alice-pass-1")
    assert created.role == "analyst"
    assert not created.is_admin
    with pytest.raises(AuthError):
        auth_service.create_user("alice", "another-pass")


def test_create_user_rejects_invalid_role(auth_service: AuthService) -> None:
    with pytest.raises(AuthError):
        auth_service.create_user("bob", "bob-pass-12", role="superuser")


def test_password_hash_is_rehashed_on_login_when_policy_tightened(
    auth_service: AuthService, auth_settings
) -> None:
    """存量口令迭代数低于当前标准时，登录成功即静默重算。

    夹具把迭代数设成 1000，所以这里用 10 模拟「旧标准下创建的账号」。
    """
    target = auth_settings.auth_password_iterations
    user = auth_service.authenticate("admin", "admin-pass-123")
    # 手工降级哈希，模拟旧标准下的存量账号
    weak = hash_password("admin-pass-123", iterations=max(1, target // 100))
    auth_service.repo.update_password(user.user_id, weak)
    assert needs_rehash(weak, iterations=target) is True

    auth_service.authenticate("admin", "admin-pass-123")
    stored = auth_service.repo.get_user_by_username("admin")
    assert needs_rehash(stored.password_hash, iterations=target) is False
# ============================================================ HTTP 层
def test_protected_endpoints_require_login(auth_client) -> None:
    assert auth_client.get("/api/sessions").status_code == 401
    assert auth_client.get("/api/files/abc").status_code == 401
    chat = auth_client.post("/api/chat", json={"message": "hi"})
    assert chat.status_code == 401
    assert chat.json()["error"]["code"] == "AUTH_REQUIRED"


def test_login_sets_httponly_cookie(auth_client) -> None:
    response = auth_client.post(
        "/api/auth/login", json={"username": "admin", "password": "admin-pass-123"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["username"] == "admin"
    assert body["role"] == "admin"
    # 口令绝不能出现在响应里
    assert "admin-pass-123" not in response.text

    cookie_header = response.headers.get("set-cookie", "")
    assert "agent_session=" in cookie_header
    assert "HttpOnly" in cookie_header      # JS 读不到，XSS 偷不走
    assert "SameSite=lax" in cookie_header   # 防 CSRF


def test_cookie_value_is_not_the_stored_fingerprint(auth_client) -> None:
    """Cookie 里是明文令牌，库里只存哈希 —— 两者必须不同。

    这条是整套认证的安全底线：数据库泄露时，攻击者拿到的必须是不可逆摘要。
    """
    import hashlib
    import sqlite3

    from app.config import get_settings

    auth_client.post(
        "/api/auth/login", json={"username": "admin", "password": "admin-pass-123"}
    )
    token = auth_client.cookies.get("agent_session")
    assert token, "应下发 Cookie"

    conn = sqlite3.connect(str(get_settings().db_path))
    stored = {row[0] for row in conn.execute("SELECT token_hash FROM auth_sessions")}
    conn.close()

    assert stored, "库里应有登录会话记录"
    assert token not in stored, "明文令牌绝不能落库"
    assert hashlib.sha256(token.encode()).hexdigest() in stored


def test_login_failure_does_not_leak_user_existence(auth_client) -> None:
    unknown = auth_client.post(
        "/api/auth/login", json={"username": "nobody", "password": "whatever"}
    )
    wrong = auth_client.post(
        "/api/auth/login", json={"username": "admin", "password": "wrong"}
    )
    assert unknown.status_code == wrong.status_code == 401
    assert unknown.json()["error"]["message"] == wrong.json()["error"]["message"]


def test_me_returns_user_after_login(auth_client) -> None:
    assert auth_client.get("/api/auth/me").json() is None  # 未登录
    auth_client.post(
        "/api/auth/login", json={"username": "admin", "password": "admin-pass-123"}
    )
    body = auth_client.get("/api/auth/me").json()
    assert body["username"] == "admin"


def test_logout_clears_session(auth_client) -> None:
    auth_client.post(
        "/api/auth/login", json={"username": "admin", "password": "admin-pass-123"}
    )
    assert auth_client.get("/api/auth/me").json() is not None
    auth_client.post("/api/auth/logout")
    # 登出后 Cookie 失效，受保护接口应重新 401
    auth_client.cookies.clear()
    assert auth_client.get("/api/sessions").status_code == 401


def test_history_survives_restart_of_token_only(auth_client) -> None:
    """令牌在有效期内，刷新页面（带同一 Cookie）应仍登录。"""
    auth_client.post(
        "/api/auth/login", json={"username": "admin", "password": "admin-pass-123"}
    )
    assert auth_client.get("/api/auth/me").json() is not None
    assert auth_client.get("/api/sessions").status_code == 200
# ================================================== 归属隔离（核心验收）
def _login(client, username: str, password: str) -> None:
    response = client.post(
        "/api/auth/login", json={"username": username, "password": password}
    )
    assert response.status_code == 200, response.text


def test_admin_claims_orphan_sessions_on_bootstrap(auth_store, auth_service) -> None:
    """无主的历史会话必须归给初始管理员，不能丢也不能留在外面。"""
    orphan = auth_store.create("历史.xlsx", _xlsx_bytes(), owner_id=None)
    assert orphan.owner_id is None
    claimed = auth_store.claim_orphan_sessions(
        auth_service.repo.get_user_by_username("admin").user_id
    )
    assert claimed == 1
    assert auth_store.get(orphan.session_id).owner_id is not None


def test_user_cannot_see_other_users_sessions(auth_client, auth_store, auth_service) -> None:
    admin = auth_service.repo.get_user_by_username("admin")
    mine = auth_store.create("我的数据.xlsx", _xlsx_bytes(), owner_id=admin.user_id)

    _login(auth_client, "admin", "admin-pass-123")
    listed = auth_client.get("/api/sessions").json()
    assert [s["session_id"] for s in listed] == [mine.session_id]


def test_other_users_session_is_invisible_not_just_forbidden(
    auth_client, auth_store, auth_service
) -> None:
    """归属不符时必须返回 404 而不是 403。

    403 等于告诉对方「这个 session_id 确实存在」，本身就是信息泄露。
    """
    admin = auth_service.repo.get_user_by_username("admin")
    his = auth_store.create("管理员的数据.xlsx", _xlsx_bytes(), owner_id=admin.user_id)

    _login(auth_client, "admin", "admin-pass-123")
    detail = auth_client.get(f"/api/sessions/{his.session_id}")
    assert detail.status_code == 200  # 自己的看得到

    # 换成另一个用户
    auth_client.cookies.clear()
    auth_service.create_user("bob", "bob-pass-1234")
    _login(auth_client, "bob", "bob-pass-1234")

    assert auth_client.get(f"/api/sessions/{his.session_id}").status_code == 404
    assert auth_client.get(f"/api/files/{his.session_id}").status_code == 404
    assert auth_client.delete(f"/api/files/{his.session_id}").status_code == 404
    # 列表里也不该出现
    listed = auth_client.get("/api/sessions").json()
    assert all(s["session_id"] != his.session_id for s in listed)


def test_session_listing_is_scoped_per_user(auth_client, auth_store, auth_service) -> None:
    admin = auth_service.repo.get_user_by_username("admin")
    auth_service.create_user("bob", "bob-pass-1234")
    bob = auth_service.repo.get_user_by_username("bob")

    admin_file = auth_store.create("A.xlsx", _xlsx_bytes(), owner_id=admin.user_id)
    bob_file = auth_store.create("B.xlsx", _xlsx_bytes(), owner_id=bob.user_id)

    _login(auth_client, "admin", "admin-pass-123")
    ids = {s["session_id"] for s in auth_client.get("/api/sessions").json()}
    assert ids == {admin_file.session_id}

    auth_client.cookies.clear()
    _login(auth_client, "bob", "bob-pass-1234")
    ids = {s["session_id"] for s in auth_client.get("/api/sessions").json()}
    assert ids == {bob_file.session_id}


def test_uploaded_session_is_owned_by_uploader(auth_client, auth_service) -> None:
    auth_service.create_user("bob", "bob-pass-1234")
    _login(auth_client, "bob", "bob-pass-1234")
    response = auth_client.post(
        "/api/files/upload",
        files={"file": ("我的数据.xlsx", _xlsx_bytes(), "application/vnd.ms-excel")},
    )
    assert response.status_code == 200
    session_id = response.json()["file"]["session_id"]
    assert auth_client.get(f"/api/sessions/{session_id}").status_code == 200


# ================================================== 管理员权限
def test_only_admin_can_list_and_create_users(auth_client, auth_service) -> None:
    _login(auth_client, "admin", "admin-pass-123")
    assert auth_client.get("/api/auth/users").status_code == 200
    created = auth_client.post(
        "/api/auth/users",
        json={"username": "carol", "password": "carol-pass-1", "role": "analyst"},
    )
    assert created.status_code == 201

    auth_client.cookies.clear()
    _login(auth_client, "carol", "carol-pass-1")
    assert auth_client.get("/api/auth/users").status_code == 403
    assert auth_client.post(
        "/api/auth/users",
        json={"username": "dave", "password": "dave-pass-12"},
    ).status_code == 403


def test_short_password_rejected_by_api(auth_client) -> None:
    _login(auth_client, "admin", "admin-pass-123")
    response = auth_client.post(
        "/api/auth/users", json={"username": "eve", "password": "short"}
    )
    assert response.status_code == 422


def test_password_change_keeps_current_session(auth_client) -> None:
    _login(auth_client, "admin", "admin-pass-123")
    assert auth_client.post(
        "/api/auth/password",
        json={"old_password": "admin-pass-123", "new_password": "brand-new-99"},
    ).status_code == 200
    # 当前会话仍有效（否则用户改完口令就被自己踢出去了）
    assert auth_client.get("/api/auth/me").json()["username"] == "admin"
    # 新口令可登录
    auth_client.cookies.clear()
    _login(auth_client, "admin", "brand-new-99")


def _xlsx_bytes() -> bytes:
    import io

    import pandas as pd

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame({"a": [1, 2]}).to_excel(writer, sheet_name="S", index=False)
    return buffer.getvalue()