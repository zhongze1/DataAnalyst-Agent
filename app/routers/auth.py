"""认证接口：登录 / 登出 / 当前用户 / 改密码 / 用户管理。

Cookie 属性：HttpOnly（JS 读不到，XSS 偷不走）、SameSite=Lax（防 CSRF）、
Secure 由 ``auth_cookie_secure`` 控制（生产走 HTTPS 时必须开）。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request, Response

from app.config import Settings, get_settings
from app.dependencies import (
    current_user,
    get_auth_service,
    read_token,
    require_admin,
    require_user,
)
from app.schemas import (
    ChangePasswordRequest,
    CreateUserRequest,
    LoginRequest,
    UserInfo,
)
from app.services.auth_service import AuthService
from app.storage.repository import UserRecord

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _to_info(user: UserRecord) -> UserInfo:
    return UserInfo(
        user_id=user.user_id,
        username=user.username,
        display_name=user.display_name or user.username,
        role=user.role,
    )


def _set_cookie(response: Response, token: str, settings: Settings) -> None:
    response.set_cookie(
        key=settings.auth_cookie_name,
        value=token,
        max_age=settings.auth_session_ttl_hours * 3600,
        httponly=True,
        samesite="lax",
        secure=settings.auth_cookie_secure,
        path="/",
    )


@router.get("/me", response_model=UserInfo | None)
async def read_me(
    user: UserRecord | None = Depends(current_user),
    settings: Settings = Depends(get_settings),
) -> UserInfo | None:
    """当前登录用户。未登录返回 null（前端据此显示登录页）。

    认证关闭时也返回 null —— 前端此时应进入「本机模式」而不是登录页。
    """
    if not settings.auth_enabled:
        return None
    if user is None:
        return None
    return _to_info(user)


@router.post("/login", response_model=UserInfo)
async def login(
    payload: LoginRequest,
    response: Response,
    request: Request,
    auth: AuthService = Depends(get_auth_service),
    settings: Settings = Depends(get_settings),
) -> UserInfo:
    """登录。成功后下发 HttpOnly Cookie。

    口令错误与用户不存在返回**同一个错误码与文案**，避免被用来枚举用户名。
    """
    user = auth.authenticate(payload.username, payload.password)
    token = auth.issue_token(user, user_agent=request.headers.get("user-agent", ""))
    _set_cookie(response, token, settings)
    return _to_info(user)


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    auth: AuthService = Depends(get_auth_service),
    settings: Settings = Depends(get_settings),
) -> dict[str, bool]:
    """登出：销毁服务端会话 + 清 Cookie。

    必须从 **request** 读令牌（从 response 读不到 Cookie）。
    """
    auth.revoke_token(read_token(request))
    response.delete_cookie(settings.auth_cookie_name, path="/")
    return {"ok": True}


@router.post("/password", response_model=UserInfo)
async def change_password(
    payload: ChangePasswordRequest,
    request: Request,
    auth: AuthService = Depends(get_auth_service),
    user: UserRecord = Depends(require_user),
) -> UserInfo:
    """改口令。改完**踢掉其他所有会话**（当前会话保留）。"""
    auth.change_password(user, payload.old_password, payload.new_password)
    auth.revoke_all_sessions(user.user_id, keep=_current_fingerprint(request))
    return _to_info(user)


def _current_fingerprint(request: Request) -> str | None:
    from app.security.passwords import token_fingerprint

    return token_fingerprint(read_token(request) or "")


@router.get("/users", response_model=list[UserInfo])
async def list_users(
    auth: AuthService = Depends(get_auth_service),
    admin: UserRecord = Depends(require_admin),
) -> list[UserInfo]:
    """用户列表，仅管理员可见。"""
    return [_to_info(u) for u in auth.list_users()]


@router.post("/users", response_model=UserInfo, status_code=201)
async def create_user(
    payload: CreateUserRequest,
    auth: AuthService = Depends(get_auth_service),
    admin: UserRecord = Depends(require_admin),
) -> UserInfo:
    """新建用户，仅管理员可见。不提供开放注册。"""
    record = auth.create_user(payload.username, payload.password, role=payload.role)
    return _to_info(record)