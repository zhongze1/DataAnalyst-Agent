"""统一的应用异常。

**核心纪律：面向用户的 ``message`` 只能是人写的文案**，真实异常放进 ``cause``
交给日志。绝不能把 ``str(cause)`` 直接回给客户端 —— 那会把 openpyxl 的解析栈、
httpx 的响应体、本地文件路径这些内部细节暴露出去。

本模块不依赖 FastAPI / Starlette，只是把 HTTP 状态码作为元数据挂在类上，
这样领域层（存储层）也能抛这些异常而不会引入框架依赖。
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """所有可预期业务错误的基类。"""

    code: str = "APP_ERROR"
    http_status: int = 500
    default_message: str = "服务器内部错误，请稍后重试"

    def __init__(self, message: str | None = None, *, cause: Exception | None = None) -> None:
        self.message = message or self.default_message
        self.cause = cause
        super().__init__(self.message)

    def to_payload(self, request_id: str | None = None) -> dict[str, Any]:
        """统一的响应体：``{"error": {"code", "message", "request_id"}}``。"""
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if request_id:
            error["request_id"] = request_id
        return {"error": error}


# ----------------------------------------------------------------- 文件类
class FileTypeError(AppError):
    code = "FILE_TYPE_UNSUPPORTED"
    http_status = 400
    default_message = "暂不支持的文件类型，仅支持 .xlsx / .xlsm"


class FileEmptyError(AppError):
    code = "FILE_EMPTY"
    http_status = 400
    default_message = "上传内容为空，请重新选择文件"


class FileTooLargeError(AppError):
    code = "FILE_TOO_LARGE"
    http_status = 413

    def __init__(self, max_bytes: int, *, cause: Exception | None = None) -> None:
        limit_mb = max_bytes // (1024 * 1024)
        super().__init__(f"文件超过 {limit_mb}MB 限制，请压缩后重试", cause=cause)


class FileParseError(AppError):
    code = "FILE_PARSE_ERROR"
    http_status = 400
    # 真实原因（openpyxl 报错、文件损坏细节）只写日志
    default_message = "文件无法解析，请确认是有效的 .xlsx / .xlsm 文件"


# ----------------------------------------------------------------- 会话类
class SessionNotFoundError(AppError):
    code = "SESSION_NOT_FOUND"
    http_status = 404
    default_message = "会话不存在或已过期，请重新上传文件"


# ----------------------------------------------------------------- 外部依赖
class LLMServiceError(AppError):
    code = "LLM_SERVICE_ERROR"
    http_status = 502
    default_message = "模型服务暂时不可用，请稍后重试"


class ConfigError(AppError):
    code = "CONFIG_ERROR"
    http_status = 500
    default_message = "服务配置不合法"

class BadRequestError(AppError):
    """请求参数不合法（缺参、格式不支持等）。

    与文件类错误区分开：这类问题用户改一下输入就能解决，文案应直接说明
    「应该传什么」。
    """

    code = "BAD_REQUEST"
    http_status = 400
    default_message = "请求参数不合法"


# ----------------------------------------------------------------- 认证类
class AuthError(AppError):
    """认证相关失败（登录态无效、账号停用等）。"""

    code = "AUTH_ERROR"
    http_status = 401
    default_message = "登录状态无效，请重新登录"


class InvalidCredentialsError(AuthError):
    """用户名或口令错误。

    用户不存在与口令错误共用这一种异常、同一句文案 —— **刻意不区分**。
    区分开就等于给了攻击者一个枚举用户名的工具。
    """

    code = "INVALID_CREDENTIALS"
    http_status = 401
    default_message = "用户名或密码错误"


class AuthRequiredError(AuthError):
    """需要登录但未登录。"""

    code = "AUTH_REQUIRED"
    http_status = 401
    default_message = "请先登录"


class PermissionDeniedError(AppError):
    """已登录但无权访问该资源。

    注意：会话归属校验失败时**应当返回 404 而非 403** —— 403 等于告诉
    对方「这个 id 确实存在」，本身就是信息泄露。
    """

    code = "PERMISSION_DENIED"
    http_status = 403
    default_message = "没有权限执行该操作"
