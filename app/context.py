"""请求级上下文：在任意深度读取 request_id / session_id。

为什么用 ContextVar 而不是一路传参：Agent Loop、pandas 工具、存储层都要能
拿到「当前请求」的标识，如果靠函数签名传递，会污染十几个函数且容易漏传。
ContextVar 让它们直接读就行。

约定：**只用于日志与错误响应，禁止参与业务判断**。它只是观测手段，
任何依赖它的逻辑都会在并发下出错。
"""

from __future__ import annotations

from contextvars import ContextVar, Token

# 缺省值用 "-"：日志里能看出「这里本来就没有」，而不是 None
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
session_id_var: ContextVar[str] = ContextVar("session_id", default="-")


def current_request_id() -> str:
    return request_id_var.get()


def current_session_id() -> str:
    return session_id_var.get()


def set_session_id(value: str | None) -> Token[str]:
    """绑定会话 ID。返回 token 供调用方还原；不还原也不会跨请求泄漏
    （每个请求都在独立上下文中执行，且中间件会重置）。"""
    return session_id_var.set(value or "-")


def reset_request_context(request_token: Token[str], session_token: Token[str]) -> None:
    request_id_var.reset(request_token)
    session_id_var.reset(session_token)
