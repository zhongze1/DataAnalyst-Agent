"""FastAPI 应用装配：中间件、路由、静态资源、生命周期。"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles


class NoCacheStaticFiles(StaticFiles):
    """开发期禁用静态资源缓存。

    本项目前端是无构建的原生 JS，文件名不带内容哈希。Starlette 默认不发
    Cache-Control，浏览器会按启发式规则把旧文件留在本地 —— 表现为「改了
    JS 但页面行为不变」，排查极费时间。

    ``no-cache`` 的语义是「每次回源校验」而非「不许存」：配合 ETag，文件
    没变回 304，变了才传新内容。既保证新鲜，又不浪费带宽。
    """

    def file_response(self, *args: Any, **kwargs: Any) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__
from app.config import (
    INDEX_FILE,
    STATIC_DIR,
    format_problems,
    get_settings,
    runtime_summary,
    validate_settings,
)
from app.dependencies import describe_settings, get_store
from app.errors import AppError, ConfigError
from app.logging_setup import setup_logging
from app.middleware import REQUEST_ID_HEADER, RequestContextMiddleware
from app.storage.migrate import migrate_meta_files
from app.routers import auth, chat, chat_stream, export, files, sessions

logger = logging.getLogger(__name__)


# ============================================================ 统一错误响应
def _request_id(request: Request) -> str | None:
    """从 ASGI scope 取 request_id。

    不能读 ContextVar：未捕获异常一路往上抛时，中间件的 finally 已经把它重置了，
    只有 scope 里的副本还在。
    """
    return request.scope.get("request_id")


def _error_body(code: str, message: str, request: Request, **extra: object) -> dict:
    body: dict[str, object] = {"code": code, "message": message}
    if request_id := _request_id(request):
        body["request_id"] = request_id
    body.update(extra)
    return {"error": body}


def _safe_validation_details(errors: list[dict]) -> list[dict]:
    """只保留 loc/msg/type。

    必须丢掉 ``input`` 与 ``ctx``：input 可能是上传的文件内容或用户原文，
    ctx 里常塞着不可序列化的异常对象。
    """
    return [
        {
            "loc": [str(part) for part in item.get("loc", ())],
            "msg": str(item.get("msg", "")),
            "type": str(item.get("type", "")),
        }
        for item in errors[:10]
    ]


def register_exception_handlers(app: FastAPI) -> None:
    """所有对外错误统一收敛成 ``{"error": {code, message, request_id}}``。"""

    @app.exception_handler(AppError)
    async def _handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        logger.warning("业务异常 %s | %s", exc.code, exc.message, extra={"error_code": exc.code})
        if exc.cause is not None:
            # 真实原因只进日志：响应体里绝不能出现 openpyxl / httpx 的内部细节
            logger.debug("异常细节", exc_info=exc.cause)
        response = JSONResponse(
            status_code=exc.http_status, content=exc.to_payload(_request_id(request))
        )
        # 配额/并发类异常带 retry_after：只放响应头，不动错误体契约
        # （{"error": {code, message, request_id}} 的键集是被测试锁住的契约）
        retry_after = getattr(exc, "retry_after", None)
        if retry_after:
            response.headers["Retry-After"] = str(int(retry_after))
        return response

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = _safe_validation_details(exc.errors())
        logger.info("参数校验失败 | %s", details[:3], extra={"error_code": "VALIDATION_ERROR"})
        return JSONResponse(
            status_code=422,
            content=_error_body(
                "VALIDATION_ERROR", "请求参数不合法，请检查后重试", request, details=details
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        # 路由不存在、方法不允许等由 Starlette 抛出的 HTTPException
        if exc.status_code == 404:
            code, message = "NOT_FOUND", "接口不存在"
        else:
            code = "HTTP_ERROR"
            message = exc.detail if isinstance(exc.detail, str) else "请求失败"
        if exc.status_code >= 500:
            logger.error("HTTP 异常 %d | %s", exc.status_code, message, extra={"error_code": code})
        return JSONResponse(
            status_code=exc.status_code, content=_error_body(code, message, request)
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        # 兜底：真实异常与 traceback 只进日志，客户端只拿到通用文案 + request_id
        logger.exception(
            "未处理异常 | %s", type(exc).__name__, extra={"error_code": "INTERNAL_ERROR"}
        )
        response = JSONResponse(
            status_code=500,
            content=_error_body(
                "INTERNAL_ERROR", "服务器内部错误，请稍后重试或联系管理员", request
            ),
        )
        # 异常穿过了中间件，中间件来不及写响应头，这里补上以保持一致
        if request_id := _request_id(request):
            response.headers[REQUEST_ID_HEADER] = request_id
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动即校验配置：宁可起不来，也不要带着错配置对外服务。"""
    settings = get_settings()
    # 先配日志再校验：否则配置错误会以无格式的 lastResort 输出落到 stderr
    setup_logging(settings.log_format, settings.log_level)

    problems = validate_settings(settings)
    if problems:
        for problem in problems:
            logger.error("配置错误 [%s] %s", problem.key, problem.message)
            if problem.hint:
                logger.error("    建议: %s", problem.hint)
        if settings.strict_config:
            # 校验放在 get_store() 之前：配置有问题时不该留下任何副作用
            raise ConfigError(format_problems(problems))
        logger.warning(
            "已忽略 %d 个配置问题（AGENT_STRICT_CONFIG=false），服务继续启动",
            len(problems),
        )

    store = get_store()
    # 旧版会话索引只存在 meta.json 里，首次启动自动导入数据库（幂等）
    try:
        migrate_meta_files(store.repo, settings.upload_dir)
    except Exception:  # 迁移失败不应阻止启动
        logger.warning("历史会话迁移失败", exc_info=True)

    # 认证引导：首次启动创建初始管理员，并把无主的历史会话认领给他
    if settings.auth_enabled:
        try:
            from app.dependencies import get_auth_service

            auth = get_auth_service()
            admin = auth.bootstrap_admin()
            if admin is not None:
                claimed = store.claim_orphan_sessions(admin.user_id)
                if claimed:
                    logger.info("已把 %d 个历史会话归属给管理员 %s", claimed, admin.username)
        except Exception:  # 引导失败不应阻止启动，但必须留痕
            logger.exception("认证引导失败")
    else:
        logger.warning("认证已关闭（auth_enabled=false）：任何人都可访问本服务，仅限本机开发")

    logger.info("运行时配置 | %s", runtime_summary(settings))
    logger.info(
        "已恢复 %d 个历史会话 | 会话目录: %s", len(store.list_sessions()), settings.upload_dir
    )
    yield
    logger.info("服务关闭")


def create_app() -> FastAPI:
    app = FastAPI(
        title="DataAnalyst Agent",
        version=__version__,
        description="上传 Excel → 单 Agent Loop 自主调用 pandas 分析",
        lifespan=lifespan,
    )

    # 必须最早挂载：异常路径与静态资源也要带 request_id
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)

    app.include_router(files.router)
    app.include_router(auth.router)
    app.include_router(chat.router)
    app.include_router(chat_stream.router)
    app.include_router(sessions.router)
    app.include_router(export.router)

    @app.get("/api/health", tags=["meta"])
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "version": __version__, **describe_settings()})

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(INDEX_FILE)

    app.mount("/static", NoCacheStaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


app = create_app()
