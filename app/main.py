"""FastAPI 应用装配：中间件、路由、静态资源、生命周期。"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

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
from app.routers import chat, files

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动即校验配置：宁可起不来，也不要带着错配置对外服务。"""
    settings = get_settings()

    problems = validate_settings(settings)
    if problems:
        for problem in problems:
            logger.error("配置错误 [%s] %s", problem.key, problem.message)
            if problem.hint:
                logger.error("    建议: %s", problem.hint)
        if settings.strict_config:
            # 校验放在 get_store() 之前：配置有问题时不该留下任何副作用
            raise RuntimeError(format_problems(problems))
        logger.warning(
            "已忽略 %d 个配置问题（AGENT_STRICT_CONFIG=false），服务继续启动",
            len(problems),
        )

    logger.info("运行时配置 | %s", runtime_summary(settings))
    store = get_store()
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

    app.include_router(files.router)
    app.include_router(chat.router)

    @app.get("/api/health", tags=["meta"])
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "version": __version__, **describe_settings()})

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(INDEX_FILE)

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


app = create_app()
