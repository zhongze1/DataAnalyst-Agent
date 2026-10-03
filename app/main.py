"""FastAPI 应用装配：中间件、路由、静态资源、生命周期。"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.config import INDEX_FILE, STATIC_DIR, get_settings
from app.dependencies import describe_settings, get_store
from app.routers import chat, files

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    store = get_store()
    logger.info("DataAnalyst Agent %s 启动 | LLM=%s", __version__, describe_settings())
    logger.info("已恢复 %d 个历史会话 | 数据目录: %s", len(store.list_sessions()), settings.upload_dir)
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
