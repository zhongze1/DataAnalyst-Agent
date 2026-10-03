"""项目入口。

    uv run uvicorn main:app --reload
    或
    uv run python -m uvicorn app.main:app --reload
"""

from app.main import app

__all__ = ["app"]
