"""全局配置。

所有配置项都可通过环境变量覆盖（前缀 ``AGENT_``），或写入项目根目录的 ``.env``：

    AGENT_LLM_PROVIDER=mock          # openai | mock
    AGENT_LLM_API_KEY=sk-xxx
    AGENT_LLM_BASE_URL=https://api.deepseek.com/v1
    AGENT_LLM_MODEL=deepseek-chat
    AGENT_MAX_AGENT_STEPS=12
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "app" / "web"
STATIC_DIR = WEB_DIR / "static"
INDEX_FILE = WEB_DIR / "index.html"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"


class Settings(BaseSettings):
    """运行时配置。"""

    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- LLM ----
    # openai : 任意 OpenAI 兼容接口（OpenAI / DeepSeek / Moonshot / OpenRouter / vLLM ...）
    # mock   : 本地假模型，不联网，用于离线跑通 Agent Loop
    llm_provider: Literal["openai", "mock"] = "openai"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_timeout: float = 120.0
    llm_max_retries: int = 2

    # ---- Agent ----
    max_agent_steps: int = 12
    max_tool_output_chars: int = 4000
    max_sandbox_code_lines: int = 80

    # ---- 上传 ----
    upload_dir: Path = UPLOAD_DIR
    max_upload_bytes: int = 50 * 1024 * 1024
    allowed_suffixes: tuple[str, ...] = (".xlsx", ".xlsm", ".xltx", ".xltm")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    return settings
