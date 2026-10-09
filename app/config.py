"""全局配置。

所有配置项都可通过环境变量覆盖（前缀 ``AGENT_``），或写入项目根目录的 ``.env``：

    AGENT_LLM_PROVIDER=mock          # openai | mock
    AGENT_LLM_API_KEY=sk-xxx
    AGENT_LLM_BASE_URL=https://api.deepseek.com/v1
    AGENT_LLM_MODEL=deepseek-chat
    AGENT_MAX_AGENT_STEPS=12
    AGENT_STRICT_CONFIG=true         # 启动时配置不合法则直接拒绝启动
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

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

    # ---- 启动行为 ----
    # 开启时，配置校验不通过直接拒绝启动（fail-fast）；
    # 关闭时只打 ERROR 日志，服务照常启动，便于临时带着问题调试
    strict_config: bool = True

    # ---- 认证 ----
    # 开发时可置 false 免登录；生产务必开启
    auth_enabled: bool = True
    # 首次启动时自动创建该管理员（不提供开放注册）
    auth_admin_username: str = ""
    auth_admin_password: str = ""
    auth_session_ttl_hours: int = 12
    auth_cookie_name: str = "agent_session"
    # 生产环境走 HTTPS 时应置 true
    auth_cookie_secure: bool = False
    # PBKDF2 迭代数：生产用 OWASP 推荐值，测试会调低以提速
    auth_password_iterations: int = 600_000

    # ---- 会话持久化 ----
    # SQLite 单文件；换 PostgreSQL 时改这里 + 换 repository 实现
    db_path: Path = DATA_DIR / "agent.db"

    # ---- 配额与并发（DA-017）----
    # 只支持单进程部署：分钟窗口与并发占用是**进程内**状态，不跨 worker 共享。
    # 每日计数与 token 用量落 SQLite（需跨重启保留）。
    quota_enabled: bool = True
    # 每日额度的自然日边界与提示文案都按此时区计算；解析失败只告警并回退 UTC
    quota_timezone: str = "Asia/Shanghai"
    quota_user_per_minute: int = 10
    quota_user_per_day: int = 50
    quota_session_per_minute: int = 5
    quota_max_concurrency_per_user: int = 1
    # 全局最大并发：同时最多跑几个 Agent 请求
    quota_max_concurrency_global: int = 4
    # 并发类拒绝（用户并发 / 全局繁忙）建议客户端等待的秒数
    quota_retry_after_seconds: int = 5

    # ---- 审计 ----
    # 工具调用审计（JSONL），与日志分开：日志给运维看，审计给合规看
    audit_dir: Path = DATA_DIR / "audit"
    audit_enabled: bool = True

    # ---- 日志 ----
    # text = 人类可读；json = 结构化，供日志系统采集聚合
    log_format: Literal["text", "json"] = "text"
    log_level: str = "INFO"


# --------------------------------------------------------------------- 校验
@dataclass(frozen=True)
class ConfigProblem:
    """一条配置问题。``hint`` 是给使用者的修复建议。"""

    key: str
    message: str
    hint: str = ""


def validate_settings(settings: Settings) -> list[ConfigProblem]:
    """检查配置能否支撑正常运行，返回问题清单（空列表 = 可启动）。

    刻意做成纯函数：不打印、不抛异常、不改配置。是否阻止启动由调用方决定
    （见 ``Settings.strict_config``）。这样每条规则都能被独立测试。
    """
    problems: list[ConfigProblem] = []

    # ---- 认证 ----
    if settings.auth_enabled:
        if not settings.auth_admin_username.strip():
            problems.append(
                ConfigProblem(
                    key="AGENT_AUTH_ADMIN_USERNAME",
                    message="已开启认证但未配置初始管理员用户名",
                    hint="首次启动需要它来创建管理员；或设 AGENT_AUTH_ENABLED=false 免登录",
                )
            )
        if not settings.auth_admin_password:
            problems.append(
                ConfigProblem(
                    key="AGENT_AUTH_ADMIN_PASSWORD",
                    message="已开启认证但未配置初始管理员口令",
                    hint="至少 8 个字符；或设 AGENT_AUTH_ENABLED=false 免登录",
                )
            )
        elif len(settings.auth_admin_password) < 8:
            problems.append(
                ConfigProblem(
                    key="AGENT_AUTH_ADMIN_PASSWORD",
                    message="初始管理员口令过短（至少 8 个字符）",
                    hint="口令太短等于没设",
                )
            )
        if settings.auth_session_ttl_hours <= 0:
            problems.append(
                ConfigProblem(
                    key="AGENT_SESSION_TTL_HOURS",
                    message="登录态有效期必须为正整数",
                    hint="例如 12（小时）",
                )
            )
        if settings.auth_password_iterations < 100_000:
            problems.append(
                ConfigProblem(
                    key="AGENT_AUTH_PASSWORD_ITERATIONS",
                    message="口令哈希迭代数过低",
                    hint="生产环境至少 100000，推荐 600000（OWASP）",
                )
            )

    if settings.llm_provider == "openai":
        if not settings.llm_api_key.strip():
            problems.append(
                ConfigProblem(
                    key="AGENT_LLM_API_KEY",
                    message="未配置 API Key",
                    hint=(
                        "在 .env 中填写 AGENT_LLM_API_KEY；"
                        "或设 AGENT_LLM_PROVIDER=mock 用本地假模型离线跑通流程"
                    ),
                )
            )
        raw_url = settings.llm_base_url
        parsed = urlparse(raw_url)
        if (
            any(char.isspace() for char in raw_url)
            or parsed.scheme not in ("http", "https")
            or not parsed.netloc
        ):
            problems.append(
                ConfigProblem(
                    key="AGENT_LLM_BASE_URL",
                    message=f"base_url 格式不合法: {raw_url!r}",
                    hint=(
                        "需带协议与主机名且不含空格，例如 https://api.deepseek.com/v1 "
                        "或 http://localhost:11434/v1（保持你填写的原样，不要自动补 /v1）"
                    ),
                )
            )
        if not settings.llm_model.strip():
            problems.append(
                ConfigProblem(
                    key="AGENT_LLM_MODEL",
                    message="未配置模型名",
                    hint="例如 deepseek-chat / deepseek-flash / gpt-4o-mini",
                )
            )

    if settings.max_agent_steps < 1:
        problems.append(
            ConfigProblem(
                key="AGENT_MAX_AGENT_STEPS",
                message=f"必须 >= 1，当前 {settings.max_agent_steps}",
                hint="单次提问允许的「模型思考 + 工具调用」轮数上限",
            )
        )
    if settings.max_tool_output_chars < 500:
        problems.append(
            ConfigProblem(
                key="AGENT_MAX_TOOL_OUTPUT_CHARS",
                message=f"过小会导致模型看不到数据，当前 {settings.max_tool_output_chars}",
                hint="建议 >= 500，低于该值截断掉的表格会让模型无法分析",
            )
        )
    if not settings.allowed_suffixes:
        problems.append(
            ConfigProblem(
                key="allowed_suffixes",
                message="未配置任何允许的文件后缀",
                hint="至少需要 .xlsx",
            )
        )

    # ---- 配额（DA-017）----
    # 全部有默认值：老 .env 不新增任何必填项，升级后不会因校验失败起不来
    if settings.quota_enabled:
        for key, value, label in (
            ("AGENT_QUOTA_USER_PER_MINUTE", settings.quota_user_per_minute, "每用户每分钟请求上限"),
            ("AGENT_QUOTA_USER_PER_DAY", settings.quota_user_per_day, "每用户每日请求上限"),
            (
                "AGENT_QUOTA_SESSION_PER_MINUTE",
                settings.quota_session_per_minute,
                "每会话每分钟请求上限",
            ),
            (
                "AGENT_QUOTA_MAX_CONCURRENCY_PER_USER",
                settings.quota_max_concurrency_per_user,
                "每用户并发上限",
            ),
            (
                "AGENT_QUOTA_MAX_CONCURRENCY_GLOBAL",
                settings.quota_max_concurrency_global,
                "全局并发上限",
            ),
            (
                "AGENT_QUOTA_RETRY_AFTER_SECONDS",
                settings.quota_retry_after_seconds,
                "建议等待秒数",
            ),
        ):
            if value < 1:
                problems.append(
                    ConfigProblem(
                        key=key,
                        message=f"{label}必须 >= 1，当前 {value}",
                        hint="设为 0 或负数会让所有请求都被拒绝，通常不是本意",
                    )
                )
        if settings.quota_user_per_minute > settings.quota_user_per_day:
            problems.append(
                ConfigProblem(
                    key="AGENT_QUOTA_USER_PER_MINUTE",
                    message=(
                        f"每分钟上限({settings.quota_user_per_minute})"
                        f"大于每日上限({settings.quota_user_per_day})"
                    ),
                    hint="这会让每日额度在一分钟内就被耗尽，通常是两个值写反了",
                )
            )

    problems.extend(_check_upload_dir(settings.upload_dir))
    return problems


def _check_upload_dir(upload_dir: Path) -> list[ConfigProblem]:
    """上传目录必须存在且可写 —— 不可写时上传接口会 500。"""
    probe = upload_dir / ".write_probe"
    try:
        upload_dir.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return [
            ConfigProblem(
                key="upload_dir",
                message=f"上传目录不可写: {upload_dir}（{exc.strerror or exc}）",
                hint="检查路径是否存在、是否被占用，以及当前用户的写权限",
            )
        ]
    return []


def format_problems(problems: list[ConfigProblem]) -> str:
    """把问题清单格式化成多行可读文本，用于异常信息。"""
    lines = [f"配置校验未通过，共 {len(problems)} 项："]
    for index, problem in enumerate(problems, 1):
        lines.append(f"  {index}. [{problem.key}] {problem.message}")
        if problem.hint:
            lines.append(f"     建议: {problem.hint}")
    return "\n".join(lines)


def runtime_summary(settings: Settings) -> str:
    """启动时打印的一行配置摘要。

    只暴露端点、模型名和 Key 长度，**绝不包含 Key 内容**。
    """
    if settings.llm_provider == "mock":
        llm_part = "provider=mock（离线演示，不联网）"
    else:
        endpoint = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
        key_part = (
            f"Key=已加载({len(settings.llm_api_key)} 字符)"
            if settings.llm_api_key
            else "Key=未配置"
        )
        llm_part = (
            f"provider=openai | 模型={settings.llm_model} | 端点={endpoint} | {key_part}"
        )
    return (
        f"{llm_part} | 上传目录={settings.upload_dir} "
        f"| 最大步数={settings.max_agent_steps} "
        f"| 严格配置={'开' if settings.strict_config else '关'}"
    )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    return settings
