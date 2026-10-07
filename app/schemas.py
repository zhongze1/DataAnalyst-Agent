"""接口层数据模型（Pydantic schemas）。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------- 文件相关
class FileInfo(BaseModel):
    """一份已上传的 Excel 文件的元信息。"""

    session_id: str
    filename: str
    size_bytes: int
    sheets: list[str]
    active_sheet: str | None = None


class UploadResponse(BaseModel):
    file: FileInfo
    message: str = Field(default="上传成功")


# ---------------------------------------------------------------- 会话/对话
class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(min_length=1, max_length=4000)


class ToolStep(BaseModel):
    """Agent Loop 中的一次工具调用记录，用于前端展示执行过程。"""

    step: int
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    result: str = ""
    # 面向用户的中文提示（已脱敏）。原始错误只给模型与日志，不出接口。
    error: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    steps: list[ToolStep] = Field(default_factory=list)
    stop_reason: Literal["final", "max_steps", "llm_error", "no_file"] = "final"
    usage: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str


# ----------------------------------------------------------------- 会话历史
class SessionSummary(BaseModel):
    """历史会话列表项（不含完整对话，避免列表响应过大）。"""

    session_id: str
    filename: str
    sheets: list[str] = Field(default_factory=list)
    created_at: str
    updated_at: str
    turn_count: int = 0
    size_bytes: int = 0
    preview: str = Field(default="", description="最后一条提问的摘要")


class SessionDetail(BaseModel):
    """会话详情：含完整消息序列，用于刷新页面后恢复界面。"""

    session_id: str
    filename: str
    sheets: list[str] = Field(default_factory=list)
    created_at: str
    updated_at: str
    turn_count: int = 0
    size_bytes: int = 0
    messages: list[dict[str, Any]] = Field(default_factory=list)
    last_step_count: int = 0


# ----------------------------------------------------------------- 错误响应
class ErrorDetail(BaseModel):
    """统一错误体。``message`` 保证是脱敏后的人工文案。"""

    code: str = Field(description="机器可读错误码，如 FILE_PARSE_ERROR")
    message: str = Field(description="面向用户的可读文案")
    request_id: str | None = Field(default=None, description="用于服务端日志排查")


class ErrorResponse(BaseModel):
    error: ErrorDetail


# 挂到 APIRouter 上，让 /docs 展示统一的错误结构
ERROR_RESPONSES: dict[int | str, dict] = {
    400: {"model": ErrorResponse, "description": "请求不合法（文件类型/为空/无法解析）"},
    404: {"model": ErrorResponse, "description": "会话或接口不存在"},
    413: {"model": ErrorResponse, "description": "文件超过大小限制"},
    500: {"model": ErrorResponse, "description": "服务器内部错误"},
}
