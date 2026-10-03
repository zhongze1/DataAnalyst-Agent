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
