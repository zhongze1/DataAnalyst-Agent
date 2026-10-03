"""LLM 抽象层：定义 Agent Loop 依赖的最小接口。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class LLMError(RuntimeError):
    """模型调用失败（网络、鉴权、返回格式异常等）。"""


@dataclass
class ToolCall:
    """一次工具调用的原始请求。"""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """模型的一次回复：要么给最终答案，要么要求调用工具。"""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@runtime_checkable
class LLMClient(Protocol):
    """支持 function calling 的聊天模型。"""

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """给定对话历史（OpenAI 格式）与工具 schema，返回模型回复。"""
        ...


def parse_tool_arguments(raw: Any) -> dict[str, Any]:
    """模型返回的 arguments 可能是 dict，也可能是 JSON 字符串。"""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError) as exc:  # pragma: no cover - 防御性
        raise LLMError(f"工具参数不是合法 JSON: {raw!r}") from exc
    if not isinstance(parsed, dict):
        raise LLMError(f"工具参数应为 JSON 对象: {raw!r}")
    return parsed
