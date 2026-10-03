"""工具抽象：Tool 协议、执行上下文、结果格式化。

一个「工具」= 名字 + 描述 + JSON Schema 参数 + 同步执行函数。
工具执行结果统一格式化为**字符串**回传给模型（模型只能读文本）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from app.config import get_settings


class ToolError(Exception):
    """工具执行失败。

    注意：这类异常**不会中断 Agent Loop**，错误信息会作为工具结果回传给模型，
    让模型自行纠正（例如 sheet 名写错时改用 list_sheets 重新选择）。
    """


@dataclass
class ToolContext:
    """一次提问过程中，Agent 持有的数据上下文。

    ``frames`` 以「工作表名」为键缓存已加载的 DataFrame，避免重复读盘。
    """

    session_id: str
    file_path: str | None = None
    file_name: str | None = None
    sheets: list[str] = field(default_factory=list)
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    active_sheet: str | None = None

    def get_frame(self, sheet: str | None = None) -> pd.DataFrame:
        """取出（必要时加载）某个工作表的 DataFrame。"""
        from app.storage.session_store import SessionStore  # 延迟导入避免循环依赖

        name = (sheet or self.active_sheet or (self.sheets[0] if self.sheets else None))
        if not name:
            raise ToolError("尚未加载任何数据，请先调用 load_sheet。")
        if name not in self.frames:
            if self.sheets and name not in self.sheets:
                raise ToolError(
                    f"工作表 {name!r} 不存在。可用工作表: {', '.join(self.sheets) or '无'}"
                )
            self.frames[name] = SessionStore.read_frame(self.file_path, name)
        self.active_sheet = name
        return self.frames[name]


@dataclass
class Tool:
    """可被模型调用的函数。"""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def run(self, ctx: ToolContext, arguments: dict[str, Any] | None = None) -> str:
        """校验参数 → 执行 → 格式化结果。任何失败都转成 ToolError 文案。"""
        arguments = arguments or {}
        self._validate(arguments)
        value = self.func(ctx, **arguments)
        return format_value(value)

    def _validate(self, arguments: dict[str, Any]) -> None:
        schema = self.parameters or {}
        properties: dict[str, Any] = schema.get("properties", {})
        required: list[str] = schema.get("required", [])

        unknown = set(arguments) - set(properties)
        if unknown:
            raise ToolError(
                f"工具 {self.name} 不支持参数 {sorted(unknown)}；"
                f"可用参数: {sorted(properties) or '无'}"
            )
        missing = [key for key in required if arguments.get(key) in (None, "")]
        if missing:
            raise ToolError(f"工具 {self.name} 缺少必填参数: {missing}")

        for key, spec in properties.items():
            if key not in arguments:
                continue
            expected = spec.get("type")
            value = arguments[key]
            if expected == "string" and not isinstance(value, str):
                raise ToolError(f"参数 {key} 应为字符串，收到 {type(value).__name__}")
            if expected == "integer" and isinstance(value, bool):
                raise ToolError(f"参数 {key} 应为整数")


def format_value(value: Any) -> str:
    """把任意执行结果转成适合喂给模型的文本（并做长度截断）。"""
    if isinstance(value, str):
        text = value
    elif isinstance(value, pd.DataFrame):
        text = _format_frame(value)
    elif isinstance(value, pd.Series):
        text = f"[Series] len={len(value)}\n{value.head(20).to_string()}"
    elif isinstance(value, dict):
        text = "\n".join(f"{k}: {v}" for k, v in value.items())
    else:
        text = str(value)
    return _truncate(text)


def _format_frame(df: pd.DataFrame) -> str:
    rows = df.head(20)
    header = f"[DataFrame] shape={df.shape} | columns={list(df.columns)}"
    if df.empty:
        return f"{header}\n(空表)"
    return f"{header}\n{rows.to_string()}"


def _truncate(text: str) -> str:
    limit = get_settings().max_tool_output_chars
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... [内容过长已截断，共 {len(text)} 字符]"
