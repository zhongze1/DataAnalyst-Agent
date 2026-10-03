"""工具注册表：集中管理所有可被模型调用的工具，并负责分发执行。"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable

from app.tools.base import Tool, ToolContext, ToolError

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def add(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"工具名重复: {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def register(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """装饰器：把函数注册成一个工具。"""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.add(
                Tool(
                    name=name,
                    description=description,
                    parameters=parameters or {"type": "object", "properties": {}},
                    func=func,
                )
            )
            return func

        return decorator

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def openai_schemas(self) -> list[dict[str, Any]]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def run(
        self,
        name: str,
        arguments: dict[str, Any] | None,
        ctx: ToolContext,
    ) -> tuple[bool, str]:
        """执行工具，返回 (是否成功, 文本结果)。**永不抛异常**。"""
        tool = self._tools.get(name)
        if tool is None:
            return False, f"工具不存在: {name}。可用工具: {', '.join(self.names())}"
        try:
            return True, tool.run(ctx, arguments)
        except ToolError as exc:
            logger.info("tool %s 业务失败: %s", name, exc)
            return False, f"[ToolError] {exc}"
        except Exception as exc:  # pragma: no cover - 兜底，防止工具把整个 Loop 打断
            logger.exception("tool %s 未预期异常", name)
            return False, f"[RuntimeError] {type(exc).__name__}: {exc}"


registry = ToolRegistry()
# 简写：@tool(name=..., description=..., parameters=...)
tool = registry.register
