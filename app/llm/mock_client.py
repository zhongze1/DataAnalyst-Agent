"""本地假模型：不联网、不需要 API Key，用于离线验证 Agent Loop 是否跑通。

它会按固定剧本依次调用 ``list_sheets`` → ``load_sheet`` → ``run_pandas``，
最后基于工具结果给出结论，从而完整覆盖「模型决策 → 工具执行 → 结果回传 → 收敛」闭环。
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from app.llm.base import LLMResponse, ToolCall


class MockLLMClient:
    """按剧本推进的确定性客户端。"""

    def __init__(self, model: str = "mock-echo") -> None:
        self.model = model

    async def aclose(self) -> None:  # 与真实客户端接口保持一致
        return None

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        available = {t["function"]["name"] for t in (tools or [])}

        if "list_sheets" in available and not tool_messages:
            return self._call("list_sheets", {})

        if "load_sheet" in available and len(tool_messages) == 1:
            sheet = self._first_sheet(tool_messages[-1].get("content", ""))
            return self._call("load_sheet", {"sheet": sheet} if sheet else {})

        if "run_pandas" in available and len(tool_messages) == 2:
            return self._call(
                "run_pandas",
                {
                    "code": "summary = {\n"
                    "    'shape': df.shape,\n"
                    "    '列名': list(df.columns),\n"
                    "    '缺失值总数': int(df.isna().sum().sum()),\n"
                    "}\n"
                    "out = summary\n"
                    "out"
                },
            )

        # 只取「本轮」的工具结果：最后一次 user 消息之后的所有 tool 消息
        last_user = max(
            (i for i, m in enumerate(messages) if m.get("role") == "user"),
            default=0,
        )
        run_tools = [m for m in messages[last_user:] if m.get("role") == "tool"]
        detail = "\n\n".join(
            f"**{i + 1}. {m.get('name', 'tool')}**\n\n```\n{m.get('content', '')}\n```"
            for i, m in enumerate(run_tools)
        )
        return LLMResponse(
            content=(
                "当前为 **Mock 模式**（未配置真实模型），以下结论由本地剧本生成，仅用于验证链路：\n\n"
                f"{detail}\n\n"
                "配置 `AGENT_LLM_API_KEY` 后重启服务，即可得到真实的自主分析结果。"
            )
        )

    @staticmethod
    def _call(name: str, arguments: dict[str, Any]) -> LLMResponse:
        return LLMResponse(
            tool_calls=[ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name=name, arguments=arguments)]
        )

    @staticmethod
    def _first_sheet(content: str) -> str | None:
        match = re.search(r"\[Sheets\]\s*(.+)", content)
        if not match:
            return None
        return match.group(1).split(",")[0].strip() or None
