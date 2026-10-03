"""测试专用的 LLM 桩。

全部继承 MockLLMClient —— 测试永不触碰真实模型接口。
放在独立模块里，避免测试文件反向 import conftest。
"""

from __future__ import annotations

from app.llm.base import LLMError, LLMResponse, ToolCall
from app.llm.mock_client import MockLLMClient

# 样本工作簿里的工作表名，测试与断言共用
SHEET_MAIN = "明细"
SHEET_SUMMARY = "摘要"


class AlwaysToolLLM(MockLLMClient):
    """永远请求工具调用、永远不给最终答案 —— 用于触发步数上限。"""

    def __init__(self, tool_name: str = "list_sheets") -> None:
        super().__init__()
        self.tool_name = tool_name
        self.calls = 0

    async def chat(self, messages, tools=None) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            tool_calls=[ToolCall(id=f"loop-{self.calls}", name=self.tool_name, arguments={})]
        )


class WrongSheetLLM(MockLLMClient):
    """第一次调用就传错工作表名，第二次给结论 —— 用于验证工具错误自愈。"""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def chat(self, messages, tools=None) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="wrong-sheet",
                        name="load_sheet",
                        arguments={"sheet": "不存在的工作表"},
                    )
                ]
            )
        return LLMResponse(content="已根据实际可用信息给出结论。")


class BrokenLLM(MockLLMClient):
    """直接抛 LLMError —— 用于验证模型故障分支。"""

    async def chat(self, messages, tools=None) -> LLMResponse:
        raise LLMError("模拟的模型故障")
