"""测试专用的 LLM 桩。

全部继承 MockLLMClient —— 测试永不触碰真实模型接口。
放在独立模块里，避免测试文件反向 import conftest。
"""

from __future__ import annotations

import asyncio

from app.llm.base import LLMError, LLMResponse, LLMStreamEvent, ToolCall
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


class CountingLLM(MockLLMClient):
    """记录 ``chat`` 被真实调用的次数（证明被拒绝的请求没有打到模型）。"""

    def __init__(self) -> None:
        super().__init__()
        self.chat_calls = 0

    async def chat(self, messages, tools=None) -> LLMResponse:
        self.chat_calls += 1
        return await super().chat(messages, tools)


class UsageLLM(MockLLMClient):
    """每轮都回报 usage 的模型：先调 ``tool_rounds`` 次工具，然后给结论。

    用于验证「usage 逐轮累加」：改造前只会留下**最后一次**的值。
    """

    def __init__(
        self,
        tool_rounds: int = 2,
        prompt: int = 10,
        completion: int = 5,
        missing_rounds: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.tool_rounds = tool_rounds
        self.prompt = prompt
        self.completion = completion
        self.missing_rounds = missing_rounds
        self.calls = 0

    def _usage(self) -> dict:
        if self.calls in self.missing_rounds:
            return {}
        return {
            "prompt_tokens": self.prompt,
            "completion_tokens": self.completion,
            "total_tokens": self.prompt + self.completion,
        }

    async def chat(self, messages, tools=None) -> LLMResponse:
        self.calls += 1
        if self.calls <= self.tool_rounds:
            return LLMResponse(
                tool_calls=[ToolCall(id=f"u{self.calls}", name="list_sheets", arguments={})],
                usage=self._usage(),
            )
        return LLMResponse(content="基于累计用量给出的结论", usage=self._usage())


class StreamingUsageLLM(UsageLLM):
    """模拟「厂商在流结束时才返回 usage」：usage 只能从 done 事件拿到。"""

    async def chat_stream(self, messages, tools=None):
        response = await self.chat(messages, tools)
        yield LLMStreamEvent(kind="delta", text=response.content)
        yield LLMStreamEvent(kind="done", response=response)


class UsageThenBrokenLLM(UsageLLM):
    """第一轮正常（回报 usage），第二轮直接抛错 —— 验证失败前的用量不被吞掉。"""

    async def chat(self, messages, tools=None) -> LLMResponse:
        if self.calls >= 1:
            raise LLMError("模拟的模型故障（第二轮）")
        return await super().chat(messages, tools)


class BlockingLLM(MockLLMClient):
    """进入后阻塞一段时间，用来制造「请求正在执行中」的并发场景。"""

    def __init__(self, delay: float = 0.3) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.delay = delay

    async def chat(self, messages, tools=None) -> LLMResponse:
        self.entered.set()
        await asyncio.sleep(self.delay)
        return LLMResponse(
            content="慢结论",
            usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        )
