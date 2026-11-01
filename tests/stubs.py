"""测试专用桩与助手。

LLM 桩全部继承 MockLLMClient —— 测试永不触碰真实模型接口。
知识库工件助手（DA-018）也放这里：它现造一份合成索引写进 tmp_path，
**不依赖仓库里被 gitignore 的 ``data/knowledge/``**，让检索用例完全确定。
放在独立模块里，避免测试文件反向 import conftest。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

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


class WrapUpFailingLLM(MockLLMClient):
    """一直要工具，直到收到「已达到最大工具调用次数」的收尾提示就抛错。

    用于验证：收尾调用失败时，异常里的厂商端点等内部信息**不得**进入用户答案。
    """

    async def chat(self, messages, tools=None) -> LLMResponse:
        last = messages[-1] if messages else {}
        if last.get("role") == "user" and "最大工具调用次数" in str(last.get("content", "")):
            raise LLMError("调用 https://secret-endpoint.vendor.com/v1 失败")
        return LLMResponse(
            tool_calls=[ToolCall(id="wrap-fail", name="list_sheets", arguments={})]
        )


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


class PhoneLLM(MockLLMClient):
    """专门制造敏感值：第 1 步用 run_pandas 造出含手机号的表，第 2 步把手机号写进结论。

    用于验证 DA-015 的**双通路**：对外（steps / answer）必须脱敏，
    而喂给模型的 tool 消息必须仍是原文，否则模型会算错。
    ``tool_contents`` 记录模型实际看到的工具输出。
    """

    def __init__(self, phone: str = "13800138000") -> None:
        super().__init__()
        self.phone = phone
        self.calls = 0
        self.tool_contents: list[str] = []

    async def chat(self, messages, tools=None) -> LLMResponse:
        self.calls += 1
        self.tool_contents.extend(
            str(m.get("content") or "") for m in messages if m.get("role") == "tool"
        )
        if self.calls == 1:
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="phone-1",
                        name="run_pandas",
                        arguments={
                            "sheet": SHEET_MAIN,
                            "code": (
                                "out = pd.DataFrame({'客户手机': ['13800138000']})\n"
                                "out"
                            ),
                        },
                    )
                ]
            )
        return LLMResponse(content="客户 13800138000 的明细如下。")


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


# ------------------------------------------------------------ 知识库工件（DA-018）
KNOWLEDGE_DIM = 4


def knowledge_row(
    chunk_id: str,
    doc_id: str,
    section: str,
    text: str,
    vector: list[float] | None,
    **extra: object,
) -> dict[str, object]:
    """一行合成索引。``vector=None`` 表示"这一片没有向量"（用于测降级）。"""
    row: dict[str, object] = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "source_path": f"corpus/{doc_id}.md",
        "section": section,
        "text": text,
        "line_start": 1,
        "line_end": 3,
        "content_hash": f"hash-{chunk_id}",
        "repeated_header": "",
    }
    if vector is not None:
        row["vector"] = vector
    row.update(extra)
    return row


# 三片语料：一片讲打码（正文里**故意**放一个真实手机号，用来验证打码与双通路），
# 一片讲口径，一片讲架构。向量两两正交，便于把余弦结果算成人眼可核对的值。
KNOWLEDGE_CHUNKS: list[dict[str, object]] = [
    knowledge_row(
        "c1",
        "DA-015-敏感字段打码",
        "打码规则",
        "敏感字段打码：联系方式 13800138000 显示为 138****8000。",
        [1.0, 0.0, 0.0, 0.0],
    ),
    knowledge_row(
        "c2",
        "指标定义",
        "客单价",
        "客单价 = 销售额 / 订单数；分母是订单数。",
        [0.0, 1.0, 0.0, 0.0],
    ),
    knowledge_row(
        "c3",
        "架构说明",
        "请求链路",
        "请求先过认证，再过配额，最后进入 Agent Loop。",
        [0.0, 0.0, 1.0, 0.0],
    ),
]


def write_knowledge_artifacts(
    base: Path,
    rows: list[dict[str, object]],
    *,
    dim: int = KNOWLEDGE_DIM,
    manifest: dict[str, object] | None = None,
) -> Path:
    """把索引与 manifest 写进 ``base``，返回索引路径。"""
    base.mkdir(parents=True, exist_ok=True)
    index = base / "knowledge_index.jsonl"
    index.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    payload: dict[str, object] = {
        "built_at": "2026-01-01 00:00",
        "space": f"api/test/dim={dim}",
        "provider": "api",
        "model": "test-embedding",
        "dim": dim,
        "chunks": len(rows),
        "ngram": 2,
        "corpus": [],
    }
    payload.update(manifest or {})
    (base / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    return index


class FakeQueryEmbedder:
    """假查询向量化：永远返回同一个向量，并记录被调用次数。

    用来在**离线**条件下把混合检索跑起来（真实端点由 httpx 挡在测试之外）。
    """

    def __init__(self, vector: list[float]) -> None:
        self.vector = list(vector)
        self.calls = 0
        self.seen: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls += 1
        self.seen.append(text)
        return list(self.vector)


class KnowledgeLLM(MockLLMClient):
    """第一轮调 ``knowledge_search``，第二轮给结论；记录模型实际看到的工具输出。

    用于验证 DA-018 第 11 条：步骤结果**对外已打码**，而喂给模型的仍是原文。
    """

    def __init__(self, query: str) -> None:
        super().__init__()
        self.query = query
        self.calls = 0
        self.tool_contents: list[str] = []

    async def chat(self, messages, tools=None) -> LLMResponse:
        self.calls += 1
        self.tool_contents.extend(
            str(m.get("content") or "") for m in messages if m.get("role") == "tool"
        )
        if self.calls == 1:
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="kb-1",
                        name="knowledge_search",
                        arguments={"query": self.query},
                    )
                ]
            )
        return LLMResponse(content="已根据项目文档给出结论。")
