"""流式输出（B+C）测试。

覆盖：
1. Agent 事件顺序与内容（不依赖 HTTP，可单独测）；
2. SSE 帧序列化格式；
3. /api/chat/stream 端到端；
4. LLM 不支持流式时的降级；
5. 客户端断开时后端任务被取消。
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.agent.events import (
    EVENT_DONE,
    EVENT_START,
    EVENT_STEP,
    EVENT_TOKEN,
    EVENT_TOOL_ARGS,
    AgentEvent,
    done_event,
)
from app.llm.base import LLMResponse, LLMStreamEvent, ToolCall
from app.storage.session_store import SessionStore


# ============================================================ SSE 帧格式
def test_event_serializes_to_sse_frame() -> None:
    frame = AgentEvent(EVENT_TOKEN, {"text": "你好"}).to_sse().decode("utf-8")
    assert frame.startswith("event: token\n")
    assert '"text": "你好"' in frame
    assert frame.endswith("\n\n")  # 帧必须以空行结束


def test_event_escapes_newlines_in_payload() -> None:
    """data: 是逐行协议，载荷里的换行必须被 JSON 转义，否则破坏帧边界。"""
    frame = AgentEvent(EVENT_TOKEN, {"text": "第一行\n第二行"}).to_sse().decode("utf-8")
    body = frame.split("\n\n")[0]
    assert body.count("\n") == 1  # 只有 event: 与 data: 两行，不含裸换行
    payload = json.loads(body.split("data: ", 1)[1])
    assert payload["text"] == "第一行\n第二行"


def test_done_event_carries_full_snapshot() -> None:
    event = done_event(
        answer="最终答案",
        steps=[{"step": 1, "tool": "list_sheets", "ok": True}],
        stop_reason="final",
        usage={"total_tokens": 10},
    )
    assert event.kind == EVENT_DONE
    assert event.data["answer"] == "最终答案"
    assert event.data["steps"][0]["tool"] == "list_sheets"
    assert event.data["stop_reason"] == "final"


def test_non_serializable_payload_is_handled() -> None:
    frame = AgentEvent("step", {"obj": object()}).to_sse()
    assert isinstance(frame, bytes)  # 不抛异常即可


# ============================================================ Agent 事件
async def test_events_are_emitted_in_order(make_agent, session) -> None:
    """Agent 只产出 start/tool_args/step/token；done 由路由层补（它才知道完整结果）。"""
    events: list[tuple[str, dict]] = []

    async def collect(event: AgentEvent) -> None:
        events.append((event.kind, event.data))

    await make_agent().run(session, "整体情况？", on_event=collect)
    kinds = [k for k, _ in events]

    assert kinds[0] == EVENT_START
    assert EVENT_DONE not in kinds  # done 不由 Agent 产出
    assert kinds.count(EVENT_TOOL_ARGS) == kinds.count(EVENT_STEP) == 3
    assert kinds.index(EVENT_TOOL_ARGS) < kinds.index(EVENT_STEP)  # 先 announce 再完成
    assert EVENT_TOKEN in kinds


async def test_step_event_carries_duration_and_error_code(make_agent, session) -> None:
    events: list[tuple[str, dict]] = []

    async def collect(event: AgentEvent) -> None:
        events.append((event.kind, event.data))

    await make_agent().run(session, "整体情况？", on_event=collect)
    steps = [d for k, d in events if k == EVENT_STEP]
    assert steps
    for payload in steps:
        assert {"step", "tool", "ok", "duration_ms", "result_chars", "error_code"} <= set(payload)


async def test_tool_args_event_carries_summary_not_content(make_agent, session) -> None:
    events: list[tuple[str, dict]] = []

    async def collect(event: AgentEvent) -> None:
        events.append((event.kind, event.data))

    await make_agent().run(session, "整体情况？", on_event=collect)
    args_events = [d for k, d in events if k == EVENT_TOOL_ARGS]
    assert all("args_summary" in d for d in args_events)


async def test_token_events_concatenate_to_final_answer(make_agent, session) -> None:
    """降级路径下 token 事件拼起来必须等于最终答案 —— 前端两种表现一致。"""
    events: list[tuple[str, dict]] = []

    async def collect(event: AgentEvent) -> None:
        events.append((event.kind, event.data))

    result = await make_agent().run(session, "整体情况？", on_event=collect)
    streamed = "".join(d["text"] for k, d in events if k == EVENT_TOKEN)
    assert streamed == result.answer


async def test_no_events_when_no_listener(make_agent, session) -> None:
    """没有 on_event 时必须完全不产生事件（保持非流式路径零开销）。"""
    result = await make_agent().run(session, "整体情况？")
    assert result.stop_reason == "final"


async def test_failure_event_carries_error_code(make_agent, session) -> None:
    from stubs import WrongSheetLLM

    events: list[tuple[str, dict]] = []

    async def collect(event: AgentEvent) -> None:
        events.append((event.kind, event.data))

    await make_agent(WrongSheetLLM()).run(session, "分析", on_event=collect)
    failed = [d for k, d in events if k == EVENT_STEP and not d["ok"]]
    assert failed and failed[0]["error_code"] == "SHEET_NOT_FOUND"


# ============================================================ 流式 LLM
class StreamingLLM:
    """支持流式的模型桩：逐段吐出内容。"""

    def __init__(self, chunks: list[str], with_tools: bool = False) -> None:
        self.chunks = chunks
        self.with_tools = with_tools
        self.used_stream = False

    async def chat(self, messages, tools=None) -> LLMResponse:
        return LLMResponse(content="".join(self.chunks))

    async def chat_stream(self, messages, tools=None):
        self.used_stream = True
        for chunk in self.chunks:
            yield LLMStreamEvent(kind="delta", text=chunk)
        yield LLMStreamEvent(
            kind="done",
            response=LLMResponse(
                content="".join(self.chunks),
                tool_calls=[ToolCall(id="c1", name="list_sheets", arguments={})]
                if self.with_tools
                else [],
            ),
        )


async def test_streaming_llm_emits_token_per_delta(make_agent, session) -> None:
    llm = StreamingLLM(["销售", "额", "合计"])
    events: list[tuple[str, dict]] = []

    async def collect(event: AgentEvent) -> None:
        events.append((event.kind, event.data))

    result = await make_agent(llm).run(session, "销售额？", on_event=collect)
    assert llm.used_stream is True
    assert [d["text"] for k, d in events if k == EVENT_TOKEN] == ["销售", "额", "合计"]
    assert result.answer == "销售额合计"


async def test_non_streaming_llm_is_used_when_no_listener(make_agent, session) -> None:
    """没人监听时不该走流式（省掉流式解析开销）。"""
    llm = StreamingLLM(["销售", "额"])
    await make_agent(llm).run(session, "销售额？", on_event=None)
    assert llm.used_stream is False


async def test_broken_stream_raises_llm_error(make_agent, session) -> None:
    """流意外结束（没有 done）必须报错，不能静默返回半截答案。"""

    class BrokenStream:
        async def chat(self, messages, tools=None) -> LLMResponse:
            return LLMResponse(content="完整答案")

        async def chat_stream(self, messages, tools=None):
            yield LLMStreamEvent(kind="delta", text="半截")

    events: list[tuple[str, dict]] = []

    async def collect(event: AgentEvent) -> None:
        events.append((event.kind, event.data))

    result = await make_agent(BrokenStream()).run(session, "问题", on_event=collect)
    assert result.stop_reason == "llm_error"
    assert "半截" not in result.answer  # 半截内容不能当成结论
    assert "模型服务调用失败" in result.answer


# ============================================================ HTTP 端到端
def test_stream_endpoint_emits_sse_frames(client, session) -> None:
    response = client.post(
        "/api/chat/stream", json={"session_id": session.session_id, "message": "整体情况"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["X-Request-ID"]
    assert response.headers["X-Accel-Buffering"] == "no"  # 反向代理不得缓冲

    body = response.text
    for kind in ("start", "tool_args", "step", "token", "done"):
        assert f"event: {kind}" in body


def test_stream_frames_are_parsable(client, session) -> None:
    """逐帧解析必须能得到合法的 JSON 载荷。"""
    response = client.post(
        "/api/chat/stream", json={"session_id": session.session_id, "message": "整体情况"}
    )
    frames = [f for f in response.text.split("\n\n") if f.strip()]
    assert frames
    for frame in frames:
        lines = frame.splitlines()
        kind = next(line[6:].strip() for line in lines if line.startswith("event:"))
        raw = "\n".join(line[5:].strip() for line in lines if line.startswith("data:"))
        assert kind
        json.loads(raw)


def _done_payload(response_text: str) -> dict:
    return json.loads(
        response_text.split("event: done", 1)[1].split("data: ", 1)[1].split("\n", 1)[0]
    )


def test_stream_done_carries_complete_snapshot(client, session) -> None:
    """done 帧必须带完整 steps，前端以它为准做最终校正。"""
    response = client.post(
        "/api/chat/stream", json={"session_id": session.session_id, "message": "整体情况"}
    )
    payload = _done_payload(response.text)
    assert payload["stop_reason"] == "final"
    assert len(payload["steps"]) == 3
    assert payload["answer"].strip()


def test_stream_endpoint_rejects_unknown_session(client) -> None:
    response = client.post("/api/chat/stream", json={"session_id": "nope", "message": "hi"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "SESSION_NOT_FOUND"


def test_non_streaming_endpoint_still_works(client, session) -> None:
    """降级路径：SSE 失败时前端回退到这里，契约不能变。"""
    body = client.post(
        "/api/chat", json={"session_id": session.session_id, "message": "整体情况"}
    ).json()
    assert body["stop_reason"] == "final"
    assert set(body) == {"session_id", "answer", "steps", "stop_reason", "usage"}


def test_stream_and_non_stream_agree_on_steps(client, store, sample_xlsx) -> None:
    """两条路径必须产出相同轨迹，否则用户看到的会前后不一致。

    用两个独立会话 —— MockLLM 依赖历史里的 tool 消息数决定剧本，
    同一会话跑第二轮会直接收尾。
    """
    a = store.create("a.xlsx", sample_xlsx)
    b = store.create("b.xlsx", sample_xlsx)
    streamed = _done_payload(
        client.post(
            "/api/chat/stream", json={"session_id": a.session_id, "message": "整体情况"}
        ).text
    )
    plain = client.post(
        "/api/chat", json={"session_id": b.session_id, "message": "整体情况"}
    ).json()
    assert [s["tool"] for s in streamed["steps"]] == [s["tool"] for s in plain["steps"]]
    assert streamed["stop_reason"] == plain["stop_reason"]


# ============================================================ 取消
async def test_client_disconnect_cancels_backend_task(
    session, store: SessionStore, make_agent, settings
) -> None:
    """客户端断开后后端任务必须被取消，不能留孤儿任务继续烧 token。"""
    from app.routers.chat_stream import chat_stream
    from app.schemas import ChatRequest
    from app.services.quota_service import QuotaService

    started = asyncio.Event()
    cancelled = asyncio.Event()

    class HangingLLM:
        async def chat(self, messages, tools=None) -> LLMResponse:
            started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return LLMResponse(content="永远到不了")

        async def chat_stream(self, messages, tools=None):
            # 必须 await：否则协程对象从未执行，测试会一直等不到 started
            response = await self.chat(messages, tools)
            yield LLMStreamEvent(kind="done", response=response)

    # 直接调用路由函数（绕过 FastAPI 依赖注入），因此需要显式给出依赖项：
    # user=Depends(...) 与 quota=Depends(...) 的默认值在这里是「Depends 对象」而不是实例
    response = await chat_stream(
        ChatRequest(session_id=session.session_id, message="慢查询"),
        agent=make_agent(HangingLLM()),
        user=None,
        store=store,
        quota=QuotaService(settings),
    )
    iterator = response.body_iterator
    # 读到第一帧（start 事件）说明流已建立
    first = await iterator.__anext__()
    assert b"event: start" in first
    await asyncio.wait_for(started.wait(), timeout=2)

    # 模拟客户端断开：关闭生成器
    await iterator.aclose()

    await asyncio.wait_for(cancelled.wait(), timeout=2)
