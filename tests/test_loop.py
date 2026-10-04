"""Agent Loop 行为测试：收敛、步数上限、错误自愈、上下文管理。"""

from __future__ import annotations

import logging

import pytest

from app.agent.loop import response_to_assistant_message
from app.llm.base import LLMResponse, ToolCall
from stubs import AlwaysToolLLM, BrokenLLM, WrongSheetLLM


async def test_run_without_session_short_circuits(make_agent) -> None:
    """没有会话时直接返回引导语，不调用模型、不消耗步数。"""
    result = await make_agent().run(None, "分析一下")
    assert result.stop_reason == "no_file"
    assert result.steps == []
    assert "上传" in result.answer


async def test_loop_converges_and_persists_history(make_agent, session) -> None:
    result = await make_agent().run(session, "整体情况如何？")
    assert result.stop_reason == "final"
    assert [s.tool for s in result.steps] == ["list_sheets", "load_sheet", "run_pandas"]
    assert all(s.ok for s in result.steps)
    # 步号连续递增，供前端展示
    assert [s.step for s in result.steps] == [1, 2, 3]
    # 历史落盘：user -> assistant(tool_calls) -> tool... -> assistant(最终答案)
    roles = [m["role"] for m in session.messages]
    assert roles[0] == "user"
    assert roles[-1] == "assistant"
    assert roles.count("tool") == 3
    assert "tool_calls" in session.messages[1]


async def test_tool_failure_does_not_break_loop(make_agent, session) -> None:
    """工具报错 → 错误进 steps 且 ok=False → 循环继续 → 模型自行纠正。"""
    result = await make_agent(WrongSheetLLM()).run(session, "分析")
    assert result.stop_reason == "final"
    assert len(result.steps) == 1
    assert result.steps[0].ok is False
    assert "不存在" in (result.steps[0].error or "")
    assert result.answer == "已根据实际可用信息给出结论。"


async def test_step_limit_triggers_wrap_up(make_agent, session) -> None:
    """模型死循环时必须在上限处停下，并给出阶段性结论。"""
    agent = make_agent(AlwaysToolLLM(), max_agent_steps=3)
    result = await agent.run(session, "分析")
    assert result.stop_reason == "max_steps"
    assert len(result.steps) == 3
    assert "最大" in result.answer
    # 收尾后历史仍可用（未被丢弃）
    assert session.messages


async def test_llm_error_is_reported_as_stop_reason(make_agent, session, caplog) -> None:
    caplog.set_level(logging.ERROR)
    result = await make_agent(BrokenLLM()).run(session, "分析")
    assert result.stop_reason == "llm_error"
    # DA-003 脱敏要求：厂商/网络异常细节不得进入 answer
    assert "模拟的模型故障" not in result.answer
    assert "模型服务调用失败" in result.answer
    # 但必须留痕，否则无法排查
    assert "模拟的模型故障" in caplog.text
    # 模型故障时至少保留用户提问，不能留下半截工具轨迹
    assert [m["role"] for m in session.messages] == ["user"]


async def test_history_carries_across_turns(make_agent, session) -> None:
    agent = make_agent()
    await agent.run(session, "第一轮")
    first = len(session.messages)
    await agent.run(session, "第二轮")
    assert len(session.messages) > first
    user_messages = [m for m in session.messages if m["role"] == "user"]
    assert len(user_messages) == 2
    assert user_messages[0]["content"] == "第一轮"
    assert user_messages[1]["content"] == "第二轮"


async def test_dataframe_cache_is_reused_across_turns(make_agent, session) -> None:
    """同一会话内不应重复读盘：frames 缓存必须被复用。"""
    agent = make_agent()
    await agent.run(session, "第一轮")
    assert "明细" in session.frames
    loaded = session.frames["明细"]
    await agent.run(session, "第二轮")
    assert session.frames["明细"] is loaded


# ------------------------------------------------------- 消息格式转换
def test_response_to_assistant_message_keeps_tool_calls() -> None:
    response = LLMResponse(
        content="",
        tool_calls=[ToolCall(id="c1", name="list_sheets", arguments={"sheet": "明细"})],
    )
    message = response_to_assistant_message(response)
    assert message["role"] == "assistant"
    call = message["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "list_sheets"
    assert call["function"]["arguments"] == '{"sheet": "明细"}'


def test_response_to_assistant_message_without_tool_calls() -> None:
    message = response_to_assistant_message(LLMResponse(content="结论"))
    assert message == {"role": "assistant", "content": "结论"}
    assert "tool_calls" not in message
