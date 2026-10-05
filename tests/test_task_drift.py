"""任务漂移告警测试（DA-009 低优先级项）。

这是**观测**手段：检测「答非所问」并写 warning 日志，不阻断任何流程。
关键词匹配必然有误报，所以只用于事后发现，不作为正确性保证。
"""

from __future__ import annotations

import logging

from app.agent.loop import _question_tokens, detect_task_drift
from app.llm.base import LLMResponse, ToolCall


# ============================================================ 关键词抽取
def test_tokens_are_extracted_from_question() -> None:
    """中文无空格，必须靠 2 字滑窗才能切出「张三」。"""
    tokens = _question_tokens("查一下张三这个用户的明细")
    assert "张三" in tokens
    assert "用户" in tokens
    assert "明细" in tokens


def test_ascii_tokens_are_kept_whole() -> None:
    """字母/数字串不应被切碎：SKU0018 要么整体出现，要么不出现。"""
    tokens = _question_tokens("查一下 SKU0018 的明细")
    assert "SKU0018" in tokens


def test_single_char_tokens_are_dropped() -> None:
    """单字无区分度，不应参与比对。"""
    assert all(len(t) >= 2 for t in _question_tokens("查 a 数据 b"))


def test_pure_punctuation_yields_nothing() -> None:
    """全角标点不是汉字，不能被当成候选词。"""
    assert _question_tokens("？？？") == []
    assert _question_tokens("，，、") == []
    assert _question_tokens("hello!!") == ["hello"]


# ============================================================ 漂移判定
def test_drift_detected_when_answer_ignores_question() -> None:
    """复现线上场景：问张三，答门店排名。"""
    drift = detect_task_drift(
        "查一下张三这个用户的明细",
        "按销售额合计排序，销售额排名前 10 的门店如下……",
        step_count=14,
    )
    assert drift is not None
    assert "疑似任务漂移" in drift
    assert "张三" in drift  # 告警里要带上关键词，便于排查


def test_no_drift_when_answer_addresses_question() -> None:
    assert detect_task_drift("查一下张三的明细", "未找到张三，数据中没有用户维度", 14) is None


def test_no_drift_below_step_threshold() -> None:
    """工具调用次数少不算漂移 —— 简短问答很常见。"""
    assert detect_task_drift("查一下张三", "销售额前 10 门店……", 2) is None


def test_no_drift_when_question_has_no_keywords() -> None:
    assert detect_task_drift("？？", "任何回答", 20) is None


# ============================================================ 接入 Loop
class DriftingLLM:
    """问东答西：调够步数后，给一个与提问完全无关的结论。"""

    def __init__(self, tool_rounds: int = 9) -> None:
        self.calls = 0
        self._tool_rounds = tool_rounds

    async def chat(self, messages, tools=None) -> LLMResponse:
        self.calls += 1
        if self.calls > self._tool_rounds:
            return LLMResponse(content="全季度毛利率环比提升 3.2 个百分点，达成年度目标。")
        name = "list_sheets" if self.calls == 1 else "data_health_check"
        return LLMResponse(
            tool_calls=[ToolCall(id=f"t{self.calls}", name=name, arguments={})]
        )


async def test_loop_emits_drift_warning(make_agent, session, caplog) -> None:
    """验证告警真的会打出来，且不影响正常返回。"""
    caplog.set_level(logging.WARNING)
    # 步数要够触发阈值（8），但不能撞上 max_steps，否则拿不到 final 答案
    agent = make_agent(DriftingLLM(tool_rounds=8), max_agent_steps=10)
    result = await agent.run(session, "查一下张三这个用户的明细")

    assert result.stop_reason == "final"
    assert result.answer  # 答案照常返回，告警不阻断
    assert "疑似任务漂移" in caplog.text
    # error_code 是结构化字段，不在消息文本里，需从 record 上读
    record = next(r for r in caplog.records if "任务漂移告警" in r.getMessage())
    assert record.error_code == "TASK_DRIFT"


async def test_loop_does_not_warn_on_normal_answer(make_agent, session, caplog) -> None:
    caplog.set_level(logging.WARNING)
    result = await make_agent().run(session, "整体情况如何？")
    assert result.stop_reason == "final"
    assert "TASK_DRIFT" not in caplog.text


async def test_drift_check_runs_on_every_exit_path(make_agent, session, caplog) -> None:
    """no_file 分支也会走 _finish，不应因告警逻辑而崩溃。"""
    from app.dependencies import get_agent  # noqa: F401  确保 agent 可构造

    result = await make_agent().run(None, "查一下张三")
    assert result.stop_reason == "no_file"
    assert "上传" in result.answer
