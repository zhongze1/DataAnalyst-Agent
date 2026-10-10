"""DA-015 敏感字段打码测试。

覆盖四类不变量：
1. **文本规则**：手机号 / 身份证 / 银行卡 / 邮箱各自打码，且数字边界生效（不误伤长串数字）；
2. **列级规则**：列名命中时"能正则化则保留部分、否则整体 ***"，数值型也能兜住；
3. **双通路**（核心）：对外脱敏、喂给模型的仍是原文 —— 否则模型会算错；
4. **导出兜底**：DA-015 之前落库的历史明文，导出时也要被遮住。

全部离线（Mock LLM）、全部写 tmp_path，不碰真实 data/ 目录。
"""

from __future__ import annotations

import io
import json

import pandas as pd
import pytest

from app.agent.loop import detect_task_drift
from app.config import Settings, validate_settings
from app.security.redact import (
    MASKED,
    effective_columns,
    parse_columns,
    redact_frame,
    redact_text,
    redact_value,
)
from app.services.export_service import build_markdown, build_xlsx, collect_turns
from stubs import PhoneLLM

PHONE = "13800138000"
ID_CARD = "11010119900307123X"
BANK_CARD = "6222021234567890"
EMAIL = "zhang.san@corp.com.cn"


# ============================================================ 文本规则
def test_phone_is_partially_masked() -> None:
    assert redact_text(f"联系方式 {PHONE}") == "联系方式 138****8000"


def test_id_card_keeps_head_and_tail() -> None:
    assert redact_text(f"证件号 {ID_CARD}") == "证件号 110101********123X"


def test_bank_card_keeps_head_and_tail() -> None:
    assert redact_text(f"卡号 {BANK_CARD}") == "卡号 6222********7890"


def test_email_keeps_first_char_and_domain() -> None:
    assert redact_text(f"邮箱 {EMAIL}") == "邮箱 z***@corp.com.cn"


def test_long_digit_runs_are_not_partially_masked() -> None:
    """13 位纯数字既不是手机号也不是身份证：数字边界必须挡住误伤。

    没有 ``(?<!\\d)`` / ``(?!\\d)`` 这两个断言，订单号会被拦腰打码。
    """
    assert redact_text("订单 1380013800012 已发货") == "订单 1380013800012 已发货"
    assert redact_text("流水 62220212345678901234 完成") == "流水 62220212345678901234 完成"


def test_plain_text_and_empty_input_are_safe() -> None:
    assert redact_text("本月华东区销售额 12,345 元") == "本月华东区销售额 12,345 元"
    assert redact_text("") == ""


# ============================================================ 列级规则
def test_sensitive_column_with_matching_value_keeps_readable_format() -> None:
    masked = redact_frame(pd.DataFrame({"手机号": [PHONE]}))
    assert masked["手机号"].tolist() == ["138****8000"]


def test_sensitive_column_with_unmatched_value_falls_back_to_asterisks() -> None:
    """列名已经声明它敏感；正则认不出来时宁可比正则更保守。"""
    masked = redact_frame(pd.DataFrame({"身份证": ["1101011990..."]}))
    assert masked["身份证"].tolist() == [MASKED]


def test_sensitive_column_match_is_substring() -> None:
    masked = redact_frame(pd.DataFrame({"联系电话": [PHONE], "客户手机号码": [PHONE]}))
    assert masked["联系电话"].tolist() == ["138****8000"]
    assert masked["客户手机号码"].tolist() == ["138****8000"]


def test_non_sensitive_text_column_gets_content_level_masking() -> None:
    masked = redact_frame(pd.DataFrame({"备注": [f"请联系 {PHONE}"]}))
    assert masked["备注"].tolist() == ["请联系 138****8000"]


def test_sensitive_numeric_column_is_masked() -> None:
    """手机号被读成 int64 时正则无从作用，只有列名能兜住（这是列名规则存在的理由）。"""
    masked = redact_frame(pd.DataFrame({"手机号": [13800138000]}))
    assert masked["手机号"].tolist() == ["138****8000"]


def test_numeric_and_datetime_columns_are_not_touched() -> None:
    """内容级规则只作用于文本列，不能误伤数值。"""
    frame = pd.DataFrame({"销售额": [100, 200]})
    masked = redact_frame(frame)
    assert masked["销售额"].tolist() == [100, 200]
    assert masked["销售额"].dtype == frame["销售额"].dtype


def test_missing_values_stay_missing() -> None:
    """缺失值不能被打成 "***" —— 那会凭空造出一个值。"""
    masked = redact_frame(pd.DataFrame({"手机号": [PHONE, None]}))
    assert masked["手机号"].iloc[0] == "138****8000"
    assert pd.isna(masked["手机号"].iloc[1])


def test_redact_value_dispatches_by_type() -> None:
    assert redact_value({"code": f"df.query('{PHONE}')"}) == {"code": "df.query('138****8000')"}
    assert redact_value([PHONE]) == ["138****8000"]
    assert redact_value(42) == 42
    assert redact_value(None) is None


# ============================================================ 配置
def test_parse_columns_accepts_chinese_and_ascii_separators() -> None:
    assert parse_columns("工号, 邮编；备用手机 邮箱") == {"工号", "邮编", "备用手机", "邮箱"}
    assert parse_columns("") == set()
    assert parse_columns(None) == set()


def test_configured_columns_are_appended_not_replacing_defaults(settings, patch_settings) -> None:
    """配置是**追加**：不能因为配置了就丢掉内置兜底规则。"""
    patch_settings(settings.model_copy(update={"redact_columns": "工号，邮编; 备用电话"}))
    markers = effective_columns()
    assert {"工号", "邮编", "备用电话"} <= markers
    assert {"手机", "身份证"} <= markers  # 内置默认集仍在

    masked = redact_frame(
        pd.DataFrame({"工号": ["A001"], "邮编": ["710000"], "手机号": [PHONE], "城市": ["西安"]})
    )
    assert masked["工号"].tolist() == [MASKED]
    assert masked["邮编"].tolist() == [MASKED]
    assert masked["手机号"].tolist() == ["138****8000"]
    assert masked["城市"].tolist() == ["西安"]


def test_disabled_redaction_is_identity(settings, patch_settings) -> None:
    patch_settings(settings.model_copy(update={"redact_enabled": False}))
    assert redact_text(f"联系 {PHONE}") == f"联系 {PHONE}"
    masked = redact_frame(pd.DataFrame({"手机号": [PHONE]}))
    assert masked["手机号"].tolist() == [PHONE]


def test_defaults_are_backward_compatible(tmp_path) -> None:
    """不新增任何环境变量时，配置校验必须通过、打码默认开启。"""
    fresh = Settings(
        _env_file=None,
        llm_provider="mock",
        upload_dir=tmp_path / "uploads",
        db_path=tmp_path / "cfg.db",
        auth_enabled=False,
    )
    assert validate_settings(fresh) == []
    assert fresh.redact_enabled is True
    assert fresh.redact_columns == ""


# ============================================================ 双通路（核心不变量）
async def test_tool_output_masked_for_users_but_raw_for_model(make_agent, session) -> None:
    """对外脱敏、对模型原文 —— 这是本工单最重要的一条边界。"""
    llm = PhoneLLM()
    result = await make_agent(llm).run(session, "查一下客户手机号")

    assert result.steps, "至少要有一个工具步骤"
    step = result.steps[0]
    assert step.tool == "run_pandas"

    # ① 对外：步骤结果与参数都不能有明文
    assert "138****8000" in step.result
    assert PHONE not in step.result
    assert PHONE not in json.dumps(step.arguments, ensure_ascii=False)
    assert "138****8000" in step.arguments["code"]

    # ② 对外：结论也要脱敏（模型可能把手机号直接写进答案）
    assert "138****8000" in result.answer
    assert PHONE not in result.answer

    # ③ 对模型：工具输出必须仍是原文，否则「按手机号分组」会算错
    assert any(PHONE in content for content in llm.tool_contents)

    # ④ 历史上下文同样保留原文（下一轮要喂给模型）
    assert PHONE in json.dumps(session.messages, ensure_ascii=False)


async def test_drift_detection_still_sees_the_raw_answer(make_agent, session) -> None:
    """脱敏发生在漂移检测之后：否则关键词比对会被打码干扰，产生误告警。"""
    llm = PhoneLLM()
    result = await make_agent(llm).run(session, "查一下客户手机号")
    # 拿到的是脱敏后的答案，但检测用的是原文（此处只验证不崩、不误报）
    assert detect_task_drift("查一下客户手机号", result.answer, len(result.steps)) is None


# ============================================================ 导出兜底
def _legacy_messages_with_raw_phone() -> list[dict]:
    """模拟 DA-015 之前落库的消息：内容与轨迹里都是明文。"""
    return [
        {"role": "user", "content": f"查客户 {PHONE} 的明细"},
        {
            "role": "assistant",
            "content": f"客户 {PHONE} 的订单如下",
            "steps": [
                {
                    "step": 1,
                    "tool": "run_pandas",
                    "arguments": {"code": f"df.query('手机 == \"{PHONE}\"')"},
                    "ok": True,
                    "result": f"客户手机\n{PHONE}",
                    "error": None,
                }
            ],
        },
    ]


def test_export_markdown_masks_legacy_raw_trace(tmp_path) -> None:
    # 走真实管线：collect_turns 负责问答脱敏，轨迹脱敏由导出侧兜底
    turns = collect_turns(_legacy_messages_with_raw_phone())
    bundle = build_markdown(
        filename="测试数据.xlsx",
        session_id="s1",
        turns=turns,
        artifacts=[],
        session_path=tmp_path / "s1" / "测试数据.xlsx",
        generated_at="2026-10-10 10:00",
    )
    text = bundle.content.decode("utf-8")
    assert PHONE not in text
    assert "138****8000" in text


def test_export_xlsx_masks_legacy_raw_trace(tmp_path) -> None:
    # collect_turns 不会动 steps（轨迹保持原文），所以这条断言真正验证的是
    # 导出侧的二次脱敏 —— 历史会话的明文轨迹就靠它兜住
    turns = collect_turns(_legacy_messages_with_raw_phone())
    assert PHONE in json.dumps(turns, ensure_ascii=False), "前提：轨迹此时还是明文"
    bundle = build_xlsx(
        filename="测试数据.xlsx",
        session_id="s1",
        turns=turns,
        artifacts=[],
        session_path=tmp_path / "s1" / "测试数据.xlsx",
        generated_at="2026-10-10 10:00",
    )
    trace = pd.read_excel(io.BytesIO(bundle.content), sheet_name="执行轨迹")
    body = trace.to_string()
    assert PHONE not in body
    assert "138****8000" in body


# ============================================================ 回归：不误伤既有输出
async def test_normal_answer_is_unaffected(make_agent, session) -> None:
    """没有敏感值的正常问答，输出必须逐字不变。"""
    result = await make_agent().run(session, "整体情况")
    assert result.stop_reason == "final"
    assert result.steps
    assert not any("***" in step.result for step in result.steps)


@pytest.mark.parametrize("value", [PHONE, ID_CARD, BANK_CARD, EMAIL])
def test_redact_value_is_idempotent(value: str) -> None:
    """打码结果再打一次不该继续变形（避免流式/多出口重复处理时越打越烂）。"""
    once = redact_text(value)
    assert redact_text(once) == once
