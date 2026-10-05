"""DA-009 测试：find_value 工具 + 「不可回答不得改题」的提示词约束。

背景：用户问「查一下张三的明细」，表里没有用户维度，模型静默改答了门店销售排名。
本文件锁定两件事：工具能给出确定的「未找到」，提示词明确禁止改题。
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.agent.prompts import build_system_prompt
from app.tools.base import ToolContext, split_tool_error

SHEET = "S"


@pytest.fixture
def ctx() -> ToolContext:
    frame = pd.DataFrame(
        {
            "门店ID": ["ST002", "ST005", "ST002", "ST011"],
            "门店名称": ["西安旗舰店02", "上海标准店05", "西安旗舰店02", "北京社区店01"],
            "城市": ["西安", "上海", "西安", "北京"],
            "会员": ["是", "否", "是", "否"],
            "销售额": [100.0, 200.0, 300.0, 400.0],
        }
    )
    return ToolContext(session_id="t", sheets=[SHEET], frames={SHEET: frame})


# ============================================================ 找不到（核心场景）
def test_absent_value_is_definitively_reported(registry, ctx: ToolContext) -> None:
    """「查一下张三」必须得到确定的「未找到」，而不是模型的自由发挥。"""
    ok, out = registry.run("find_value", {"value": "张三"}, ctx)
    assert ok is True
    assert "未找到" in out
    assert "张三" in out  # 明确指出查的是谁
    # 必须给出建设性替代信息，否则模型只能改答
    assert "可用于筛选的类别列" in out
    assert "门店名称(3 个取值)" in out


def test_absent_value_lists_id_columns(registry, ctx: ToolContext) -> None:
    """没有值时也要告诉模型有哪些标识类列可查。"""
    ok, out = registry.run("find_value", {"value": "不存在"}, ctx)
    assert ok is True
    assert "标识类列" in out


# ============================================================ 找得到
def test_present_value_reports_columns_and_rows(registry, ctx: ToolContext) -> None:
    ok, out = registry.run("find_value", {"value": "西安旗舰店02"}, ctx)
    assert ok is True
    assert "找到" in out and "未找到" not in out
    assert "门店名称：2 行" in out
    assert "样例数据" in out


def test_exact_match_differs_from_contains(registry, ctx: ToolContext) -> None:
    """「社区」只是「北京社区店01」的子串：包含匹配命中，完全匹配不命中。"""
    _, partial = registry.run("find_value", {"value": "社区"}, ctx)
    assert "找到" in partial
    assert "门店名称" in partial
    _, exact = registry.run("find_value", {"value": "社区", "exact": True}, ctx)
    assert "未找到" in exact


def test_value_found_in_multiple_columns(registry, ctx: ToolContext) -> None:
    ok, out = registry.run("find_value", {"value": "西安"}, ctx)
    assert ok is True
    assert "出现在 2 个列中" in out
    assert "门店名称" in out and "城市" in out


# ============================================================ 作用域
def test_search_is_scoped_to_given_sheet(ctx: ToolContext) -> None:
    other = pd.DataFrame({"门店名称": ["张三"], "城市": ["西安"]})
    ctx.frames["其他表"] = other
    ok, out = _run(ctx, {"value": "张三", "sheet": SHEET})
    assert "未找到" in out
    ok, out = _run(ctx, {"value": "张三", "sheet": "其他表"})
    assert "找到" in out


def test_columns_parameter_restricts_search(registry, ctx: ToolContext) -> None:
    ok, out = registry.run("find_value", {"value": "西安", "columns": ["城市"]}, ctx)
    assert ok is True
    assert "出现在 1 个列中" in out
    assert "城市：" in out


def test_unknown_column_is_rejected(registry, ctx: ToolContext) -> None:
    ok, out = registry.run("find_value", {"value": "西安", "columns": ["不存在"]}, ctx)
    assert ok is False
    code, _ = split_tool_error(out)
    assert code == "COLUMN_NOT_FOUND"


def test_numeric_columns_are_not_searched(registry, ctx: ToolContext) -> None:
    """数值列不该参与文本搜索，否则会得到「命中 1 行」这种误导性结果。"""
    ok, out = registry.run("find_value", {"value": "100"}, ctx)
    assert ok is True
    # 门店ID/门店名称/城市/会员 参与搜索，销售额被排除
    assert "已搜索 4 个文本列" in out
    assert "销售额" not in out.split("样例数据")[0].split("类别列")[-1]


def test_reports_current_df_so_model_sees_which_table(registry, ctx: ToolContext) -> None:
    ok, out = registry.run("find_value", {"value": "西安"}, ctx)
    assert out.startswith(f"[当前 df = {SHEET}（")


def test_readonly_tool_does_not_switch_working_table(registry, ctx: ToolContext) -> None:
    """find_value 是只读体检工具，不能改当前工作表。"""
    registry.run("load_sheet", {"sheet": SHEET}, ctx)
    registry.run("find_value", {"value": "西安"}, ctx)
    assert ctx.active_sheet == SHEET


def test_table_without_text_columns_errors_cleanly(registry) -> None:
    numeric_only = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    ctx = ToolContext(session_id="t", sheets=["N"], frames={"N": numeric_only})
    ok, out = registry.run("find_value", {"value": "x"}, ctx)
    assert ok is False
    code, _ = split_tool_error(out)
    assert code == "NO_COLUMNS"


def _run(ctx: ToolContext, args: dict) -> tuple[bool, str]:
    from app.tools import registry

    return registry.run("find_value", args, ctx)


# ============================================================ 提示词约束
def test_prompt_forbids_silent_question_substitution(session) -> None:
    """核心不变量：提示词必须明确禁止静默改题。"""
    prompt = build_system_prompt(session, ["find_value"])
    assert "严禁把问题偷换成另一个问题" in prompt
    assert "静默改题" in prompt
    assert "未找到" in prompt or "没有这个维度" in prompt


def test_prompt_requires_existence_check_first(session) -> None:
    """提到具体值时必须先查存在性。"""
    prompt = build_system_prompt(session, ["find_value"])
    assert "find_value" in prompt
    assert "先调用它确认这个值是否存在" in prompt


def test_prompt_lists_find_value_in_workflow_before_run_pandas(session) -> None:
    """工作流顺序：存在性检查必须排在计算类工具之前。"""
    prompt = build_system_prompt(session, ["find_value", "run_pandas"])
    assert prompt.index("find_value") < prompt.index("run_pandas")


def test_find_value_is_registered(registry) -> None:
    assert "find_value" in registry.names()
