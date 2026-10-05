"""工作表状态与沙箱健壮性测试。

背景：DA-004 批次发现模型在多表场景下反复 KeyError。根因是
``data_health_check`` 等只读工具会改写 ``active_sheet``，导致模型体检完
其他表后，``run_pandas`` 里的 ``df`` 已经悄悄换成最后一张表。

本文件锁定三条不变量：
1. 只有显式意图（load_sheet / run_pandas(sheet=)）能切换工作表；
2. 任何读取 df 的工具，输出首行都要说明当前 df 是哪张表；
3. 沙箱报错必须带上下文，且预置模块的 import 不该浪费轮次。
"""

from __future__ import annotations

import pytest

from app.tools.base import ToolContext

SHEET_MAIN = "明细"
SHEET_OTHER = "摘要"


@pytest.fixture
def ctx(session, store) -> ToolContext:
    return ToolContext(
        session_id=session.session_id,
        file_path=str(session.path),
        file_name=session.filename,
        sheets=list(session.sheets),
        frames=session.frames,
    )


def _load_main(registry, ctx: ToolContext) -> None:
    ok, out = registry.run("load_sheet", {"sheet": SHEET_MAIN}, ctx)
    assert ok, out


# ================================================== 不变量 1：显式切换
def test_load_sheet_switches_working_table(registry, ctx: ToolContext) -> None:
    _load_main(registry, ctx)
    assert ctx.active_sheet == SHEET_MAIN


@pytest.mark.parametrize(
    "name,args",
    [
        ("data_health_check", {}),
        ("preview_rows", {}),
        ("column_profile", {}),
        ("describe_dataframe", {}),
        ("correlation", {}),
    ],
)
def test_readonly_tools_never_switch_working_table(
    registry, ctx: ToolContext, name: str, args: dict
) -> None:
    """核心回归：只读工具不得有副作用。

    这正是线上 KeyError 的根因 —— 体检完三张表后 df 变了，模型却不知道。
    """
    _load_main(registry, ctx)
    ok, out = registry.run(name, {**args, "sheet": SHEET_OTHER}, ctx)
    assert ok, out
    assert ctx.active_sheet == SHEET_MAIN, f"{name} 不应切换当前工作表"


def test_run_pandas_after_inspecting_other_sheet_uses_original(registry, ctx: ToolContext) -> None:
    """完整复现线上场景：体检别表后写代码，df 仍应是原来那张。"""
    _load_main(registry, ctx)
    registry.run("data_health_check", {"sheet": SHEET_OTHER}, ctx)
    registry.run("preview_rows", {"sheet": SHEET_OTHER}, ctx)

    ok, out = registry.run("run_pandas", {"code": "out = list(df.columns)"}, ctx)
    assert ok, out
    assert "当前 df = 明细" in out
    assert "订单号" in out


def test_run_pandas_with_explicit_sheet_switches(registry, ctx: ToolContext) -> None:
    _load_main(registry, ctx)
    ok, out = registry.run(
        "run_pandas", {"sheet": SHEET_OTHER, "code": "out = df.shape[0]"}, ctx
    )
    assert ok, out
    assert ctx.active_sheet == SHEET_OTHER
    assert "当前 df = 摘要" in out


# ================================================== 不变量 2：状态可见
@pytest.mark.parametrize(
    "name,args",
    [
        ("data_health_check", {}),
        ("preview_rows", {}),
        ("column_profile", {}),
        ("value_counts", {"column": "备注"}),
        ("describe_dataframe", {}),
        ("correlation", {}),
        ("run_pandas", {"code": "df.shape"}),
    ],
)
def test_frame_reading_tools_report_active_df(
    registry, ctx: ToolContext, name: str, args: dict
) -> None:
    _load_main(registry, ctx)
    ok, out = registry.run(name, args, ctx)
    assert ok, out
    assert out.startswith(f"[当前 df = {SHEET_MAIN}"), f"{name} 缺少当前 df 标记"


@pytest.mark.parametrize("name,args", [("list_sheets", {}), ("load_sheet", {"sheet": SHEET_OTHER})])
def test_non_frame_tools_have_no_df_marker(registry, ctx: ToolContext, name: str, args: dict) -> None:
    ok, out = registry.run(name, args, ctx)
    assert ok, out
    assert "当前 df" not in out


def test_df_marker_shape_includes_dimensions(registry, ctx: ToolContext) -> None:
    _load_main(registry, ctx)
    out = registry.run("run_pandas", {"code": "df.shape"}, ctx)[1]
    assert out.splitlines()[0] == "[当前 df = 明细（22 行 × 6 列）]"


def test_df_marker_says_unloaded_when_nothing_loaded(registry) -> None:
    empty = ToolContext(session_id="t", sheets=["x"])
    assert empty.describe_active_frame() == "[当前 df = 未加载]"


# ================================================== 沙箱报错带上下文
def test_key_error_carries_sheet_and_columns(registry, ctx: ToolContext) -> None:
    """KeyError 必须告诉模型当前是哪张表、有哪些列，否则它只能再猜一次。"""
    _load_main(registry, ctx)
    ok, out = registry.run("run_pandas", {"code": "df['不存在的列'].sum()"}, ctx)
    assert ok is False
    assert "KeyError" in out
    assert "明细" in out
    assert "订单号" in out


def test_attribute_error_suggests_bracket_access(registry, ctx: ToolContext) -> None:
    """模型按英文习惯猜列名（df.sales）时，报错要给出真实列名与正确写法。"""
    _load_main(registry, ctx)
    ok, out = registry.run("run_pandas", {"code": "df.sales.sum()"}, ctx)
    assert ok is False
    assert "AttributeError" in out
    assert "df['列名']" in out
    assert "销售额" in out  # 同时给出该表的真实列名


def test_wrong_sheet_columns_in_error_are_visible(registry, ctx: ToolContext) -> None:
    """切到别的表后写错列名，报错要指向那张表的真实列。"""
    ok, out = registry.run(
        "run_pandas", {"sheet": SHEET_OTHER, "code": "df['库存量'].sum()"}, ctx
    )
    assert ok is False
    assert "摘要" in out  # 报错指向当前表，而不是先前那张


# ================================================== import 处理
@pytest.mark.parametrize(
    "code",
    [
        "import pandas as pd\nout = df.shape",
        "import pandas\nout = df.shape",
        "import numpy as np\nimport pandas as pd\nout = int(np.prod([2, 3]))",
        "from __future__ import annotations\nout = 1",  # 合法但无意义，应被拒
    ],
)
def test_preloaded_imports_are_stripped_or_handled(registry, ctx: ToolContext, code: str) -> None:
    """模型有写 import 的肌肉记忆，pandas/numpy 已在沙箱里，不该为此浪费一轮。"""
    _load_main(registry, ctx)
    ok, out = registry.run("run_pandas", {"code": code}, ctx)
    if "from __future__" in code:
        assert ok is False
    else:
        assert ok, out


def test_stripped_import_alias_is_rebound(registry, ctx: ToolContext) -> None:
    """`import pandas as pdd` 剥掉 import 后，别名 pdd 仍要能用。"""
    _load_main(registry, ctx)
    ok, out = registry.run(
        "run_pandas",
        {"code": "import pandas as pdd\nout = int(pdd.DataFrame({'a': [1, 2]}).shape[0])"},
        ctx,
    )
    assert ok, out
    assert "2" in out


@pytest.mark.parametrize(
    "code",
    ["import os", "import sys", "import subprocess", "from os import path", "from pandas import DataFrame"],
)
def test_other_imports_remain_blocked(registry, ctx: ToolContext, code: str) -> None:
    _load_main(registry, ctx)
    ok, out = registry.run("run_pandas", {"code": code}, ctx)
    assert ok is False
    assert "import" in out


def test_columns_variable_is_injected(registry, ctx: ToolContext) -> None:
    _load_main(registry, ctx)
    ok, out = registry.run("run_pandas", {"code": "out = len(columns)"}, ctx)
    assert ok, out
    assert "6" in out
