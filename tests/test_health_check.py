"""`data_health_check` 关键分支测试。

只覆盖**会改变结论的分支**：各类告警的触发与静默、参数边界、错误路径。
不追求 pandas_tools.py 的边缘分支覆盖率 —— 那些是防御性代码，测了不产生价值。

所有用例直接把 DataFrame 塞进 ToolContext.frames，不碰磁盘、不需要 Excel。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.tools.base import ToolContext
from app.tools.registry import ToolRegistry

SHEET = "测试表"


def _ctx(frames: dict[str, pd.DataFrame]) -> ToolContext:
    return ToolContext(
        session_id="test",
        file_path=None,
        file_name=None,
        sheets=list(frames),
        frames=dict(frames),
    )


def check(frames: dict[str, pd.DataFrame], registry: ToolRegistry, **args) -> str:
    """执行体检，断言成功并返回报告文本。"""
    ok, text = registry.run("data_health_check", args, _ctx(frames))
    assert ok, f"体检不应失败: {text}"
    return text


def check_fails(frames: dict[str, pd.DataFrame], registry: ToolRegistry, **args) -> str:
    """执行体检，断言失败并返回错误文本。"""
    ok, text = registry.run("data_health_check", args, _ctx(frames))
    assert not ok, f"体检应当失败: {text}"
    return text


def frame(**columns) -> dict[str, pd.DataFrame]:
    return {SHEET: pd.DataFrame(columns)}


# ============================================================ 干净基线
def test_clean_table_reports_no_alert(registry: ToolRegistry) -> None:
    """四种列类型都覆盖到，且任何告警都不该触发。"""
    report = check(
        frame(
            金额=[100.0, 200.0, 300.0, 400.0],
            类别=["a", "b", "a", "b"],
            日期=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]),
            标记=[True, False, True, False],
        ),
        registry,
    )
    assert "4 行 × 4 列" in report
    assert "布尔列: True:2/False:2" in report
    assert "范围=2025-01-01 ~ 2025-01-04" in report
    assert "跨度=3天" in report
    assert "日均 1.0 条" in report
    assert "未发现明显问题" in report


def test_explicit_sheet_is_used(registry: ToolRegistry) -> None:
    frames = {"A表": pd.DataFrame({"x": [1, 2]}), "B表": pd.DataFrame({"y": [3, 4, 5]})}
    assert "B表" in check(frames, registry, sheet="B表")
    assert "A表" in check(frames, registry)  # 省略时取第一张表


# ============================================================ 缺失值
def test_high_missing_rate_raises_alert(registry: ToolRegistry) -> None:
    report = check(frame(备注=["x"] * 5 + [None] * 20), registry)
    assert "高缺失: 备注（20 行，占 80.0%）" in report


def test_missing_below_threshold_stays_silent(registry: ToolRegistry) -> None:
    """缺失率 10% 未达 20% 阈值 —— 不该报高缺失。"""
    report = check(frame(行号=range(100), 值=[float(i) for i in range(90)] + [None] * 10), registry)
    assert "缺失=10 (10.0%)" in report
    assert "高缺失" not in report


def test_entirely_empty_column(registry: ToolRegistry) -> None:
    report = check(frame(空列=[None] * 10), registry)
    assert "备注=整列为空" in report


# ============================================================ 离群点
def test_outlier_over_threshold_raises_alert(registry: ToolRegistry) -> None:
    values = [float(i) for i in range(1, 46)] + [10000.0] * 5
    report = check(frame(值=values), registry)
    assert "离群点=5 (5.0%)" in report or "离群点=5 (10.0%)" in report
    assert "离群点偏多: 值" in report


def test_outlier_under_threshold_reported_without_alert(registry: ToolRegistry) -> None:
    """有离群点但占比不高：写进明细，但不进告警清单。"""
    values = [float(i) for i in range(1, 50)] + [10000.0]
    report = check(frame(值=values), registry)
    assert "离群点=1 (2.0%)" in report
    assert "离群点偏多" not in report


def test_zero_iqr_skips_outlier_detection(registry: ToolRegistry) -> None:
    """Q1 == Q3 时 IQR 为 0，无法判定离群点，应静默跳过而不是报错。"""
    report = check(frame(值=[0.0] * 5 + [5.0] * 10 + [10.0] * 5), registry)
    assert "离群点" not in report
    assert "偏度" not in report


def test_fewer_than_four_valid_values_skips_stats(registry: ToolRegistry) -> None:
    """有效值不足 4 个时不做 IQR/偏度，但描述统计仍要给出。"""
    report = check(frame(少量=[1.0, 2.0, None]), registry)
    assert "均值=1.50" in report
    assert "离群点" not in report
    assert "偏度" not in report


# ============================================================ 偏态
def test_right_skew_raises_alert(registry: ToolRegistry) -> None:
    values = np.random.default_rng(0).lognormal(0, 1.5, 200)
    report = check(frame(值=values), registry)
    assert "严重右偏: 值" in report
    assert "建议看中位数" in report


def test_left_skew_raises_alert(registry: ToolRegistry) -> None:
    values = -np.random.default_rng(1).lognormal(0, 1.5, 200)
    report = check(frame(值=values), registry)
    assert "严重左偏: 值" in report


def test_symmetric_data_has_no_skew_alert(registry: ToolRegistry) -> None:
    values = np.random.default_rng(2).uniform(0, 1, 200)
    report = check(frame(值=values), registry)
    assert "偏度=" not in report
    assert "严重" not in report


# ============================================================ 列形态
def test_constant_column_flagged(registry: ToolRegistry) -> None:
    report = check(frame(常量=[7.0] * 10), registry)
    assert "备注=常量列（无区分度）" in report


def test_high_cardinality_column_flagged_as_id(registry: ToolRegistry) -> None:
    report = check(frame(ID=[f"x{i}" for i in range(30)]), registry)
    assert "高基数（疑似主键/ID）: ID（唯一值 30 = 有效行数）" in report


def test_id_alert_requires_more_than_twenty_rows(registry: ToolRegistry) -> None:
    """20 行及以下不做 ID 判定，否则小表会被误报。"""
    report = check(frame(ID=[f"x{i}" for i in range(20)]), registry)
    assert "高基数" not in report


def test_single_day_datetime_has_no_daily_average(registry: ToolRegistry) -> None:
    """同一天内的多个时刻：跨度为 0，不应输出「日均」。"""
    report = check(frame(时刻=pd.to_datetime(["2025-01-01 08:00", "2025-01-01 20:00"])), registry)
    assert "跨度=0天" in report
    assert "日均" not in report


# ============================================================ 重复行
def test_duplicate_rows_detected(registry: ToolRegistry) -> None:
    """[1,2,3,1,2,3] 中 3 行各自与前面的行重复。"""
    report = check(frame(a=[1, 2, 3, 1, 2, 3]), registry)
    assert "完全重复行 3 行（占 50.0%）" in report


def test_column_subset_skips_duplicate_check(registry: ToolRegistry) -> None:
    """回归测试：按列子集判断「重复行」没有业务意义，曾误报 98% 重复。"""
    frames = frame(a=[1, 2, 3, 1, 2, 3], b=["x", "y", "z", "x", "y", "z"])
    assert "完全重复行 3 行" in check(frames, registry)
    subset_report = check(frames, registry, columns=["a"])
    assert "完全重复行" not in subset_report


# ============================================================ 参数边界
def test_wide_table_is_truncated_with_notice(registry: ToolRegistry) -> None:
    frames = frame(**{f"c{i}": [1.0, 2.0] for i in range(10)})
    report = check(frames, registry, max_columns=3)
    assert "10 列" in report
    assert "仅体检前 3 列" in report
    assert report.count("\n- c") == 3


def test_max_columns_zero_falls_back_to_default(registry: ToolRegistry) -> None:
    """max_columns=0 视为未传（回退 40），不该把表截成 0 列。"""
    frames = frame(**{f"c{i}": [1.0, 2.0] for i in range(10)})
    report = check(frames, registry, max_columns=0)
    assert "仅体检前" not in report
    assert report.count("\n- c") == 10


def test_top_controls_category_value_count(registry: ToolRegistry) -> None:
    frames = frame(区域=["华东", "华北", "华南", "西部"])
    single = check(frames, registry, top=1)
    assert "Top: 华东(1)" in single
    assert "华北(1)" not in single
    # 四个值计数相同，value_counts 按出现顺序取前 N
    assert "华南(1)" in check(frames, registry, top=3)
    assert "西部(1)" not in check(frames, registry, top=3)
    assert "西部(1)" in check(frames, registry, top=4)


def test_unknown_column_is_rejected_with_available_list(registry: ToolRegistry) -> None:
    error = check_fails(frame(金额=[1.0, 2.0]), registry, columns=["不存在"])
    assert "不存在" in error
    assert "可用列" in error


def test_empty_column_list_is_rejected(registry: ToolRegistry) -> None:
    """回归测试：columns=[] 曾因真值判断被当成「未传」，静默退化成全表体检。"""
    error = check_fails(frame(金额=[1.0, 2.0]), registry, columns=[])
    assert "没有可体检的列" in error


def test_omitted_columns_still_means_full_table(registry: ToolRegistry) -> None:
    """对照组：不传 columns 才是全表体检。"""
    assert "1 列" in check(frame(金额=[1.0, 2.0]), registry)


# ============================================================ 数值格式
def test_numbers_are_formatted_by_magnitude(registry: ToolRegistry) -> None:
    """千分位 / 小数位自适应，保证报告紧凑可读。"""
    report = check(frame(值=[9036.5, 3108.0, 0.575, 0.0]), registry)
    assert "最大=9,036.5" in report
    assert "最小=0" in report
    assert "中位数=1,554.3" in report  # (0.575 + 3108.0) / 2 = 1554.2875
    assert "均值=3,036.3" in report


def test_row_count_uses_thousands_separator(registry: ToolRegistry) -> None:
    report = check(frame(值=[float(i) for i in range(1200)]), registry)
    assert "1,200 行 × 1 列" in report


def test_empty_dataframe_is_handled(registry: ToolRegistry) -> None:
    """0 行数据不应崩溃，缺失率等比率计算要除零保护。"""
    report = check(frame(值=pd.Series([], dtype="float64")), registry)
    assert "0 行 × 1 列" in report
