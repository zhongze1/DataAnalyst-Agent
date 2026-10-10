"""统一脱敏钩子（DA-015 / PRD W2-4）。

**双通路原则**（沿用 DA-007）：脱敏只作用于**对外**内容 —— 界面展示、导出报告、
落盘产物。喂给模型的工具输出必须是**原文**，否则模型会拿着 ``138****8888``
去 groupby，分析结果直接错。

**为什么钩子先建、规则后填**：导出是唯一会把原始数据值「整表带出去」的链路。
如果导出不走同一个脱敏函数，做完只遮界面而报告里仍有明文 —— 这类漏洞最难被发现。

打码覆盖三个出口：
1. 界面：``steps[].arguments`` / ``steps[].result`` / 结论 ``answer``（在 Agent Loop 的统一出口做）；
2. 产物 CSV 落盘与导出「数据明细」sheet（本模块 + ``artifacts`` / ``export_service``）；
3. 导出报告的正文与「执行轨迹」（``export_service``）。
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from app.config import get_settings

MASKED = "***"

# (正则, 替换模板)。前后加数字边界 ``(?<!\d)`` / ``(?!\d)`` 是必须的：
# 不加的话 18 位身份证的前 11 位会被当成手机号，打码结果面目全非。
PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # 手机号 -> 保留前 3 后 4：13800138000 -> 138****8000
    (re.compile(r"(?<!\d)(1[3-9]\d)\d{4}(\d{4})(?!\d)"), r"\1****\2"),
    # 身份证（18 位，末位可能是 X）-> 11010119900307123X -> 110101********123X
    (re.compile(r"(?<!\d)(\d{6})\d{8}(\d{3}[\dXx])(?!\d)"), r"\1********\2"),
    # 银行卡（16–19 位）-> 6222021234567890 -> 6222********7890
    (re.compile(r"(?<!\d)(\d{4})\d{8,11}(\d{4})(?!\d)"), r"\1********\2"),
    # 邮箱 -> 保留首字符与域名：zhang.san@corp.com -> z***@corp.com
    (re.compile(r"([\w.+-])[\w.+-]*@([\w-]+\.[\w.-]+)"), r"\1***@\2"),
]

# 内置列名关键词（子串匹配）。
# 列级命中比正则更可靠：值被读成 int64 时文本正则无从作用，只有列名能兜住。
DEFAULT_SENSITIVE_COLUMNS: frozenset[str] = frozenset(
    {
        "手机",
        "电话",
        "联系方式",
        "身份证",
        "证件",
        "银行卡",
        "卡号",
        "邮箱",
        "email",
        "地址",
        "住址",
        "薪资",
        "工资",
    }
)

# 运行期生效的关键词集合。配置是**追加**语义，不提供覆盖 ——
# 否则一次配错就把兜底规则全丢了，而这类问题不会有人发现。
SENSITIVE_COLUMNS: set[str] = set(DEFAULT_SENSITIVE_COLUMNS)

_COLUMN_SPLIT = re.compile(r"[,，;；\s]+")


def enabled() -> bool:
    """打码总开关。关闭时三个函数全部恒等（用于临时排障）。"""
    return bool(get_settings().redact_enabled)


def parse_columns(raw: str | None) -> set[str]:
    """解析 ``AGENT_REDACT_COLUMNS``：逗号 / 中文逗号 / 分号 / 空白分隔，忽略空项。"""
    return {item.strip() for item in _COLUMN_SPLIT.split(raw or "") if item.strip()}


def effective_columns() -> set[str]:
    """生效的列名关键词 = 内置默认集 ∪ 配置追加集。"""
    return SENSITIVE_COLUMNS | parse_columns(get_settings().redact_columns)


def redact_text(text: str) -> str:
    """按规则遮蔽自由文本中的敏感片段。"""
    if not text or not enabled():
        return text
    result = text
    for pattern, template in PATTERNS:
        result = pattern.sub(template, result)
    return result


def redact_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """按列名或内容规则遮蔽 DataFrame。

    - **列名命中**关键词：整列按值处理 —— 能匹配上正则的保留部分
      （``138****8888``，PRD 要求的可读格式），匹配不上的一律整体替换成 ``***``
      （列名已经声明它敏感，宁可比正则更保守）。
    - **列名未命中**：只对文本列做内容级正则替换，不动数值 / 日期列。
    """
    if frame is None or frame.empty:
        return frame
    masked = frame.copy()
    if not enabled():
        return masked
    markers = effective_columns()
    for column in masked.columns:
        if _is_sensitive_column(column, markers):
            masked[column] = masked[column].map(_mask_value)
        elif _is_text_column(masked[column]):
            masked[column] = masked[column].map(
                lambda v: redact_text(v) if isinstance(v, str) else v
            )
    return masked


def redact_value(value: Any) -> Any:
    """按类型分派。DataFrame / Series 走列级遮蔽，其余转文本走文本规则。"""
    if isinstance(value, pd.DataFrame):
        return redact_frame(value)
    if isinstance(value, pd.Series):
        return redact_value(value.to_frame())
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {k: redact_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(redact_value(item) for item in value)
    return value


def _is_sensitive_column(column: Any, markers: set[str]) -> bool:
    """列名子串匹配：``联系电话`` 命中 ``电话``。"""
    name = str(column)
    return any(marker in name for marker in markers)


def _is_text_column(series: pd.Series) -> bool:
    """文本列判据。

    **不能用 ``dtype == object``**：pandas 3 起字符串列默认是 ``str``
    （``StringDtype``），object 判据对它恒为 False —— 内容级规则会整条失效，
    而且失效得静默无声（这个坑踩过一次，由 test_redact 的内容级用例发现）。
    """
    return isinstance(series.dtype, pd.StringDtype) or series.dtype == object


def _mask_value(value: Any) -> Any:
    """单个单元格：能正则化的保留可读格式，否则整体遮蔽。"""
    if _is_missing(value):
        return value
    text = str(value)
    if not text.strip():
        return value
    masked = redact_text(text)
    return masked if masked != text else MASKED


def _is_missing(value: Any) -> bool:
    """缺失值原样保留 —— 不能把 NaN 写成 ``***``（那会凭空造出一个"值"）。"""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
