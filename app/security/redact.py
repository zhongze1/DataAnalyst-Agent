"""统一脱敏钩子（W2-2 预留，W2-4 填充规则）。

**为什么现在就建**：导出是唯一会把原始数据值「整表带出去」的链路。如果导出
不走同一个脱敏函数，W2-4 做完只遮界面而报告里仍有明文 —— 这类漏洞最难被发现。

三个函数当前都是恒等实现，但**所有导出路径必须经过它们**，规则只在
:data:`PATTERNS` 一处添加。
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

# (正则, 替换模板)。W2-4 在这里加手机号 / 身份证 / 银行卡等规则。
PATTERNS: list[tuple[re.Pattern[str], str]] = []

# 列名关键词（子串匹配）。W2-4 补充，例如 {"手机", "身份证", "银行卡"}
SENSITIVE_COLUMNS: set[str] = set()


def redact_text(text: str) -> str:
    """按规则遮蔽自由文本中的敏感片段。"""
    if not text:
        return text
    result = text
    for pattern, template in PATTERNS:
        result = pattern.sub(template, result)
    return result


def redact_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """按列名或内容规则遮蔽 DataFrame。

    列名命中 ``SENSITIVE_COLUMNS`` 则整列打码；未命中则对每个单元格做文本规则替换。
    """
    if frame is None or frame.empty:
        return frame
    masked = frame.copy()
    for column in masked.columns:
        if any(marker in str(column) for marker in SENSITIVE_COLUMNS):
            masked[column] = "***"
            continue
        if masked[column].dtype == object:
            masked[column] = masked[column].map(
                lambda v: redact_text(v) if isinstance(v, str) else v
            )
    return masked


def redact_value(value: Any) -> Any:
    """按类型分派。DataFrame/Series 走列级遮蔽，其余转文本走文本规则。"""
    if isinstance(value, pd.DataFrame):
        return redact_frame(value)
    if isinstance(value, pd.Series):
        return redact_value(value.to_frame())
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {k: redact_value(v) for k, v in value.items()}
    return value