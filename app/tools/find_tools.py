"""DA-009：补上「这个值在不在数据里」的确定性查询能力。

起因：用户问「查一下张三的明细」，而表里根本没有用户维度。模型既无法表达
「找不到」这个检查，也没有任何工具能告诉它「没找到」——于是它改答了另一个
问题（门店销售排名）并静默交付，一个看起来像答案的非答案。

本工具把「存在性检查」变成一次确定的工具调用，模型没有理由再去猜列名。
"""

from __future__ import annotations

import pandas as pd

from app.tools.base import ToolContext, ToolError
from app.tools.registry import tool


@tool(
    name="find_value",
    description=(
        "在表里查找某个值/关键词出现在哪些列、共多少行。\n"
        "**提问里提到具体的人名、商品名、门店名、订单号、日期等具体值时，"
        "必须先调用本工具确认它是否存在**，再决定怎么回答。\n"
        "若返回「未找到」，说明该值在数据中不存在，此时应明确告诉用户"
        "「查不到 / 没有这个维度」，并给出数据中能回答的最接近内容，"
        "**不要改答另一个问题**。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "value": {"type": "string", "description": "要查找的值或关键词"},
            "sheet": {"type": "string", "description": "工作表名，省略则用当前表"},
            "columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "可选，只在这些列里找；省略则搜索全部文本列",
            },
            "exact": {
                "type": "boolean",
                "description": "true=完全相等，false=包含（默认）。中文人名建议用 false",
            },
        },
        "required": ["value"],
    },
    uses_active_frame=True,
)
def find_value(
    ctx: ToolContext,
    value: str,
    sheet: str | None = None,
    columns: list[str] | None = None,
    exact: bool = False,
) -> str:
    df = ctx.get_frame(sheet)

    if columns is not None:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ToolError(
                f"列 {missing} 不存在。可用列: {list(df.columns)}", code="COLUMN_NOT_FOUND"
            )
        candidates = list(columns)
    else:
        candidates = [
            col
            for col in df.columns
            if not pd.api.types.is_numeric_dtype(df[col])
            and not pd.api.types.is_datetime64_any_dtype(df[col])
        ]

    if not candidates:
        raise ToolError(
            f"表里没有可搜索的文本列。列类型: {dict(df.dtypes.astype(str))}",
            code="NO_COLUMNS",
        )

    hits: list[tuple[str, int]] = []
    for col in candidates:
        text = df[col].dropna().astype(str)
        if text.empty:
            continue
        if exact:
            mask = text.eq(value)
        else:
            mask = text.str.contains(value, case=False, regex=False, na=False)
        count = int(mask.sum())
        if count:
            hits.append((col, count))

    match_mode = "完全匹配" if exact else "包含匹配"
    header = f"[查找结果 · {ctx.active_sheet}] 关键字「{value}」（{match_mode}）"
    searched = f"已搜索 {len(candidates)} 个文本列，共 {len(df):,} 行"

    if not hits:
        # 把「能搜什么」一并给出，便于模型给出有建设性的替代建议
        options = []
        for col in candidates:
            n_unique = int(df[col].nunique(dropna=True))
            if 0 < n_unique <= 30:
                options.append(f"{col}({n_unique} 个取值)")
        lines = [
            header,
            "**未找到**：该值在所有文本列中都不存在。",
            searched,
            "",
            "表里可用于筛选的类别列及其取值数量："
            + ("、".join(options) if options else "无"),
        ]
        id_cols = [c for c in df.columns if "ID" in c or "号" in c]
        if id_cols:
            lines.append("标识类列: " + "、".join(id_cols))
        return "\n".join(lines)

    lines = [header, f"**找到**：出现在 {len(hits)} 个列中", searched, ""]
    for col, count in sorted(hits, key=lambda x: -x[1]):
        lines.append(f"  - {col}：{count:,} 行")
    first_col = hits[0][0]
    mask = (
        df[first_col].astype(str).eq(value)
        if exact
        else df[first_col].astype(str).str.contains(value, case=False, regex=False, na=False)
    )
    sample = df[mask].head(3)
    lines.append("")
    lines.append(f"样例数据（前 3 行，取自 {first_col}）:\n{sample.to_string()}")
    return "\n".join(lines)
