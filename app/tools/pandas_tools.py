"""pandas 分析工具集 —— 模型的「手和眼」。

工具清单（按推荐调用顺序）：
    list_sheets          列出工作表
    load_sheet           把某个工作表读成 DataFrame
    preview_rows         看前几行原始数据
    column_profile       逐列画像（类型/缺失/唯一值/极值/频次）
    value_counts         单列取值分布
    describe_dataframe   数值列描述统计
    correlation          相关性矩阵
    run_pandas           沙箱内执行任意 pandas 代码（兜底万能工具）
"""

from __future__ import annotations

import ast
import builtins
import io
from contextlib import redirect_stdout
from typing import Any

import numpy as np
import pandas as pd

from app.config import get_settings
from app.tools.base import ToolContext, ToolError
from app.tools.registry import tool

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)

# 判定「值得告警」的阈值
MISSING_WARN = 20.0        # 缺失率超过 20%
OUTLIER_WARN = 5.0         # 离群点占比超过 5%
SKEW_WARN = 2.0            # 绝对偏度超过 2 视为严重偏态


# ----------------------------------------------------------------- 显示辅助
def _fmt_count(count: int, total: int) -> str:
    """把数量和占比拼在一起，如 ``12 (1.0%)``。"""
    if not total:
        return "n/a"
    return f"{count:,} ({count / total * 100:.1f}%)"


def _fmt_num(value: Any) -> str:
    """紧凑数值显示：按量级自适应小数位。"""
    if value is None:
        return "NaN"
    try:
        if pd.isna(value):
            return "NaN"
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    number = float(value)
    if number == 0:
        return "0"
    if abs(number) >= 1000:
        return f"{number:,.1f}"
    if abs(number) >= 1:
        return f"{number:.2f}"
    return f"{number:.4g}"


def _fmt_date(value: Any) -> str:
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return str(value)



# ----------------------------------------------------------------- 元信息
@tool(
    name="list_sheets",
    description="列出已上传 Excel 文件中的所有工作表名称。分析前先调用它。",
    parameters={"type": "object", "properties": {}},
)
def list_sheets(ctx: ToolContext) -> str:
    if not ctx.sheets:
        raise ToolError("当前会话没有已上传的文件。")
    rows = []
    for name in ctx.sheets:
        shape = "未加载"
        if name in ctx.frames:
            shape = f"已加载 shape={ctx.frames[name].shape}"
        rows.append(f"  - {name}（{shape}）")
    return f"[Sheets] {', '.join(ctx.sheets)}\n{chr(10).join(rows)}"


@tool(
    name="load_sheet",
    description="把指定工作表读入内存作为 DataFrame，返回行列数、列名与数据类型。这是所有后续分析的前提。",
    parameters={
        "type": "object",
        "properties": {
            "sheet": {"type": "string", "description": "工作表名，来自 list_sheets"}
        },
        "required": ["sheet"],
    },
)
def load_sheet(ctx: ToolContext, sheet: str) -> str:
    df = ctx.get_frame(sheet)
    dtypes = "\n".join(f"  - {col}: {dtype}" for col, dtype in df.dtypes.items())
    return (
        f"已加载工作表 {sheet!r}: shape={df.shape}\n"
        f"列名: {list(df.columns)}\n数据类型:\n{dtypes}\n"
        f"前 5 行:\n{df.head(5).to_string()}"
    )


# ----------------------------------------------------------------- 聚合体检
@tool(
    name="data_health_check",
    description=(
        "数据体检（聚合工具，优先使用）：一次调用即可返回整表的 结构 + 缺失 + 异常值 信息。"
        "包含行列数、每列类型/缺失率/唯一值/IQR 离群点数量与占比/偏度、类别列 Top 取值、"
        "日期列取值范围，并汇总异常提示（高缺失、常量列、高基数疑似 ID、严重偏态、离群点过多、重复行）。"
        "回答『整体数据结构 / 有没有缺失和异常值 / 数据质量如何』这类问题时优先调用它，"
        "一次即可覆盖，不必再逐列调用其他探查工具。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "sheet": {"type": "string", "description": "工作表名，省略则用当前表"},
            "columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "可选，只体检指定列",
            },
            "max_columns": {"type": "integer", "description": "最多体检多少列，默认 40"},
            "top": {"type": "integer", "description": "类别列展示的 Top 取值个数，默认 3"},
        },
    },
)
def data_health_check(
    ctx: ToolContext,
    sheet: str | None = None,
    columns: list[str] | None = None,
    max_columns: int = 40,
    top: int = 3,
) -> str:
    df = ctx.get_frame(sheet)
    full_df = df
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ToolError(f"列 {missing} 不存在。可用列: {list(df.columns)}")
        df = df[list(columns)]
    if df.shape[1] == 0:
        raise ToolError("没有可体检的列。")

    total_rows, total_cols = df.shape
    limit = max(1, min(int(max_columns or 40), 200))
    n_top = max(1, min(int(top or 3), 10))
    scanned = df.columns[:limit]

    lines = [f"[体检报告 · {ctx.active_sheet}] {total_rows:,} 行 × {total_cols} 列"]
    lines.append(f"列清单: {list(df.columns)}")
    if total_cols > limit:
        lines.append(f"⚠ 列数较多，仅体检前 {limit} 列（可用 columns 参数指定需要体检的列）")

    alerts: list[str] = []

    for col in scanned:
        series = df[col]
        present = series.notna()
        valid = int(present.sum())
        n_unique = int(series.nunique(dropna=True))
        missing_count = int((~present).sum())

        bits = [
            f"类型={series.dtype}",
            f"缺失={_fmt_count(missing_count, total_rows)}",
            f"唯一值={n_unique:,}",
        ]
        note = ""

        if valid == 0:
            note = "整列为空"
        elif n_unique <= 1:
            note = "常量列（无区分度）"
        elif pd.api.types.is_bool_dtype(series):
            top_values = series.value_counts()
            note = "布尔列: " + "/".join(f"{k}:{v:,}" for k, v in top_values.items())
        elif pd.api.types.is_datetime64_any_dtype(series):
            low, high = series.min(), series.max()
            if pd.notna(low) and pd.notna(high):
                days = (pd.Timestamp(high) - pd.Timestamp(low)).days
                bits.append(f"范围={_fmt_date(low)} ~ {_fmt_date(high)}")
                bits.append(f"跨度={days:,}天")
                if days > 0:
                    per_day = valid / (days + 1)
                    note = f"日均 {per_day:.1f} 条"
        elif pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            bits.append(f"均值={_fmt_num(clean.mean())}")
            bits.append(f"中位数={_fmt_num(clean.median())}")
            bits.append(f"最小={_fmt_num(clean.min())}")
            bits.append(f"最大={_fmt_num(clean.max())}")
            if len(clean) >= 4:
                q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    outliers = int(
                        ((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).sum()
                    )
                    if outliers:
                        rate = outliers / len(clean) * 100
                        bits.append(f"离群点={_fmt_count(outliers, len(clean))}")
                        if rate > OUTLIER_WARN:
                            alerts.append(
                                f"离群点偏多: {col}（IQR 口径，{outliers:,} 行，占 {rate:.1f}%）"
                            )
                skew = clean.skew()
                if pd.notna(skew) and abs(float(skew)) > SKEW_WARN:
                    direction = "右" if skew > 0 else "左"
                    bits.append(f"偏度={skew:.2f}")
                    alerts.append(
                        f"严重{direction}偏: {col}（偏度 {skew:.2f}，均值受极端值影响，"
                        f"建议看中位数）"
                    )
        else:
            counts = series.dropna().astype(str).value_counts().head(n_top)
            if len(counts):
                note = "Top: " + "/".join(f"{k}({v:,})" for k, v in counts.items())

        if missing_count and missing_count / total_rows * 100 > MISSING_WARN:
            alerts.append(
                f"高缺失: {col}（{missing_count:,} 行，占 "
                f"{missing_count / total_rows * 100:.1f}%）"
            )
        if n_unique == valid and valid > 20 and not pd.api.types.is_bool_dtype(series):
            alerts.append(f"高基数（疑似主键/ID）: {col}（唯一值 {n_unique:,} = 有效行数）")
        if note:
            bits.append(f"备注={note}")

        lines.append(f"- {col}｜" + "｜".join(bits))

    # 重复行检测只在全表体检时做：按列子集判断「重复」没有业务意义
    if not columns:
        duplicates = int(full_df.duplicated().sum())
        if duplicates:
            alerts.append(
                f"完全重复行 {duplicates:,} 行（占 {duplicates / total_rows * 100:.1f}%）"
            )

    if alerts:
        lines.append("")
        lines.append(f"⚠ 关注点（共 {len(alerts)} 项）:")
        lines.extend(f"  - {item}" for item in alerts)
    else:
        lines.append("")
        lines.append("⚠ 关注点: 未发现明显问题（缺失率、离群点、偏度、重复行均在阈值内）")

    return "\n".join(lines)


# ----------------------------------------------------------------- 查看数据
@tool(
    name="preview_rows",
    description="查看数据前 N 行（含列名），用于确认数据长什么样、有哪些脏值。",
    parameters={
        "type": "object",
        "properties": {
            "sheet": {"type": "string", "description": "工作表名，省略则用当前表"},
            "rows": {"type": "integer", "description": "返回行数，默认 10，最大 50"},
        },
    },
)
def preview_rows(ctx: ToolContext, sheet: str | None = None, rows: int = 10) -> str:
    df = ctx.get_frame(sheet)
    n = max(1, min(int(rows or 10), 50))
    return f"[工作表 {ctx.active_sheet}]\n{df.head(n).to_string()}"


@tool(
    name="column_profile",
    description="逐列画像：数据类型、非空数、缺失数、唯一值数、数值列的均值/极值、文本列的 Top 频次。一次调用即可掌握整表结构。",
    parameters={
        "type": "object",
        "properties": {
            "sheet": {"type": "string", "description": "工作表名，省略则用当前表"},
            "column": {"type": "string", "description": "只分析某一列，省略则分析全部列"},
        },
    },
)
def column_profile(ctx: ToolContext, sheet: str | None = None, column: str | None = None) -> str:
    df = ctx.get_frame(sheet)
    if column:
        if column not in df.columns:
            raise ToolError(f"列 {column!r} 不存在。可用列: {list(df.columns)}")
        df = df[[column]]

    records: list[dict[str, Any]] = []
    for col in df.columns:
        series = df[col]
        record: dict[str, Any] = {
            "列名": col,
            "dtype": str(series.dtype),
            "非空": int(series.notna().sum()),
            "缺失": int(series.isna().sum()),
            "唯一值": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            clean = series.dropna()
            if len(clean):
                record["均值"] = round(float(clean.mean()), 4)
                record["最小"] = round(float(clean.min()), 4)
                record["最大"] = round(float(clean.max()), 4)
        else:
            top = series.dropna().astype(str).value_counts().head(3)
            record["Top值"] = ", ".join(f"{k}({v})" for k, v in top.items())
        records.append(record)

    return f"[列画像 · {ctx.active_sheet}]\n{pd.DataFrame(records).to_string(index=False)}"


@tool(
    name="value_counts",
    description="统计某一列的取值分布（Top N），适合类别列/标签列。",
    parameters={
        "type": "object",
        "properties": {
            "column": {"type": "string", "description": "列名"},
            "sheet": {"type": "string", "description": "工作表名，省略则用当前表"},
            "top": {"type": "integer", "description": "返回前 N 个取值，默认 10，最大 50"},
        },
        "required": ["column"],
    },
)
def value_counts(
    ctx: ToolContext,
    column: str,
    sheet: str | None = None,
    top: int = 10,
) -> str:
    df = ctx.get_frame(sheet)
    if column not in df.columns:
        raise ToolError(f"列 {column!r} 不存在。可用列: {list(df.columns)}")
    counts = df[column].value_counts(dropna=False).head(max(1, min(int(top or 10), 50)))
    return f"[{column} 取值分布 · 共 {len(df)} 行]\n{counts.to_string()}"


@tool(
    name="describe_dataframe",
    description="对数值列做描述统计（count/mean/std/min/四分位/max），也可只统计指定列。",
    parameters={
        "type": "object",
        "properties": {
            "sheet": {"type": "string", "description": "工作表名，省略则用当前表"},
            "columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "可选，只统计这些数值列",
            },
        },
    },
)
def describe_dataframe(
    ctx: ToolContext,
    sheet: str | None = None,
    columns: list[str] | None = None,
) -> str:
    df = ctx.get_frame(sheet)
    target = df[list(columns)] if columns else df
    numeric = target.select_dtypes(include="number")
    if numeric.empty:
        return f"[{ctx.active_sheet}] 该表没有数值列，列类型为: {dict(df.dtypes.astype(str))}"
    return f"[描述统计 · {ctx.active_sheet}]\n{numeric.describe().T.to_string()}"


@tool(
    name="correlation",
    description="计算数值列之间的相关系数矩阵，识别关键影响因素。列很多时只返回 |r| 最大的若干对。",
    parameters={
        "type": "object",
        "properties": {
            "sheet": {"type": "string", "description": "工作表名，省略则用当前表"},
            "method": {
                "type": "string",
                "description": "pearson / spearman / kendall，默认 pearson",
            },
            "top": {"type": "integer", "description": "只返回相关性最高的 N 对，默认 10"},
        },
    },
)
def correlation(
    ctx: ToolContext,
    sheet: str | None = None,
    method: str = "pearson",
    top: int = 10,
) -> str:
    df = ctx.get_frame(sheet)
    method = (method or "pearson").lower()
    if method not in {"pearson", "spearman", "kendall"}:
        raise ToolError("method 只能是 pearson / spearman / kendall")
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        raise ToolError("数值列少于 2 个，无法计算相关性。")

    matrix = numeric.corr(method=method)
    pairs: list[tuple[float, str]] = []
    cols = list(matrix.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            value = matrix.loc[left, right]
            if pd.notna(value):
                pairs.append((abs(float(value)), f"{left} ~ {right}: r={value:.4f}"))
    pairs.sort(reverse=True)
    n = max(1, min(int(top or 10), 50))
    header = f"[相关性 {method} · {ctx.active_sheet}] 共 {len(pairs)} 对，展示 Top {min(n, len(pairs))}"
    body = "\n".join(f"  - {text}" for _, text in pairs[:n])
    return f"{header}\n{body}"


# ----------------------------------------------------------------- 沙箱执行
_BANNED_NAMES = {
    "eval", "exec", "open", "compile", "input", "exit", "quit", "help",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "breakpoint",
    "__import__", "memoryview", "super", "classmethod", "staticmethod",
}

_SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
        "float", "format", "int", "isinstance", "len", "list", "map", "max",
        "min", "pow", "print", "range", "repr", "reversed", "round", "set",
        "sorted", "str", "sum", "tuple", "type", "zip",
    )
    if hasattr(builtins, name)
}


def _validate_code(code: str) -> ast.Expression | ast.Module:
    """用 AST 做白名单校验：禁止 import、dunder、危险内建函数。"""
    limit = get_settings().max_sandbox_code_lines
    if len([ln for ln in code.splitlines() if ln.strip()]) > limit:
        raise ToolError(f"代码过长（超过 {limit} 行），请拆成多次调用。")

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ToolError(f"代码语法错误: 第 {exc.lineno} 行 {exc.msg}") from exc

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ToolError("禁止 import，请直接使用已提供的 pd / np / df。")
        if isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            raise ToolError(f"禁止使用 {node.id}()。")
        if isinstance(node, ast.Attribute):
            parts: list[str] = []
            current: ast.AST = node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name) and current.id in {"__builtins__", "__import__"}:
                raise ToolError("禁止访问解释器内部属性。")
            if any(part.startswith("__") for part in parts):
                raise ToolError("禁止访问以下划线开头的属性。")
    return tree


def _run_code(ctx: ToolContext, code: str) -> Any:
    tree = _validate_code(code)
    df = ctx.get_frame()

    buffer: list[str] = []
    safe_print = lambda *args, **kwargs: buffer.append(" ".join(str(a) for a in args))  # noqa: E731
    scope: dict[str, Any] = {
        "__builtins__": {**_SAFE_BUILTINS, "print": safe_print},
        "pd": pd,
        "np": np,
        "df": df,
        "sheet": ctx.active_sheet,
    }

    # 整段代码只有一个表达式时按 eval 执行，否则按 exec 执行（取 out 变量）
    single_expr = len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr)
    try:
        with redirect_stdout(io.StringIO()):
            if single_expr:
                result = eval(  # noqa: S307 - AST 已做白名单校验
                    compile(ast.Expression(tree.body[0].value), "<pandas>", "eval"), scope
                )
            else:
                exec(  # noqa: S102 - AST 已做白名单校验
                    compile(tree, "<pandas>", "exec"), scope
                )
                result = scope.get("out", "（代码已执行，无 out 变量）")
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"{type(exc).__name__}: {exc}") from exc

    if buffer:
        scope["_printed"] = "\n".join(buffer)
        return f"print 输出:\n{chr(10).join(buffer)}\n\n返回值:\n{result}"
    return result


@tool(
    name="run_pandas",
    description=(
        "在沙箱内执行 pandas 代码（兜底万能工具）。可用变量: df(当前工作表 DataFrame)、"
        "pd、np、sheet。支持表达式或语句；语句形式请把结果赋给 out 变量。"
        "禁止 import、文件/网络操作。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "要执行的 pandas 代码，例如 df.groupby('地区')['销售额'].sum()",
            }
        },
        "required": ["code"],
    },
)
def run_pandas(ctx: ToolContext, code: str) -> Any:
    return _run_code(ctx, code)
