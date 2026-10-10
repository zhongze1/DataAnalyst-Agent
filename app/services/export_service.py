"""分析报告生成（W2-2）。

两种格式：
- **Markdown**：给同事直接贴到邮件/IM 里看
- **Excel**：结论 + 执行轨迹 + 每个「数据明细」一个 sheet（可自行复核）

复用现有结构，不另造数据模型：
- 结论/问答：``session.messages``（已在 W2-1 落库）
- 执行轨迹：assistant 消息上的 ``steps_json``（``save_turn`` 已写入）
- 数据明细：``app.storage.artifacts`` 的 CSV 产物

**所有对外文本都过 :mod:`app.security.redact`**，W2-4 只需在那一个模块加规则。
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.logging_setup import describe_arguments
from app.security.redact import redact_text, redact_value
from app.storage.artifacts import ArtifactRef, list_artifacts, load_artifact

# Excel sheet 名限制：<=31 字符，且不能含 : \ / ? * [ ]
_ILLEGAL_SHEET_CHARS = ':\\/?*[]'


@dataclass
class ExportBundle:
    """导出产物。``filename`` 建议的下载名，``content`` 是字节内容。"""

    filename: str
    content: bytes
    media_type: str


def collect_turns(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把消息序列拆成「问答轮次」。

    只取 user / assistant —— tool 消息的结论已包含在assistant 回复里，
    单独列出会重复且难懂。
    """
    turns: list[dict[str, Any]] = []
    pending_question: str | None = None
    for message in messages:
        role = message.get("role")
        if role == "user":
            pending_question = redact_text(str(message.get("content") or ""))
            continue
        if role == "assistant":
            content = str(message.get("content") or "")
            if not content.strip():
                # 只调工具、没有文字结论的中间轮次，并入下一轮
                if pending_question is None:
                    pending_question = ""
                continue
            turns.append(
                {
                    "question": pending_question or "",
                    "answer": redact_text(content),
                    "steps": _steps_of(message),
                }
            )
            pending_question = None
    if pending_question:
        # 有提问但模型没给出文字结论（例如撞上 max_steps）
        turns.append({"question": pending_question, "answer": "", "steps": []})
    return turns


def _steps_of(message: dict[str, Any]) -> list[dict[str, Any]]:
    """取该轮的工具轨迹。

    正常路径下 steps 由 :meth:`SessionStore.save_turn` 挂在 assistant 消息上；
    若来自旧数据或结构不符，退回从 ``tool`` 消息里粗略还原，保证不空。
    """
    steps = message.get("steps")
    if isinstance(steps, list):
        return [s for s in steps if isinstance(s, dict)]
    return []


def build_markdown(
    *,
    filename: str,
    session_id: str,
    turns: list[dict[str, Any]],
    artifacts: list[ArtifactRef],
    session_path,
    generated_at: str,
    only_last_turn: bool = False,
) -> ExportBundle:
    """生成 Markdown 报告。"""
    if only_last_turn and turns:
        turns = turns[-1:]
    artifact_rows = _artifact_rows(artifacts)

    lines: list[str] = [
        f"# 分析报告 · {filename}",
        "",
        f"- 会话：`{session_id}`",
        f"- 生成时间：{generated_at}",
        f"- 对话轮次：{len(turns)}",
        "",
    ]

    if not turns:
        lines += ["> 本次会话还没有分析结论。", ""]

    for index, turn in enumerate(turns, start=1):
        heading = f"## 第 {index} 轮"
        if only_last_turn:
            heading = "## 结论"
        lines.append(heading)
        lines.append("")
        if turn["question"]:
            lines += [f"**提问**：{turn['question']}", ""]
        lines += [turn["answer"] or "_（本轮无文字结论）_", ""]
        if turn["steps"]:
            lines.append("<details><summary>本轮执行过程</summary>")
            lines.append("")
            lines += _trace_table(turn["steps"])
            lines += ["", "</details>", ""]

    if artifact_rows:
        lines += ["## 数据明细", "", "以下内容见同名 Excel 文件的数据 sheet。", ""]
        lines += _artifact_table(artifact_rows)
        lines.append("")

    lines += ["---", "", f"由 DataAnalyst Agent 生成 · {generated_at}", ""]

    return ExportBundle(
        filename=f"{Path(filename).stem}-分析报告.md",
        content="\n".join(lines).encode("utf-8"),
        media_type="text/markdown; charset=utf-8",
    )


# --------------------------------------------------------------------- Excel
def build_xlsx(
    *,
    filename: str,
    session_id: str,
    turns: list[dict[str, Any]],
    artifacts: list[ArtifactRef],
    session_path,
    generated_at: str,
    only_last_turn: bool = False,
) -> ExportBundle:
    """生成 Excel 报告：结论 / 执行轨迹 / 每个产物一个 sheet。"""
    from openpyxl import Workbook

    if only_last_turn and turns:
        turns = turns[-1:]

    book = Workbook()
    sheet = book.active
    sheet.title = "结论"

    meta_rows = [
        ["数据文件", filename],
        ["会话 ID", session_id],
        ["生成时间", generated_at],
        ["对话轮次", len(turns)],
    ]
    for row in meta_rows:
        sheet.append(row)
    sheet.append([])

    if not turns:
        sheet.append(["本次会话还没有分析结论。"])
    for index, turn in enumerate(turns, start=1):
        label = "结论" if only_last_turn else f"第 {index} 轮"
        sheet.append([f"【{label}】"])
        if turn["question"]:
            sheet.append(["提问", turn["question"]])
        for line in (turn["answer"] or "（本轮无文字结论）").splitlines():
            if line.strip():
                sheet.append([redact_text(line)])
        sheet.append([])

    _write_trace_sheet(book, turns)
    _write_artifact_sheets(book, artifacts, session_path)

    buffer = io.BytesIO()
    book.save(buffer)
    return ExportBundle(
        filename=f"{Path(filename).stem}-分析报告.xlsx",
        content=buffer.getvalue(),
        media_type=(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
    )


def _write_trace_sheet(book, turns: list[dict[str, Any]]) -> None:
    sheet = book.create_sheet("执行轨迹")
    sheet.append(["轮次", "步骤", "工具", "参数", "状态", "结果摘要"])
    for round_index, turn in enumerate(turns, start=1):
        for step in turn["steps"]:
            ok = bool(step.get("ok", True))
            # DA-015：轨迹来自落库的 steps_json，可能是 DA-015 之前存下的**明文**，
            # 导出时再脱一次（与下面「数据明细」的处理同理），否则历史会话照样泄露
            summary = redact_text(str(step.get("error") or step.get("result") or ""))
            sheet.append(
                [
                    round_index,
                    step.get("step", ""),
                    str(step.get("tool", "")),
                    describe_arguments(step.get("arguments") or {}),
                    "成功" if ok else "失败",
                    _excel_safe(summary[:300]),
                ]
            )
    _autofit(sheet, [8, 6, 20, 30, 8, 80])


def _write_artifact_sheets(book, artifacts, session_path) -> None:
    saved = [a for a in artifacts if a.saved]
    skipped = [a for a in artifacts if not a.saved]
    if not artifacts:
        return

    if saved:
        for ref in saved:
            frame = load_artifact(session_path, ref)
            if frame is None or frame.empty:
                continue
            title = _sheet_title(f"数据-{ref.seq}-{ref.tool}")
            sheet = book.create_sheet(title)
            # 再次脱敏：产物是历史落盘的，规则可能已更新
            safe = redact_value(frame)
            sheet.append([_excel_safe(str(c)) for c in safe.columns])
            for row in safe.itertuples(index=False):
                sheet.append([_excel_safe(v) for v in row])
            _autofit(sheet)

    if skipped:
        sheet = book.create_sheet("未导出说明")
        sheet.append(["步骤", "工具", "原因"])
        for ref in skipped:
            sheet.append([ref.step or "", ref.tool, ref.reason])
        _autofit(sheet, [8, 20, 60])


def _trace_table(steps: list[dict[str, Any]]) -> list[str]:
    """Markdown 形式的执行轨迹表。"""
    lines = [
        "| 步骤 | 工具 | 参数 | 状态 | 结果摘要 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for step in steps:
        ok = bool(step.get("ok", True))
        # DA-015：同 xlsx 轨迹，历史落库的明文也要在这里兜住
        summary = redact_text(str(step.get("error") or step.get("result") or ""))
        summary = summary.replace("|", "\\|").replace("\n", " ")[:120]
        lines.append(
            f"| {step.get('step', '')} | `{step.get('tool', '')}` "
            f"| `{describe_arguments(step.get('arguments') or {})}` "
            f"| {'成功' if ok else '失败'} | {summary} |"
        )
    return lines


def _artifact_rows(artifacts: list[ArtifactRef]) -> list[dict[str, Any]]:
    return [
        {
            "seq": ref.seq,
            "saved": ref.saved,
            "name": f"数据-{ref.seq}-{ref.tool}",
            "rows": ref.rows,
            "cols": ref.cols,
            "reason": ref.reason,
        }
    for ref in artifacts]


def _artifact_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| Sheet | 状态 | 规模 |", "| --- | --- | --- |"]
    for row in rows:
        if row["saved"]:
            detail = "已导出"
            size = f"{row['rows']} 行 &times; {row['cols']} 列"
        else:
            detail = "**未导出**"
            size = row["reason"] or "结果过大"
        lines.append(f"| {row['name']} | {detail} | {size} |")
    return lines


def _excel_safe(value: Any) -> Any:
    """防 CSV/Excel 公式注入。

    以 = + - @ 开头的单元格会被 Excel 当成公式执行（=cmd|'...'!A1 可读本地文件）。
    """
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def _sheet_title(title: str) -> str:
    """Excel sheet 名限制：<=31 字符、不能含 : \\ / ? * [ ]、不能重复。"""
    cleaned = "".join("_" if c in _ILLEGAL_SHEET_CHARS else c for c in title)[:31]
    return cleaned or "Sheet"


def _autofit(sheet, widths: list[int] | None = None) -> None:
    """按内容估算列宽。openpyxl 不支持 auto-fit，只能近似。"""
    from openpyxl.utils import get_column_letter

    if widths:
        for index, width in enumerate(widths, start=1):
            sheet.column_dimensions[get_column_letter(index)].width = width
        return
    max_width = 12
    for column_cells in sheet.columns:
        length = 0
        for cell in column_cells:
            if cell.value is not None:
                length = max(length, len(str(cell.value)))
        sheet.column_dimensions[get_column_letter(column_cells[0].column)].width = min(
            max(length + 2, 8), 50
        )
        max_width = max(max_width, 1)
    _ = max_width