"""结构化产物落盘（W2-2）。

**动机**：工具结果在返回前就被 :func:`format_value` 转成了**截断的文本**，
原始 DataFrame 没有任何地方保留。因此「导出报告里带图表数据」必须先把数据
另存一份。

存放位置：``data/uploads/<session_id>/artifacts/`` —— 由 ``Session.path`` 推导，
**不新增 Session 字段**。会话删除时该目录应一并清理。

超限策略：单表 > 2000 行或 > 50 列**不落盘**，但仍记一条 manifest，导出时如实
标注「结果过大未导出」，而不是静默消失。
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.security.redact import redact_frame

logger = logging.getLogger(__name__)

ARTIFACT_DIR_NAME = "artifacts"
MANIFEST_NAME = "manifest.json"

MAX_ARTIFACT_ROWS = 2000
MAX_ARTIFACT_COLS = 50


@dataclass
class ArtifactRef:
    """一次工具调用的结构化产物记录。"""

    seq: int
    tool: str
    step: int | None
    filename: str | None          # None 表示未落盘（超限）
    rows: int | None = None
    cols: int | None = None
    status: str = "saved"         # saved | too_large
    reason: str = ""

    @property
    def saved(self) -> bool:
        return self.status == "saved" and bool(self.filename)


def artifacts_dir(session_path: Path) -> Path:
    """从 Excel 文件路径推导产物目录（不往 Session 加字段）。"""
    return Path(session_path).parent / ARTIFACT_DIR_NAME


def _load_manifest(directory: Path) -> list[ArtifactRef]:
    manifest = directory / MANIFEST_NAME
    if not manifest.exists():
        return []
    try:
        raw = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("产物清单损坏，按空处理: %s", manifest)
        return []
    refs: list[ArtifactRef] = []
    for item in raw:
        try:
            refs.append(ArtifactRef(**item))
        except TypeError:
            continue
    return refs


def _save_manifest(directory: Path, refs: list[ArtifactRef]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / MANIFEST_NAME).write_text(
        json.dumps([asdict(r) for r in refs], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def list_artifacts(session_path: Path) -> list[ArtifactRef]:
    """读产物清单。清单丢失时退化为扫描目录，至少不丢已有文件。"""
    directory = artifacts_dir(session_path)
    if not directory.is_dir():
        return []
    refs = _load_manifest(directory)
    if refs:
        return refs
    for csv in sorted(directory.glob("*.csv")):
        stem = csv.stem.split("-", 1)
        seq = int(stem[0]) if stem[0].isdigit() else 0
        refs.append(
            ArtifactRef(seq=seq, tool=stem[-1], step=None, filename=csv.name)
        )
    return sorted(refs, key=lambda r: r.seq)


def save_artifact(
    session_path: Path,
    tool: str,
    value: Any,
    *,
    step: int | None = None,
) -> ArtifactRef | None:
    """把工具结果落成 CSV，返回清单记录（未落盘时也返回，带原因）。

    导出内容含**原始数据值**，因此落盘前先过一遍 redact_frame。
    """
    if not isinstance(value, (pd.DataFrame, pd.Series)):
        return None

    frame = value.to_frame() if isinstance(value, pd.Series) else value
    directory = artifacts_dir(session_path)
    refs = list_artifacts(session_path)
    seq = max((r.seq for r in refs), default=0) + 1

    rows, cols = int(frame.shape[0]), int(frame.shape[1])
    if rows == 0 or cols == 0:
        return None

    if rows > MAX_ARTIFACT_ROWS or cols > MAX_ARTIFACT_COLS:
        ref = ArtifactRef(
            seq=seq,
            tool=tool,
            step=step,
            filename=None,
            rows=rows,
            cols=cols,
            status="too_large",
            reason=(
                f"结果 {rows} 行 x {cols} 列，超过上限"
                f"（{MAX_ARTIFACT_ROWS} 行 / {MAX_ARTIFACT_COLS} 列），未导出"
            ),
        )
        refs.append(ref)
        _save_manifest(directory, refs)
        logger.info("产物未落盘（超限） | tool=%s | %sx%s", tool, rows, cols)
        return ref

    safe = redact_frame(frame)
    filename = f"{seq:03d}-{_safe_name(tool)}.csv"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        # utf-8-sig：Excel 不认无 BOM 的 UTF-8 CSV，中文会乱码
        safe.to_csv(directory / filename, index=False, encoding="utf-8-sig")
    except (OSError, ValueError):
        logger.warning("产物落盘失败 | tool=%s", tool, exc_info=True)
        return None

    ref = ArtifactRef(
        seq=seq, tool=tool, step=step, filename=filename, rows=rows, cols=cols
    )
    refs.append(ref)
    _save_manifest(directory, refs)
    return ref


def load_artifact(session_path: Path, ref: ArtifactRef) -> pd.DataFrame | None:
    """读回一个产物。文件可能已被手工删除，因此失败返回 None。"""
    if not ref.filename:
        return None
    path = artifacts_dir(session_path) / ref.filename
    if not path.exists():
        logger.warning("产物文件缺失: %s", path)
        return None
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, ValueError):
        logger.warning("产物读取失败: %s", path, exc_info=True)
        return None


def _safe_name(name: str) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in (name or "tool")]
    return "".join(keep)[:40] or "tool"