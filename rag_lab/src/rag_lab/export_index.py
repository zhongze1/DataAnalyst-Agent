"""DA-018：把分块与向量导出成**主工程能直接读的服务工件**。

为什么用"数据文件 + 主工程自己实现检索"，而不是让主工程 import `rag_lab`：

- 两个子工程的 venv 与依赖完全分开（`rag_lab` 有自己的 `.venv`），
  主工程必须能**独立部署**；
- 接口是数据格式，不是代码耦合 —— 换掉 lab 的实现不影响线上服务。

产物（默认写到主工程的 `data/knowledge/`）：

| 文件 | 内容 |
| --- | --- |
| ``knowledge_index.jsonl`` | 一行一个片段：与 chunks.jsonl 同字段 + ``vector`` |
| ``manifest.json`` | ``space / provider / model / dim / chunks / corpus 文件哈希 / built_at`` |

**不允许导出半截索引**：只要有片段缺向量（或向量维度不一致）就整体失败，
并在错误信息里说清缺多少 —— 半截索引会让线上"悄悄少召回一截"，比直接失败危险得多。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rag_lab.chunker import LAB_ROOT, Chunk, iter_corpus_files, split_document
from rag_lab.embedding import (
    DEFAULT_CACHE_FILE,
    DEFAULT_CHUNKS_FILE,
    EmbeddingCache,
    EmbeddingConfig,
    EmbeddingError,
    config_from_env,
    load_chunks,
)

REPO_ROOT = LAB_ROOT.parent
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "knowledge"
INDEX_FILE = "knowledge_index.jsonl"
MANIFEST_FILE = "manifest.json"


class ExportError(RuntimeError):
    """导出失败（缺向量、维度不一致、没有片段……）。"""

    code = "EXPORT_FAILED"


@dataclass(frozen=True)
class ExportResult:
    """一次导出的结果：写了哪些文件、多少条。"""

    index_path: Path
    manifest_path: Path
    chunks: int
    dim: int
    space: str


def index_row(chunk: Chunk, vector: list[float]) -> dict[str, object]:
    """一行索引：沿用链路字段名，外加 ``vector``（主工程读起来不用做映射）。"""
    return {
        **chunk.to_dict(),
        "vector": vector,
    }


def _corpus_fingerprint(
    chunks: list[Chunk], lab_root: Path
) -> list[dict[str, object]]:
    """语料文件指纹：用来回答"这份知识库是哪一版语料构建的"。"""
    per_doc: dict[str, int] = {}
    for chunk in chunks:
        per_doc[chunk.source_path] = per_doc.get(chunk.source_path, 0) + 1

    fingerprints: list[dict[str, object]] = []
    corpus_dir = lab_root / "corpus"
    for path in iter_corpus_files(corpus_dir):
        relative = path.relative_to(lab_root).as_posix()
        raw = path.read_bytes()
        fingerprints.append(
            {
                "source_path": relative,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest()[:16],
                "chunks": per_doc.get(relative, 0),
            }
        )
    return fingerprints


def build_index(
    chunks: list[Chunk],
    cache: EmbeddingCache,
    config: EmbeddingConfig,
    *,
    lab_root: Path | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """组装索引行与 manifest；**任何缺失/不一致都直接报错**。"""
    if not chunks:
        raise ExportError("没有片段可导出（先跑 uv run python -m rag_lab.chunker）")

    rows: list[dict[str, object]] = []
    missing: list[str] = []
    dimensions: set[int] = set()
    for chunk in chunks:
        vector = cache.get(config.space, chunk.content_hash)
        if vector is None:
            missing.append(chunk.chunk_id)
            continue
        dimensions.add(len(vector))
        rows.append(index_row(chunk, vector))

    if missing:
        raise ExportError(
            f"有 {len(missing)} 个片段没有向量，拒绝导出半截索引"
            f"（例如 {', '.join(missing[:3])}）"
            "；先跑 uv run python -m rag_lab.embedding"
        )
    if len(dimensions) != 1:
        raise ExportError(f"向量维度不一致：{sorted(dimensions)}，拒绝导出")

    dim = next(iter(dimensions))
    manifest = {
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "space": config.space,
        "provider": config.provider,
        "model": config.model,
        "dim": dim,
        "chunks": len(rows),
        "ngram": 2,
        "corpus": _corpus_fingerprint(chunks, Path(lab_root or LAB_ROOT)),
    }
    return rows, manifest


def write_index(
    rows: list[dict[str, object]],
    manifest: dict[str, object],
    out_dir: Path | None = None,
) -> ExportResult:
    """落盘。先写临时文件再改名，避免中途失败留下半截索引。"""
    target = Path(out_dir or DEFAULT_OUT_DIR)
    target.mkdir(parents=True, exist_ok=True)
    index_path = target / INDEX_FILE
    manifest_path = target / MANIFEST_FILE

    body = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    temporary = index_path.with_suffix(".jsonl.tmp")
    temporary.write_text(body, encoding="utf-8")
    temporary.replace(index_path)  # 原子替换

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return ExportResult(
        index_path=index_path,
        manifest_path=manifest_path,
        chunks=len(rows),
        dim=int(manifest["dim"]),
        space=str(manifest["space"]),
    )


def load_index(
    index_path: Path | None = None, manifest_path: Path | None = None
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """读回索引（测试与人工核对用；主工程有自己的读取实现）。"""
    target = Path(index_path or (DEFAULT_OUT_DIR / INDEX_FILE))
    if not target.exists():
        raise ExportError(f"索引不存在: {target}")
    rows = [
        json.loads(line)
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest_file = Path(manifest_path or (target.parent / MANIFEST_FILE))
    manifest = (
        json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest_file.exists()
        else {}
    )
    return rows, manifest


def export(
    *,
    chunks_path: Path | None = None,
    cache_path: Path | None = None,
    out_dir: Path | None = None,
    lab_root: Path | None = None,
) -> ExportResult:
    """完整流程：读分块 → 校验向量齐备 → 写索引与 manifest。"""
    config = config_from_env()
    chunks = load_chunks(chunks_path or config.chunks_path)
    cache = EmbeddingCache(cache_path or config.cache_path)
    rows, manifest = build_index(
        chunks, cache, config, lab_root=lab_root or LAB_ROOT
    )
    return write_index(rows, manifest, out_dir)


def main(argv: list[str] | None = None) -> int:
    """命令行入口：``uv run python -m rag_lab.export_index``。"""
    parser = argparse.ArgumentParser(
        prog="python -m rag_lab.export_index",
        description="把分块与向量导出成主工程可读的服务工件（DA-018）",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_FILE)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_FILE)
    args = parser.parse_args(argv)

    try:
        result = export(chunks_path=args.chunks, cache_path=args.cache, out_dir=args.out_dir)
    except EmbeddingError as exc:
        print(f"[{exc.code}] {exc}", flush=True)
        return 3
    except ExportError as exc:
        print(f"[{exc.code}] {exc}", flush=True)
        return 4

    print(f"片段数: {result.chunks} | 维度: {result.dim} | 空间: {result.space}")
    print(f"索引: {result.index_path}")
    print(f"清单: {result.manifest_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - 由 CLI 测试覆盖 main()
    sys.exit(main())
