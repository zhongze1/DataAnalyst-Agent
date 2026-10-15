"""DA-018 服务工件导出的测试（**全离线**：假向量，不碰向量服务）。

重点：
- 索引行字段与 manifest 的完整性（主工程就靠它读）；
- **拒绝导出半截索引**（缺向量 / 维度不一致都必须整体失败，且不留文件）；
- 语料指纹能反映语料变化（"这份知识库是哪一版"）；
- 落盘是原子的（临时文件 + replace）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_lab.chunker import LAB_ROOT, split_document
from rag_lab.embedding import EmbeddingCache, EmbeddingConfig, build_embedder, embed_chunks
from rag_lab.export_index import (
    DEFAULT_OUT_DIR,
    INDEX_FILE,
    MANIFEST_FILE,
    ExportError,
    build_index,
    export,
    load_index,
    main,
)


def _corpus(tmp_path: Path, texts: dict[str, str]) -> Path:
    """在 tmp 下搭一个假语料目录（lab_root 布局：corpus/ 在其下）。"""
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        (corpus / name).write_text(text, encoding="utf-8")
    return corpus


def _chunks_from_corpus(tmp_path: Path, *, dim: int = 0):
    """切现有 corpus/ 下的文件 + 用 FakeEmbedder 写向量（**不重建语料**）。"""
    corpus = tmp_path / "corpus"
    chunks = []
    for path in sorted(corpus.glob("*.md")):
        chunks.extend(
            split_document(
                path.read_text(encoding="utf-8"),
                doc_id=path.stem,
                source_path=path.relative_to(tmp_path).as_posix(),
            )
        )
    config = EmbeddingConfig(provider="fake", dim=dim, cache_path=tmp_path / "emb.jsonl")
    cache = EmbeddingCache(config.cache_path)
    embed_chunks(chunks, build_embedder(config), cache, space=config.space, batch_size=10)
    return chunks, config, cache


def _chunks(tmp_path: Path, texts: dict[str, str], *, dim: int = 0):
    """建语料 + 切块 + 写向量，返回 (chunks, config, cache)。"""
    _corpus(tmp_path, texts)
    return _chunks_from_corpus(tmp_path, dim=dim)


SAMPLE = {
    "甲.md": "# 甲文档\n\n客单价的分母是下单客户数。\n\n## 第二节\n\n另一种口径是笔单价。\n",
    "乙.md": "# 乙文档\n\n沙箱禁止 import 与文件读写。\n",
}


# ============================================================ 组装
def test_build_index_rows_and_manifest(tmp_path: Path) -> None:
    chunks, config, cache = _chunks(tmp_path, SAMPLE)

    rows, manifest = build_index(chunks, cache, config, lab_root=tmp_path)

    assert len(rows) == len(chunks)
    first = rows[0]
    for field in (
        "chunk_id", "doc_id", "source_path", "section", "text",
        "line_start", "line_end", "content_hash", "vector",
    ):
        assert field in first, f"索引行缺字段 {field}"
    assert all(len(row["vector"]) == manifest["dim"] for row in rows)

    assert manifest["chunks"] == len(chunks)
    assert manifest["space"] == config.space
    assert manifest["ngram"] == 2
    assert manifest["dim"] > 0
    assert {item["source_path"] for item in manifest["corpus"]} == {"corpus/甲.md", "corpus/乙.md"}
    assert all(item["chunks"] > 0 and item["sha256"] for item in manifest["corpus"])


def test_build_index_refuses_partial_index_when_vectors_missing(tmp_path: Path) -> None:
    chunks, config, _ = _chunks(tmp_path, SAMPLE)
    # 换一个空缓存：所有向量都"缺失"
    empty = EmbeddingCache(tmp_path / "empty.jsonl")

    with pytest.raises(ExportError, match="没有向量"):
        build_index(chunks, empty, config, lab_root=tmp_path)


def test_build_index_rejects_inconsistent_dimensions(tmp_path: Path) -> None:
    chunks, config, cache = _chunks(tmp_path, SAMPLE)
    cache.put(config.space, chunks[0].content_hash, [0.1, 0.2, 0.3])  # 故意换个维度

    with pytest.raises(ExportError, match="维度不一致"):
        build_index(chunks, cache, config, lab_root=tmp_path)


def test_build_index_rejects_empty_input(tmp_path: Path) -> None:
    _, config, cache = _chunks(tmp_path, SAMPLE)
    with pytest.raises(ExportError, match="没有片段"):
        build_index([], cache, config, lab_root=tmp_path)


# ============================================================ 落盘
def test_write_and_reload_roundtrip(tmp_path: Path) -> None:
    result = export(
        chunks_path=_write_chunks(tmp_path, SAMPLE),
        cache_path=tmp_path / "emb.jsonl",
        out_dir=tmp_path / "out",
        lab_root=tmp_path,
    )

    assert result.index_path.exists() and result.manifest_path.exists()
    rows, manifest = load_index(result.index_path, result.manifest_path)
    assert len(rows) == result.chunks and manifest["dim"] == result.dim
    assert rows[0]["text"] and rows[0]["vector"]
    # 不留临时文件
    assert not list(result.index_path.parent.glob("*.tmp"))


def test_load_index_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ExportError, match="索引不存在"):
        load_index(tmp_path / "nope.jsonl")


def test_manifest_tracks_corpus_changes(tmp_path: Path) -> None:
    """"这份知识库是哪一版语料构建的" —— 靠语料指纹回答。"""
    chunks_path = _write_chunks(tmp_path, SAMPLE)
    first = export(
        chunks_path=chunks_path,
        cache_path=tmp_path / "emb.jsonl",
        out_dir=tmp_path / "out",
        lab_root=tmp_path,
    )
    _, before = load_index(first.index_path, first.manifest_path)

    # 改语料 → 重新切块 + 重新组装（注意不能重建语料，否则改动会被覆盖）
    (tmp_path / "corpus" / "乙.md").write_text(
        "# 乙文档\n\n沙箱禁止 import 与文件读写。\n\n新增一段。\n", encoding="utf-8"
    )
    chunks, config, cache = _chunks_from_corpus(tmp_path)
    rows, after = build_index(chunks, cache, config, lab_root=tmp_path)

    fingerprints_before = {item["source_path"]: item["sha256"] for item in before["corpus"]}
    fingerprints_after = {item["source_path"]: item["sha256"] for item in after["corpus"]}
    assert fingerprints_before["corpus/乙.md"] != fingerprints_after["corpus/乙.md"]
    assert fingerprints_before["corpus/甲.md"] == fingerprints_after["corpus/甲.md"]
    assert after["chunks"] == len(rows)


def test_default_out_dir_points_into_the_main_project() -> None:
    """默认落在主工程的 data/knowledge/ —— 这是有意的耦合，用测试写清楚。"""
    assert DEFAULT_OUT_DIR == LAB_ROOT.parent / "data" / "knowledge"
    assert INDEX_FILE == "knowledge_index.jsonl" and MANIFEST_FILE == "manifest.json"


# ============================================================ CLI
def _write_chunks(tmp_path: Path, texts: dict[str, str]) -> Path:
    chunks, _, _ = _chunks(tmp_path, texts)
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        "\n".join(json.dumps(chunk.to_dict(), ensure_ascii=False) for chunk in chunks) + "\n",
        encoding="utf-8",
    )
    return path


def test_cli_exports_the_index(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    code = main(
        [
            "--chunks",
            str(_write_chunks(tmp_path, SAMPLE)),
            "--cache",
            str(tmp_path / "emb.jsonl"),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )

    assert code == 0
    printed = capsys.readouterr().out
    assert "片段数:" in printed and "清单:" in printed
    rows, manifest = load_index(tmp_path / "out" / INDEX_FILE)
    assert rows and manifest["dim"] > 0


def test_cli_reports_missing_vectors_without_writing(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    out = tmp_path / "out"
    code = main(
        [
            "--chunks",
            str(_write_chunks(tmp_path, SAMPLE)),
            "--cache",
            str(tmp_path / "empty.jsonl"),
            "--out-dir",
            str(out),
        ]
    )

    assert code == 4
    assert "EXPORT_FAILED" in capsys.readouterr().out
    assert not (out / INDEX_FILE).exists(), "失败时不能留下半截索引"


def test_cli_reports_missing_chunks_file(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    code = main(
        [
            "--chunks",
            str(tmp_path / "nope.jsonl"),
            "--cache",
            str(tmp_path / "emb.jsonl"),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert code == 3
    assert "EMBEDDING_CONFIG" in capsys.readouterr().out
