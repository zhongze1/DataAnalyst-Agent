"""RAG-04 检索测试（**全离线**：FakeEmbedder + tmp 缓存，不碰向量服务）。

覆盖重点：
- 文本归一化与字符 2-gram；
- BM25 的关键性质（命中专有词、长度归一化）；
- 查询向量**进缓存**（重复跑不重复计费）；
- **三种结果各自独立**（这正是本阶段的设计要求）；
- RRF 的数学与"两路都进榜的优先"；
- 边界：k 大于语料数、纯标点查询、向量没备齐时的报错。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_lab.chunker import Chunk, write_jsonl
from rag_lab.embedding import (
    EmbeddingCache,
    EmbeddingConfig,
    build_embedder,
    embed_chunks,
)
from rag_lab.retrieve import (
    BM25Index,
    RRF_POOL_FACTOR,
    Hit,
    Retriever,
    char_ngrams,
    cosine,
    main,
    normalize,
    query_cache_key,
)

DOC = "指标定义"
ARCH = "架构说明"


def _chunk(text: str, *, doc_id: str = DOC, section: str = "s", index: int = 1) -> Chunk:
    import hashlib

    return Chunk(
        chunk_id=f"{doc_id}#{index:04d}",
        doc_id=doc_id,
        source_path=f"corpus/{doc_id}.md",
        section=section,
        text=text,
        line_start=1,
        line_end=1,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
    )


CORPUS_TEXTS = {
    "客单价的分母是下单客户数，另一种口径是笔单价。": (DOC, "3. 客单价"),
    "毛利率 = 毛利 / 订单金额，已取消订单必须剔除后再算。": (DOC, "2. 利润率"),
    "SSE 事件包括 start / tool_args / step / token / done / error。": (ARCH, "6. 流式响应"),
    "沙箱禁止 import 与文件读写，违规返回 SANDBOX_VIOLATION。": (ARCH, "4. 工具注册与执行"),
}


def _build(tmp_path: Path, *, dim: int = 0) -> tuple[list[Chunk], EmbeddingConfig, EmbeddingCache]:
    """小语料 + 把向量真正写进缓存。

    ``dim=0``（默认）时空间是 ``dim=auto`` —— 必须和命令行 `config_from_env()`
    算出来的空间一致，否则 CLI 会认为"一条向量都没有"。
    """
    chunks = [
        _chunk(text, doc_id=doc, section=section, index=index)
        for index, (text, (doc, section)) in enumerate(CORPUS_TEXTS.items(), start=1)
    ]
    config = EmbeddingConfig(provider="fake", dim=dim, cache_path=tmp_path / "emb.jsonl")
    cache = EmbeddingCache(config.cache_path)
    # 用 build_embedder 而不是手搓 FakeEmbedder：与生产路径（含 fake 的默认维度）保持一致
    embed_chunks(chunks, build_embedder(config), cache, space=config.space, batch_size=10)
    return chunks, config, cache


# ============================================================ 文本处理
def test_normalize_drops_punctuation_and_keeps_cjk_and_alnum() -> None:
    assert normalize("客单价（元）：1,234.5 元/人！") == "客单价元12345元人"


def test_char_ngrams_basic_and_edges() -> None:
    assert char_ngrams("abcde") == ["ab", "bc", "cd", "de"]
    assert char_ngrams("猫") == ["猫"], "短于 n 的文本整体作为一个 gram"
    assert char_ngrams("！？…") == [], "纯标点没有可用的 gram"
    assert char_ngrams("") == []


def test_cosine_handles_degenerate_inputs() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0, "零向量不该炸（返回 0）"
    assert cosine([1.0, 0.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0), "维度不一致时按短的算"


# ============================================================ BM25
def test_bm25_ranks_the_document_with_the_term_first() -> None:
    chunks = [_chunk(text, index=index) for index, text in enumerate(CORPUS_TEXTS, start=1)]
    index = BM25Index(chunks)
    found = index.search("客单价的分母是谁", 3)

    assert found, "应当有命中"
    top_index, _, grams = found[0]
    assert "客单价" in chunks[top_index].text
    assert grams, "必须给出命中的 gram，用来解释为什么召回"


def test_bm25_length_normalisation_prefers_short_chunk() -> None:
    """同一个词只出现一次时，短片段应当排前面 —— 这正是用 BM25 而不是数重叠的原因。"""
    short = _chunk("毛利率的定义。", index=1)
    long = _chunk("填充" * 200 + "毛利率" + "填充" * 200, index=2)
    found = BM25Index([long, short]).search("毛利率", 2)

    assert found[0][0] == 1, "短片段（下标 1）应当胜出"
    assert found[0][1] > found[1][1]


def test_bm25_returns_nothing_for_unrelated_query() -> None:
    chunks = [_chunk(text, index=index) for index, text in enumerate(CORPUS_TEXTS, start=1)]
    assert BM25Index(chunks).search("完全不相关的词xyz", 3) == []


# ============================================================ 向量检索
def test_vector_search_caches_the_query_vector(tmp_path: Path) -> None:
    """查询向量必须进缓存：评估脚本重复跑时不该重复计费。"""
    chunks, config, cache = _build(tmp_path)
    retriever = Retriever(chunks, config, cache=cache)
    query = "客单价的分母"

    assert cache.get(f"{config.space}|query", query_cache_key(query)) is None
    retriever.vector_search(query, 3)
    assert cache.get(f"{config.space}|query", query_cache_key(query)) is not None

    before = cache.path.read_text(encoding="utf-8").count("\n")
    retriever.vector_search(query, 3)
    after = cache.path.read_text(encoding="utf-8").count("\n")
    assert before == after, "第二次检索应命中缓存，不再写新行"


def test_vector_search_puts_identical_text_first(tmp_path: Path) -> None:
    chunks, config, cache = _build(tmp_path)
    retriever = Retriever(chunks, config, cache=cache)
    text = next(iter(CORPUS_TEXTS))

    hits = retriever.vector_search(text, 3)
    assert hits[0].chunk.text == text, "与查询完全相同的片段必须是第 1 名"
    assert hits[0].score == pytest.approx(1.0)
    assert "语义相似度" in hits[0].why


def test_vector_search_reports_missing_vectors(tmp_path: Path) -> None:
    chunks, config, _ = _build(tmp_path)
    empty_cache = EmbeddingCache(tmp_path / "empty.jsonl")
    retriever = Retriever(chunks, config, cache=empty_cache)

    with pytest.raises(ValueError, match="没有向量"):
        retriever.vector_search("随便问问", 3)
    assert retriever.vector_coverage == f"0/{len(chunks)} 个片段已有向量"


# ============================================================ 三种结果独立
def test_keyword_hit_appears_in_keyword_and_hybrid_only_if_present(
    tmp_path: Path,
) -> None:
    """三种结果各自独立：keyword 命中的专有词，不保证向量也命中。"""
    chunks, config, cache = _build(tmp_path)
    retriever = Retriever(chunks, config, cache=cache)

    result = retriever.search("SANDBOX_VIOLATION", k=4, mode="all")

    assert result.keyword, "专有词应当被关键词检索命中"
    assert result.keyword[0].chunk.text.startswith("沙箱禁止")
    assert "命中 2-gram" in result.keyword[0].why
    assert result.hybrid, "混合结果里应当包含它（RRF 会把关键词的贡献算进去）"
    assert {hit.chunk.chunk_id for hit in result.keyword} & {
        hit.chunk.chunk_id for hit in result.hybrid
    }, "混合结果必须覆盖关键词的第一名"
    # 三份结果互不覆盖
    assert result.vector is not result.keyword is not result.hybrid


def test_mode_selects_only_that_result(tmp_path: Path) -> None:
    chunks, config, cache = _build(tmp_path)
    retriever = Retriever(chunks, config, cache=cache)

    only_keyword = retriever.search("毛利率", k=2, mode="keyword")
    assert only_keyword.keyword and not only_keyword.vector and not only_keyword.hybrid

    only_vector = retriever.search("毛利率", k=2, mode="vector")
    assert only_vector.vector and not only_vector.keyword and not only_vector.hybrid

    hybrid = retriever.search("毛利率", k=2, mode="hybrid")
    # 只跑混合时，两路只是内部输入：对外只给融合结果（要三份齐看请用 mode="all"）
    assert hybrid.hybrid and not hybrid.vector and not hybrid.keyword
    assert "#" in hybrid.hybrid[0].why, "融合结果必须说明它两路的排名"


# ============================================================ RRF
def test_rrf_score_and_ordering() -> None:
    a = _chunk("甲", index=1)
    b = _chunk("乙", index=2)
    fused = Retriever.rrf_fuse(
        {"vector": [Hit(a, 0.9, 1, "v1"), Hit(b, 0.8, 2, "v2")], "keyword": [Hit(b, 5.0, 1, "k1")]},
        k=5,
    )

    assert [hit.chunk.chunk_id for hit in fused] == [b.chunk_id, a.chunk_id]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1].score == pytest.approx(1 / 61)
    assert "向量#2" in fused[0].why and "关键词#1" in fused[0].why
    assert "向量#1" in fused[1].why and "关键词未进榜" in fused[1].why


def test_rrf_prefers_items_found_by_both_methods() -> None:
    """两路都进榜的，要胜过只在单路拿第一的 —— 这是 RRF 的核心价值。"""
    both = _chunk("两边都出现", index=1)
    single = _chunk("只在一路由现", index=2)
    fused = Retriever.rrf_fuse(
        {
            "vector": [Hit(single, 0.99, 1, "v"), Hit(both, 0.50, 3, "v")],
            "keyword": [Hit(both, 9.9, 3, "k")],
        },
        k=5,
    )

    assert fused[0].chunk.chunk_id == both.chunk_id
    assert fused[0].score == pytest.approx(2 / 63)
    assert fused[1].score == pytest.approx(1 / 61)


def test_search_uses_a_deeper_pool_for_fusion(tmp_path: Path, monkeypatch) -> None:
    """融合要用更深的候选池，同时**对外仍只暴露 top-k**（三种方式才能公平比较）。"""
    chunks, config, cache = _build(tmp_path)
    retriever = Retriever(chunks, config, cache=cache)
    asked: dict[str, int] = {}
    real_vector, real_keyword = retriever.vector_search, retriever.keyword_search

    def spy_vector(query: str, k: int):
        asked["vector"] = k
        return real_vector(query, k)

    def spy_keyword(query: str, k: int):
        asked["keyword"] = k
        return real_keyword(query, k)

    monkeypatch.setattr(retriever, "vector_search", spy_vector)
    monkeypatch.setattr(retriever, "keyword_search", spy_keyword)

    result = retriever.search("毛利率", k=2, mode="all")

    assert asked == {"vector": 2 * RRF_POOL_FACTOR, "keyword": 2 * RRF_POOL_FACTOR}
    assert len(result.vector) <= 2 and len(result.keyword) <= 2 and len(result.hybrid) <= 2
    assert str(2 * RRF_POOL_FACTOR) in result.notes["hybrid"]


def test_rrf_respects_k_and_is_deterministic() -> None:
    chunks = [_chunk(f"片段{i}", index=i) for i in range(1, 6)]
    ranked = {"vector": [Hit(chunk, 1.0, rank, "v") for rank, chunk in enumerate(chunks, 1)]}
    first = Retriever.rrf_fuse(ranked, k=2)
    second = Retriever.rrf_fuse(ranked, k=2)

    assert len(first) == 2
    assert [hit.chunk.chunk_id for hit in first] == [hit.chunk.chunk_id for hit in second]


# ============================================================ 边界
def test_k_larger_than_corpus_is_safe(tmp_path: Path) -> None:
    chunks, config, cache = _build(tmp_path)
    retriever = Retriever(chunks, config, cache=cache)
    result = retriever.search("毛利率", k=99, mode="all")

    assert len(result.vector) == len(chunks)
    assert len(result.keyword) <= len(chunks)
    assert len(result.hybrid) <= len(chunks)


def test_punctuation_only_query_still_returns_vector_hits(tmp_path: Path) -> None:
    """纯标点：关键词路没有可用 gram（返回空），向量路仍然给结果 —— 这个差异要显式知道。"""
    chunks, config, cache = _build(tmp_path)
    retriever = Retriever(chunks, config, cache=cache)
    result = retriever.search("！？…", k=2, mode="all")

    assert result.keyword == []
    assert result.vector


# ============================================================ CLI
def test_cli_keyword_mode_is_offline(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    chunks, _, _ = _build(tmp_path)
    chunks_file = write_jsonl(chunks, tmp_path / "chunks.jsonl")

    code = main(
        [
            "SANDBOX_VIOLATION",
            "--mode",
            "keyword",
            "--chunks",
            str(chunks_file),
            "--cache",
            str(tmp_path / "emb.jsonl"),
        ]
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "== keyword ==" in out and "== vector ==" not in out
    assert "为什么:" in out


def test_cli_reports_missing_vectors(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    chunks, _, _ = _build(tmp_path)
    chunks_file = write_jsonl(chunks, tmp_path / "chunks.jsonl")

    code = main(
        [
            "毛利率",
            "--chunks",
            str(chunks_file),
            "--cache",
            str(tmp_path / "empty.jsonl"),
            "--provider",
            "fake",
        ]
    )

    assert code == 4
    assert "RETRIEVAL_NOT_READY" in capsys.readouterr().out


def test_cli_fake_provider_runs_vector_mode(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    chunks, _, _ = _build(tmp_path)
    chunks_file = write_jsonl(chunks, tmp_path / "chunks.jsonl")

    code = main(
        [
            "毛利率",
            "--chunks",
            str(chunks_file),
            "--cache",
            str(tmp_path / "emb.jsonl"),
            "--provider",
            "fake",
            "--k",
            "2",
        ]
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "== vector ==" in out and "== hybrid ==" in out
