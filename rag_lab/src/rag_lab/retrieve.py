"""RAG-04：三种检索方式，**各自独立保留结果**。

刻意不做成"一个融合黑盒"，因为现在的目标是**比较**，不是调优：

| 方式 | 做法 | 优点 | 已知弱点 |
| --- | --- | --- | --- |
| ``vector`` | 查询向量与片段向量的余弦 Top-K（复用 RAG-03 的缓存） | 能处理同义改写 | 对专有名词/编号不敏感 |
| ``keyword`` | **字符 2-gram** + BM25 Top-K（中文不引 jieba，零新依赖） | 专有名词、数字、代码标识符命中准 | 不懂同义改写 |
| ``hybrid`` | 两路结果的 **RRF** 融合（``Σ 1/(60+rank)``） | 无参数可调、不易过拟合 | 只是排序融合，不产生新召回 |

每个 :class:`Hit` 都带 ``why`` —— **"为什么召回"比"分数是多少"更重要**：
向量给出相似度，关键词给出命中的 2-gram，混合给出它分别排在第几名。

查询向量也进缓存（单独空间 ``<space>|query``，按文本哈希），所以评估脚本重复跑不重复计费。
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from rag_lab.chunker import Chunk
from rag_lab.embedding import (
    DEFAULT_CACHE_FILE,
    DEFAULT_CHUNKS_FILE,
    EmbeddingCache,
    EmbeddingConfig,
    EmbeddingError,
    Embedder,
    build_embedder,
    config_from_env,
    load_chunks,
)

logger = logging.getLogger(__name__)

NGRAM = 2
# RRF 的平滑常数：60 是文献里的常用值，这里**不做调参**（调了就等于在评估集上过拟合）
RRF_K0 = 60
# 融合时的候选池倍数：只在两路的 top-k 上融合，会把"单路排第 5 但另一路很强"的
# 正确结果挤出混合前 k（评估里真踩到了）。所以在 **k 的若干倍** 深度上融合，
# 最后再截到 k —— 这不是调参，是让 RRF 有足够的候选可融合。
RRF_POOL_FACTOR = 4


@dataclass(frozen=True)
class Hit:
    """一条召回结果：片段 + 分数 + 排名 + **为什么召回**。"""

    chunk: Chunk
    score: float
    rank: int
    why: str

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id

    @property
    def section(self) -> str:
        return self.chunk.section


@dataclass
class RetrievalResult:
    """一次查询的完整结果：三种方式**并列保留**，互不覆盖。"""

    query: str
    space: str
    k: int
    vector: list[Hit] = field(default_factory=list)
    keyword: list[Hit] = field(default_factory=list)
    hybrid: list[Hit] = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)

    def mode(self, name: str) -> list[Hit]:
        return {"vector": self.vector, "keyword": self.keyword, "hybrid": self.hybrid}[name]


# --------------------------------------------------------------------- 文本处理
def normalize(text: str) -> str:
    """检索用的归一化文本：转小写、只保留中日韩字符与字母数字。

    标点与空白一律丢弃 —— 它们参与 2-gram 只会产生噪声片段（如「，」+「本」）。
    """
    return "".join(
        char.lower()
        for char in text
        if char.isalnum() or "\u4e00" <= char <= "\u9fff" or "\u3400" <= char <= "\u4dbf"
    )


def char_ngrams(text: str, n: int = NGRAM) -> list[str]:
    """字符 n-gram（默认 2-gram）。短于 n 的文本整体作为一个 gram。"""
    normalized = normalize(text)
    if not normalized:
        return []
    if len(normalized) <= n:
        return [normalized]
    return [normalized[i : i + n] for i in range(len(normalized) - n + 1)]


# --------------------------------------------------------------------- 关键词
class BM25Index:
    """字符 2-gram 上的 BM25。

    为什么不是"数一数重叠了几个词"：片段长度差异很大（42~1051 字符），
    朴素重叠会让长片段天然占优；BM25 同时做了词频饱和与长度归一化，
    再叠上 IDF 压掉"的、是、在"这类高频 gram。参数用文献默认值，不调。
    """

    def __init__(
        self, chunks: Sequence[Chunk], *, n: int = NGRAM, k1: float = 1.2, b: float = 0.75
    ) -> None:
        self._k1 = k1
        self._b = b
        self._n = n
        self._doc_ids: list[str] = [chunk.chunk_id for chunk in chunks]
        self._postings: dict[str, dict[int, int]] = {}
        self._df: Counter[str] = Counter()
        self._lengths: list[int] = []

        for index, chunk in enumerate(chunks):
            grams = char_ngrams(chunk.text, n)
            self._lengths.append(len(grams))
            for gram, tf in Counter(grams).items():
                self._postings.setdefault(gram, {})[index] = tf
                self._df[gram] += 1
        self._average_length = (sum(self._lengths) / len(self._lengths)) if chunks else 0.0

    def __len__(self) -> int:
        return len(self._doc_ids)

    @property
    def vocabulary_size(self) -> int:
        return len(self._postings)

    def _idf(self, gram: str) -> float:
        total = len(self._doc_ids)
        df = self._df.get(gram, 0)
        return math.log(1 + (total - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int) -> list[tuple[int, float, list[str]]]:
        """返回 ``[(片段下标, BM25 分数, 命中的关键 gram), ...]``，按分数降序。"""
        query_grams = set(char_ngrams(query, self._n))
        if not query_grams:
            return []

        scores: dict[int, float] = {}
        matched: dict[int, list[tuple[float, str]]] = {}
        for gram in query_grams:
            postings = self._postings.get(gram)
            if not postings:
                continue
            idf = self._idf(gram)
            for index, tf in postings.items():
                length = self._lengths[index] or 1
                denominator = tf + self._k1 * (
                    1 - self._b + self._b * length / (self._average_length or 1)
                )
                contribution = idf * tf * (self._k1 + 1) / denominator
                scores[index] = scores.get(index, 0.0) + contribution
                matched.setdefault(index, []).append((contribution, gram))

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            (
                index,
                score,
                [gram for _, gram in sorted(matched[index], reverse=True)[:5]],
            )
            for index, score in ranked
        ]


# --------------------------------------------------------------------- 向量
def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """余弦相似度。维度不一致时按较短的算并告警（正常不该发生）。"""
    if len(left) != len(right):
        logger.warning("向量维度不一致（%d vs %d），按较短的比较", len(left), len(right))
    size = min(len(left), len(right))
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for index in range(size):
        a = left[index]
        b = right[index]
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


def query_cache_key(query: str) -> str:
    """查询向量在缓存里的键（与片段用同一套哈希，靠空间区分）。"""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]


def embed_query(query: str, embedder: Embedder, cache: EmbeddingCache, space: str) -> list[float]:
    """取（或算并缓存）查询向量。查询空间独立，不和片段向量混在一起。"""
    key = query_cache_key(query)
    cached = cache.get(space, key)
    if cached is not None:
        return cached
    vector = embedder.embed([query])[0]
    cache.put(space, key, vector)
    return vector


# --------------------------------------------------------------------- 检索器
class Retriever:
    """把三种检索方式绑在一起；**三种结果各自独立返回，不做隐式融合**。"""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        config: EmbeddingConfig,
        *,
        embedder: Embedder | None = None,
        cache: EmbeddingCache | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.config = config
        self.space = config.space
        # 必须显式判 None：EmbeddingCache 定义了 __len__，
        # 空缓存是 falsy —— 写成 `cache or EmbeddingCache(...)` 会把传进来的空缓存
        # 静默换成配置里的路径（踩过：测试里传空缓存，却读到了另一份有向量的缓存）
        self.cache = cache if cache is not None else EmbeddingCache(config.cache_path)
        self._embedder = embedder
        self._bm25: BM25Index | None = None
        self._vectors: dict[str, list[float]] = {}
        self._missing_vectors = 0
        self._load_vectors()

    # ----------------------------------------------------------------- 准备
    def _load_vectors(self) -> None:
        for chunk in self.chunks:
            vector = self.cache.get(self.space, chunk.content_hash)
            if vector is None:
                self._missing_vectors += 1
            else:
                self._vectors[chunk.content_hash] = vector

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = build_embedder(self.config)
        return self._embedder

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index(self.chunks)
        return self._bm25

    @property
    def vector_coverage(self) -> str:
        total = len(self.chunks)
        ready = total - self._missing_vectors
        return f"{ready}/{total} 个片段已有向量"

    # ----------------------------------------------------------------- 三种检索
    def vector_search(self, query: str, k: int) -> list[Hit]:
        if self._missing_vectors:
            raise ValueError(
                f"还有 {self._missing_vectors} 个片段没有向量，先跑 "
                "uv run python -m rag_lab.embedding"
            )
        query_vector = embed_query(query, self.embedder, self.cache, f"{self.space}|query")
        scored = [
            (cosine(query_vector, self._vectors[chunk.content_hash]), index)
            for index, chunk in enumerate(self.chunks)
        ]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [
            Hit(
                chunk=self.chunks[index],
                score=score,
                rank=rank,
                why=f"语义相似度 {score:.4f}",
            )
            for rank, (score, index) in enumerate(scored[:k], start=1)
        ]

    def keyword_search(self, query: str, k: int) -> list[Hit]:
        found = self.bm25.search(query, k)
        hits: list[Hit] = []
        for rank, (index, score, grams) in enumerate(found, start=1):
            hits.append(
                Hit(
                    chunk=self.chunks[index],
                    score=score,
                    rank=rank,
                    why=f"命中 2-gram: {'、'.join(grams)}（BM25 {score:.2f}）",
                )
            )
        return hits

    @staticmethod
    def rrf_fuse(
        ranked_lists: dict[str, list[Hit]], k: int, *, k0: int = RRF_K0
    ) -> list[Hit]:
        """RRF 融合：``Σ 1/(k0 + rank)``。

        只用**排名**、不用分数 —— 余弦与 BM25 的量纲不可比，
        强行加权求和就得为语料调权重，那是过拟合。
        """
        scores: dict[str, float] = {}
        ranks: dict[str, dict[str, int]] = {}
        chunk_of: dict[str, Chunk] = {}
        for name, hits in ranked_lists.items():
            for hit in hits:
                key = hit.chunk.chunk_id
                chunk_of[key] = chunk_of.get(key, hit.chunk)
                scores[key] = scores.get(key, 0.0) + 1.0 / (k0 + hit.rank)
                ranks.setdefault(key, {})[name] = hit.rank

        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
        fused: list[Hit] = []
        for rank, (key, score) in enumerate(ordered, start=1):
            detail = ranks[key]
            parts = [
                f"向量#{detail['vector']}" if "vector" in detail else "向量未进榜",
                f"关键词#{detail['keyword']}" if "keyword" in detail else "关键词未进榜",
            ]
            fused.append(
                Hit(
                    chunk=chunk_of[key],
                    score=score,
                    rank=rank,
                    why=f"{' + '.join(parts)} → RRF {score:.4f}",
                )
            )
        return fused

    def search(self, query: str, *, k: int = 5, mode: str = "all") -> RetrievalResult:
        """按 ``mode`` 取结果；``all`` 会同时给出三种（各自独立）。"""
        result = RetrievalResult(query=query, space=self.space, k=k)
        need_vector = mode in {"all", "vector", "hybrid"}
        need_keyword = mode in {"all", "keyword", "hybrid"}
        # 融合参与的两路要取更深：只在 top-k 上融合会把深度候选挤掉（见 RRF_POOL_FACTOR）
        pool = k * RRF_POOL_FACTOR if mode in {"all", "hybrid"} else k

        deep_vector = self.vector_search(query, pool) if need_vector else []
        deep_keyword = self.keyword_search(query, pool) if need_keyword else []

        # 对外仍然只暴露 top-k，便于三种方式公平比较
        if mode in {"all", "vector"}:
            result.vector = deep_vector[:k]
            result.notes["vector"] = f"{self.vector_coverage}；候选 {len(self.chunks)} 条"
        if mode in {"all", "keyword"}:
            result.keyword = deep_keyword[:k]
            result.notes["keyword"] = (
                f"2-gram 词表 {self.bm25.vocabulary_size} 个，片段 {len(self.chunks)} 条"
            )
        if mode in {"all", "hybrid"}:
            result.hybrid = self.rrf_fuse(
                {"vector": deep_vector, "keyword": deep_keyword}, k
            )
            result.notes["hybrid"] = (
                f"RRF k0={RRF_K0}；在每路前 {pool} 条上融合，再截到前 {k}"
            )
        return result


# --------------------------------------------------------------------- 输出
def format_hits(hits: Iterable[Hit], *, limit: int = 60, show_text: bool = False) -> str:
    """把一组 Hit 打印成人能读的块。"""
    lines: list[str] = []
    for hit in hits:
        head = hit.chunk.text.splitlines()[0][:limit] if hit.chunk.text else ""
        lines.append(
            f"  #{hit.rank}  {hit.score:.4f}  [{hit.doc_id}] {hit.section}\n"
            f"       行 {hit.chunk.line_start}-{hit.chunk.line_end} | {head}\n"
            f"       为什么: {hit.why}"
        )
        if show_text:
            body = "\n".join(f"         {line}" for line in hit.chunk.text.splitlines())
            lines.append(body)
    return "\n".join(lines) if lines else "  （无结果）"


def format_result(result: RetrievalResult, *, show_text: bool = False) -> str:
    blocks = [f"查询: {result.query}", f"空间: {result.space} | k={result.k}"]
    for name in ("vector", "keyword", "hybrid"):
        hits = result.mode(name)
        if not hits and name not in result.notes:
            continue
        blocks.append(f"\n== {name} ==  {result.notes.get(name, '')}")
        blocks.append(format_hits(hits, show_text=show_text))
    return "\n".join(blocks)


# --------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    """命令行入口：``uv run python -m rag_lab.retrieve "你的问题"``。"""
    parser = argparse.ArgumentParser(
        prog="python -m rag_lab.retrieve",
        description="三种检索各自独立输出（RAG-04，只检索不生成）",
    )
    parser.add_argument("query", nargs="+", help="查询文本（多个词会拼成一句）")
    parser.add_argument("--k", type=int, default=5, help="每种方式取前 K 条")
    parser.add_argument(
        "--mode",
        choices=["all", "vector", "keyword", "hybrid"],
        default="all",
        help="只跑某一种方式（keyword 完全离线，不需要向量服务）",
    )
    parser.add_argument(
        "--provider",
        choices=["fake", "api"],
        help="覆盖 .env 里的 provider；keyword 模式根本用不到向量",
    )
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_FILE)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_FILE)
    parser.add_argument("--show-text", action="store_true", help="把召回片段的正文也打出来")
    args = parser.parse_args(argv)

    query = " ".join(args.query)
    try:
        config = replace(
            config_from_env(), chunks_path=args.chunks, cache_path=args.cache
        )
        if args.provider:
            config = replace(config, provider=args.provider)
        chunks = load_chunks(config.chunks_path)
        retriever = Retriever(chunks, config)
        result = retriever.search(query, k=args.k, mode=args.mode)
    except EmbeddingError as exc:
        print(f"[{exc.code}] {exc}", flush=True)
        return 3
    except ValueError as exc:  # 向量没备齐
        print(f"[RETRIEVAL_NOT_READY] {exc}", flush=True)
        return 4

    print(format_result(result, show_text=args.show_text))
    return 0


if __name__ == "__main__":  # pragma: no cover - 由 CLI 测试覆盖 main()
    sys.exit(main())
