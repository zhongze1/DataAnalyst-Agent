"""DA-018 / RAG-05：主工程运行期的知识库检索。

与 ``rag_lab`` 的边界是**数据文件**，不是代码：本模块只读 ``data/knowledge/`` 下的
工件（``knowledge_index.jsonl`` + ``manifest.json``），自己实现检索。原因：两个子工程
的虚拟环境与依赖完全分开，主工程必须能**独立部署**（不能要求部署机上有 lab 的 venv）。

检索公式与 RAG-04 **逐字一致**（否则评估过的 recall 就不再适用）：

| 方式 | 做法 |
| --- | --- |
| 关键词 | **字符 2-gram** + BM25（``k1=1.2`` / ``b=0.75``），中文不引 jieba，零新依赖 |
| 向量 | 查询向量与片段向量的余弦 Top-K，最低分阈值挡噪声 |
| 混合 | 两路在**每路前 4×k** 上 RRF 融合（``Σ 1/(60+rank)``），再截到 k |

每条结果都带 ``why`` —— "为什么召回"比"分数是多少"更重要。

**降级是一等公民**：向量服务没配、连不上、维度不符、工件缺向量、工件损坏……
全都不抛异常，而是退回关键词模式或返回"不可用"，并在 ``notes`` 里说清楚。
工具层"永不抛异常"的既有约定由这里先兜住。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import httpx

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

NGRAM = 2
# 与 rag_lab/retrieve.py 保持同值：文献默认，不做调参（调了就是在评估集上过拟合）
BM25_K1 = 1.2
BM25_B = 0.75
RRF_K0 = 60
# 融合候选池倍数：只在两路 top-k 上融合会把"单路排第 5 但另一路很强"的正确结果挤掉
RRF_POOL_FACTOR = 4
# 查询向量的进程内缓存上限（不落盘：查询是无限的，片段是有限的）
QUERY_CACHE_MAX = 256


class _EmbeddingFailed(RuntimeError):
    """查询向量化失败。**只在内部流转**，对外表现为降级为关键词模式。"""


@dataclass(frozen=True)
class KnowledgeHit:
    """一条召回结果。``chunk_id`` 用于融合去重（同一片段在两路里的身份）。"""

    chunk_id: str
    doc_id: str
    section: str
    line_start: int
    line_end: int
    text: str
    score: float
    why: str


@dataclass(frozen=True)
class KnowledgeSearchResult:
    """一次检索的完整结果。

    ``status``：``ok``（有命中）/ ``empty``（确实没有相关内容）/ ``unavailable``（工件不可用）。
    ``notes`` 是给模型看的模式与降级说明；``detail`` 是状态的具体原因。
    """

    query: str
    mode: str  # "hybrid" | "keyword"
    status: str  # "ok" | "empty" | "unavailable"
    hits: tuple[KnowledgeHit, ...] = ()
    notes: tuple[str, ...] = ()
    detail: str = ""
    total_chunks: int = 0


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
    """字符 n-gram。短于 n 的文本整体作为一个 gram。"""
    normalized = normalize(text)
    if not normalized:
        return []
    if len(normalized) <= n:
        return [normalized]
    return [normalized[i : i + n] for i in range(len(normalized) - n + 1)]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """余弦相似度。维度不一致按较短的算（正常不该发生，上游已拦）；零向量返回 0。

    绝不返回 NaN：NaN 参与比较会让排序结果不确定。
    """
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
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


class _BM25Index:
    """字符 2-gram 上的 BM25。

    为什么不是"数一数重叠了几个词"：片段长度差异很大（42~1051 字符），
    朴素重叠会让长片段天然占优；BM25 同时做了词频饱和与长度归一化，
    再叠上 IDF 压掉"的、是、在"这类高频 gram。参数用文献默认值，不调。
    """

    def __init__(self, texts: Sequence[str], *, n: int = NGRAM) -> None:
        self._k1 = BM25_K1
        self._b = BM25_B
        self._n = n
        self._postings: dict[str, dict[int, int]] = {}
        self._df: Counter[str] = Counter()
        self._lengths: list[int] = []

        for index, text in enumerate(texts):
            grams = char_ngrams(text, n)
            self._lengths.append(len(grams))
            for gram, frequency in Counter(grams).items():
                self._postings.setdefault(gram, {})[index] = frequency
                self._df[gram] += 1
        self._total = len(texts)
        self._average_length = (sum(self._lengths) / self._total) if self._total else 0.0

    def _idf(self, gram: str) -> float:
        df = self._df.get(gram, 0)
        return math.log(1 + (self._total - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int) -> list[tuple[int, float, list[str]]]:
        """返回 ``[(片段下标, 分数, 命中的前 5 个 gram)]``，按分数降序。"""
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
            for index, frequency in postings.items():
                length = self._lengths[index] or 1
                denominator = frequency + self._k1 * (
                    1 - self._b + self._b * length / (self._average_length or 1)
                )
                contribution = idf * frequency * (self._k1 + 1) / denominator
                scores[index] = scores.get(index, 0.0) + contribution
                matched.setdefault(index, []).append((contribution, gram))

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        return [
            (
                index,
                score,
                [gram for _, gram in sorted(matched[index], reverse=True)[:5]],
            )
            for index, score in ranked
        ]


# --------------------------------------------------------------------- 查询向量化
class QueryEmbedder(Protocol):
    """把查询文本转成向量。失败应抛 :class:`_EmbeddingFailed`（或任何异常，调用方兜住）。"""

    def embed(self, text: str) -> list[float]: ...


class HttpQueryEmbedder:
    """OpenAI 兼容 ``POST {base_url}/embeddings``。

    只发**一条**查询（不做批量）：查询是交互式的，批量化没有收益。
    任何传输/协议错误都转成 :class:`_EmbeddingFailed`，由服务层决定降级。
    """

    def __init__(self, settings: Settings, *, client: httpx.Client | None = None) -> None:
        self._settings = settings
        # 注入的 client 由调用方（测试）负责关闭；自建的一次性用完即关
        self._client = client

    def embed(self, text: str) -> list[float]:
        settings = self._settings
        url = f"{settings.embedding_base_url.rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {settings.embedding_api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": settings.embedding_model, "input": [text]}
        try:
            if self._client is not None:
                response = self._client.post(url, headers=headers, json=payload)
            else:
                with httpx.Client(
                    timeout=settings.embedding_timeout,
                    trust_env=settings.embedding_trust_env,
                ) as client:
                    response = client.post(url, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            # 只带异常类型：消息里可能含 URL/代理信息，没必要进工具输出
            raise _EmbeddingFailed(f"连接向量服务失败（{type(exc).__name__}）") from exc

        status = response.status_code
        if status in (401, 403):
            raise _EmbeddingFailed(f"向量服务拒绝鉴权（HTTP {status}）")
        if status != 200:
            raise _EmbeddingFailed(f"向量服务返回 HTTP {status}")
        try:
            data = response.json()
        except ValueError as exc:
            raise _EmbeddingFailed("向量服务返回的不是 JSON") from exc

        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list) or not items:
            raise _EmbeddingFailed("向量服务响应缺少 data 字段")
        first = sorted(
            (item for item in items if isinstance(item, dict)),
            key=lambda item: item.get("index", 0),
        )
        vector = first[0].get("embedding") if first else None
        if not isinstance(vector, list) or not vector:
            raise _EmbeddingFailed("向量服务响应缺少 embedding 字段")
        return [float(value) for value in vector]


# --------------------------------------------------------------------- 融合
def rrf_fuse(
    vector_hits: Sequence[KnowledgeHit],
    keyword_hits: Sequence[KnowledgeHit],
    k: int,
    *,
    k0: int = RRF_K0,
) -> list[KnowledgeHit]:
    """RRF 融合：``Σ 1/(k0 + rank)``，只用**排名**不用分数。

    余弦与 BM25 的量纲不可比，强行加权求和就得为语料调权重 —— 那是过拟合。
    """
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    hit_of: dict[str, KnowledgeHit] = {}
    for name, hits in (("vector", vector_hits), ("keyword", keyword_hits)):
        for rank, hit in enumerate(hits, start=1):
            key = hit.chunk_id
            hit_of.setdefault(key, hit)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k0 + rank)
            ranks.setdefault(key, {})[name] = rank

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
    fused: list[KnowledgeHit] = []
    for key, score in ordered:
        detail = ranks[key]
        parts = [
            f"向量#{detail['vector']}" if "vector" in detail else "向量未进榜",
            f"关键词#{detail['keyword']}" if "keyword" in detail else "关键词未进榜",
        ]
        base = hit_of[key]
        fused.append(
            KnowledgeHit(
                chunk_id=base.chunk_id,
                doc_id=base.doc_id,
                section=base.section,
                line_start=base.line_start,
                line_end=base.line_end,
                text=base.text,
                score=score,
                why=f"{' + '.join(parts)} → RRF {score:.4f}",
            )
        )
    return fused


# --------------------------------------------------------------------- 服务
def _hit_of(row: dict[str, object], index: int, score: float, why: str) -> KnowledgeHit:
    """把一行索引转成结果。字段缺失时退化为空值，不抛异常。"""
    return KnowledgeHit(
        chunk_id=str(row.get("chunk_id") or f"#{index}"),
        doc_id=str(row.get("doc_id") or row.get("source_path") or "(未知文档)"),
        section=str(row.get("section") or "(无标题)"),
        line_start=int(row.get("line_start") or 0),
        line_end=int(row.get("line_end") or 0),
        text=str(row.get("text") or ""),
        score=score,
        why=why,
    )


def _vector_of(row: dict[str, object]) -> list[float]:
    vector = row.get("vector")
    return [float(value) for value in vector] if isinstance(vector, list) else []


class KnowledgeService:
    """Runtime 检索服务：加载工件 + 三种检索，**任何失败都不抛异常**。"""

    def __init__(
        self,
        settings: Settings,
        *,
        embedder: QueryEmbedder | None = None,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._rows: list[dict[str, object]] = []
        self._load_error = ""
        self._missing_vectors = 0
        self._dim = 0
        self._bm25: _BM25Index | None = None
        self._query_cache: dict[str, list[float]] = {}
        self._degrade_note = ""
        self._load()

    # ----------------------------------------------------------------- 加载
    def _load(self) -> None:
        index_path = Path(self._settings.knowledge_index_path)
        manifest_path = (
            Path(self._settings.knowledge_manifest_path)
            if self._settings.knowledge_manifest_path
            else index_path.with_name("manifest.json")
        )
        try:
            if not index_path.exists():
                self._load_error = (
                    f"知识库未构建：找不到 {index_path}。"
                    "请先跑一次导出（uv run python -m rag_lab.export_index）"
                )
                return
            rows: list[dict[str, object]] = []
            for number, line in enumerate(
                index_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or "text" not in row:
                    raise ValueError(f"第 {number} 行不是合法的片段对象")
                rows.append(row)
        except (OSError, ValueError) as exc:
            self._rows = []
            self._load_error = (
                f"知识库不可用（{type(exc).__name__}: {exc}）。"
                "工件可能损坏或未导出完成，请重跑 rag_lab 的导出"
            )
            return

        if not rows:
            self._load_error = (
                f"知识库为空（0 个片段）：{index_path}。请重新导出，否则检索恒无结果"
            )
            return

        self._rows = rows
        manifest: object = None
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                # manifest 只用于核对维度/来源，坏了不影响检索，不去打断它
                logger.warning("知识库 manifest 不可读，改用工件自身推断: %s", manifest_path)

        declared = manifest.get("dim") if isinstance(manifest, dict) else None
        dimensions = {len(vector) for vector in (_vector_of(row) for row in rows) if vector}
        self._missing_vectors = sum(1 for row in rows if not _vector_of(row))
        if isinstance(declared, int) and declared > 0:
            self._dim = declared
        elif len(dimensions) == 1:
            self._dim = next(iter(dimensions))
        else:
            self._dim = 0
        logger.info(
            "知识库已加载 | 片段 %d | 维度 %d | 缺向量 %d",
            len(rows),
            self._dim,
            self._missing_vectors,
        )

    # ----------------------------------------------------------------- 状态
    @property
    def total_chunks(self) -> int:
        return len(self._rows)

    @property
    def mode(self) -> str:
        """当前可用的检索模式。配齐向量且工件齐备才是 hybrid。"""
        ready, _ = self._vector_status()
        return "hybrid" if ready else "keyword"

    def _vector_status(self) -> tuple[bool, str]:
        """返回 (向量能否用, 不能用时的说明)。"""
        settings = self._settings
        if not (
            settings.embedding_base_url.strip()
            and settings.embedding_api_key.strip()
            and settings.embedding_model.strip()
        ):
            return False, (
                "未配置 AGENT_EMBEDDING_BASE_URL / API_KEY / MODEL，"
                "当前按**关键词模式**检索（效果略低于混合）"
            )
        if self._missing_vectors:
            return False, (
                f"有 {self._missing_vectors}/{len(self._rows)} 个片段缺向量，"
                "当前按**关键词模式**检索"
            )
        if self._dim <= 0 or any(len(_vector_of(row)) != self._dim for row in self._rows):
            return False, "工件内向量维度不一致，当前按**关键词模式**检索"
        return True, ""

    @property
    def _bm25_index(self) -> _BM25Index:
        if self._bm25 is None:
            self._bm25 = _BM25Index([str(row.get("text") or "") for row in self._rows])
        return self._bm25

    def _clamp_top_k(self, top_k: int | None) -> int:
        settings = self._settings
        if top_k is None:
            return max(1, settings.knowledge_top_k)
        # 工具层只校验类型不校验范围，越界在这里收敛而不是报错：
        # 模型偶尔写 top_k=0 时，给一个可用结果比给一条错误更有用
        return max(1, min(int(top_k), max(1, settings.knowledge_top_k_max)))

    # ----------------------------------------------------------------- 检索
    def _embed_query(self, query: str) -> list[float] | None:
        """取（或算并缓存）查询向量；失败/维度不符 → 记下降级原因并返回 None。"""
        key = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
        cached = self._query_cache.get(key)
        if cached is not None:
            return cached

        embedder = self._embedder
        if embedder is None:
            embedder = HttpQueryEmbedder(self._settings)
            self._embedder = embedder
        try:
            vector = embedder.embed(query)
        except _EmbeddingFailed as exc:
            self._degrade_note = f"向量检索不可用（{exc}），已降级为**关键词模式**"
            logger.warning("查询向量化失败，降级关键词模式: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001 - 注入的 embedder 可能抛任何东西
            self._degrade_note = f"向量检索异常（{type(exc).__name__}），已降级为**关键词模式**"
            logger.warning("查询向量化异常，降级关键词模式", exc_info=True)
            return None

        if len(vector) != self._dim:
            # 绝不拿不同模型的向量硬比：那会给出看起来合理、实则无意义的相似度
            self._degrade_note = (
                f"查询向量维度 {len(vector)} 与工件 {self._dim} 不一致，"
                "已降级为**关键词模式**（不做向量比较）"
            )
            logger.warning("查询向量维度不符：%d vs %d", len(vector), self._dim)
            return None

        if len(self._query_cache) >= QUERY_CACHE_MAX:
            self._query_cache.clear()
        self._query_cache[key] = vector
        return vector

    def _keyword_search(self, query: str, candidates: int) -> list[KnowledgeHit]:
        threshold = self._settings.knowledge_min_keyword_score
        hits: list[KnowledgeHit] = []
        for index, score, grams in self._bm25_index.search(query, candidates):
            if score < threshold:
                continue
            hit = _hit_of(
                self._rows[index],
                index,
                score,
                f"命中 2-gram: {'、'.join(grams)}（BM25 {score:.2f}）",
            )
            hits.append(hit)
        return hits

    def _vector_search(self, query_vector: list[float], candidates: int) -> list[KnowledgeHit]:
        threshold = self._settings.knowledge_min_vector_score
        scored = [
            (cosine(query_vector, _vector_of(row)), index)
            for index, row in enumerate(self._rows)
        ]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        hits: list[KnowledgeHit] = []
        for score, index in scored:
            if score < threshold:
                continue
            hits.append(_hit_of(self._rows[index], index, score, f"语义相似度 {score:.4f}"))
            if len(hits) >= candidates:
                break
        return hits

    def search(self, query: str, *, top_k: int | None = None) -> KnowledgeSearchResult:
        """检索 ``query``，返回带来源与 why 的结果。**永不抛异常**。"""
        if self._load_error:
            return KnowledgeSearchResult(
                query=query, mode="keyword", status="unavailable", detail=self._load_error
            )

        limit = self._clamp_top_k(top_k)
        if not char_ngrams(query):
            # 纯标点/空白：关键词必然无果，也别白花一次向量调用
            return KnowledgeSearchResult(
                query=query,
                mode=self.mode,
                status="empty",
                detail="查询里没有可检索的字符（只含标点或空白），未调用向量服务",
                total_chunks=len(self._rows),
            )

        vector_ready, vector_note = self._vector_status()
        query_vector = self._embed_query(query) if vector_ready else None

        if query_vector is not None:
            # 融合的两路要取更深：只在 top-k 上融合会把深度候选挤掉
            deep = limit * RRF_POOL_FACTOR
            vector_hits = self._vector_search(query_vector, deep)
            keyword_hits = self._keyword_search(query, deep)
            hits = rrf_fuse(vector_hits, keyword_hits, limit)
            mode = "hybrid"
            notes = (
                f"混合 = 向量(余弦 ≥{self._settings.knowledge_min_vector_score:g})"
                f" + 关键词(BM25 ≥{self._settings.knowledge_min_keyword_score:g})，"
                f"RRF(k0={RRF_K0}) 融合后取前 {limit}"
                f"（候选 向量 {len(vector_hits)} / 关键词 {len(keyword_hits)}）",
                "两路分数量纲不可比，融合只看名次、不调权重",
            )
        else:
            hits = self._keyword_search(query, limit)
            mode = "keyword"
            notes = (
                self._degrade_note or vector_note,
                f"关键词 = 字符 2-gram BM25，最低分 {self._settings.knowledge_min_keyword_score:g}",
            )

        if not hits:
            return KnowledgeSearchResult(
                query=query,
                mode=mode,
                status="empty",
                detail=(
                    "候选片段都没有达到最低分阈值"
                    f"（余弦 ≥{self._settings.knowledge_min_vector_score:g}、"
                    f"BM25 ≥{self._settings.knowledge_min_keyword_score:g}）"
                ),
                notes=notes,
                total_chunks=len(self._rows),
            )

        return KnowledgeSearchResult(
            query=query,
            mode=mode,
            status="ok",
            hits=tuple(hits),
            notes=notes,
            total_chunks=len(self._rows),
        )


@lru_cache(maxsize=1)
def get_knowledge_service() -> KnowledgeService:
    """进程级单例：73×1024 维工件解析一次就够，没必要每个请求重读。

    测试必须 ``cache_clear()`` 隔离（与 ``get_quota_service`` 同理），
    否则会把第一次用到的 settings 一直留着。
    """
    return KnowledgeService(get_settings())
