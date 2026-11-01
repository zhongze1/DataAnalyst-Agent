"""DA-018：主工程运行期的知识库检索（全程离线）。

工件**现造**在 tmp_path：不依赖仓库里被 gitignore 的 ``data/knowledge/``，
也不发任何真实网络请求（httpx 的传输层被夹具挡住）。

覆盖：纯函数数学（归一化 / 2-gram / 余弦 / RRF）、四种降级路径
（未配向量 / 查询维度不符 / 向量服务失败 / 工件缺向量）、
工件缺失与损坏、最低分阈值与空结果、top_k 收敛，以及真实工件的契约。
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.config import BASE_DIR, Settings
from app.services import knowledge_service as ks
from app.services.knowledge_service import (
    KnowledgeHit,
    KnowledgeService,
    char_ngrams,
    cosine,
    get_knowledge_service,
    normalize,
    rrf_fuse,
)
from stubs import (
    KNOWLEDGE_CHUNKS,
    FakeQueryEmbedder,
    knowledge_row,
    write_knowledge_artifacts,
)

# ----------------------------------------------------------------- 夹具
@pytest.fixture(autouse=True)
def forbid_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """验收 17：任何用例真去联网都必须立刻炸出来。

    只放行**显式带了 transport** 的客户端（MockTransport 走这条），
    其余一律拒绝构造。
    """
    real_client = httpx.Client

    def guarded(*args: object, **kwargs: object) -> httpx.Client:
        if "transport" not in kwargs:
            raise AssertionError("测试禁止构造会联网的 httpx.Client")
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", guarded)


@pytest.fixture
def kb_dir(tmp_path: Path) -> Path:
    return tmp_path / "knowledge"


@pytest.fixture
def kb_settings(tmp_path: Path, kb_dir: Path) -> Settings:
    return Settings(
        _env_file=None,
        llm_provider="mock",
        upload_dir=tmp_path / "uploads",
        knowledge_index_path=kb_dir / "knowledge_index.jsonl",
    )


@pytest.fixture
def kb_index(kb_dir: Path) -> Path:
    """一份合成索引（3 片，4 维正交向量）。"""
    return write_knowledge_artifacts(kb_dir, KNOWLEDGE_CHUNKS)


def _hybrid(settings: Settings) -> Settings:
    """把 settings 变成"配齐了向量服务"的样子。"""
    return settings.model_copy(
        update={
            "embedding_base_url": "https://vector.example/v1",
            "embedding_api_key": "test-key",
            "embedding_model": "test-embedding",
        }
    )


# ============================================================ 纯函数数学
def test_normalize_keeps_only_cjk_and_alnum() -> None:
    assert normalize("单价 = 12,345 元/单！") == "单价12345元单"


def test_char_ngrams_handles_short_and_empty_text() -> None:
    assert char_ngrams("打码") == ["打码"]  # 短于 n：整体一个 gram
    assert char_ngrams("码") == ["码"]
    assert char_ngrams("！！！") == []  # 全是标点：没有任何可检索的字符
    assert char_ngrams("敏感字段打码")[:2] == ["敏感", "感字"]


def test_cosine_is_zero_for_zero_vector_and_never_nan() -> None:
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    # 维度不一致按较短的算（正常不该发生，但绝不能抛异常或产出 NaN）
    assert cosine([1.0, 0.0, 9.0], [1.0, 0.0]) == 1.0


def test_rrf_fuse_uses_ranks_not_scores() -> None:
    """余弦与 BM25 量纲不可比，融合只看名次：两路都进榜的排最前。"""
    vector = [_hit("c2"), _hit("c1")]
    keyword = [_hit("c1"), _hit("c3")]
    fused = rrf_fuse(vector, keyword, 3)
    assert [hit.chunk_id for hit in fused] == ["c1", "c2", "c3"]
    assert fused[0].why.startswith("向量#2 + 关键词#1")
    assert "关键词未进榜" in fused[1].why


def _hit(chunk_id: str) -> KnowledgeHit:
    return KnowledgeHit(
        chunk_id=chunk_id,
        doc_id=chunk_id,
        section="s",
        line_start=1,
        line_end=2,
        text="t",
        score=0.5,
        why="",
    )


# ============================================================ 关键词模式
def test_keyword_mode_returns_source_and_why(kb_settings: Settings, kb_index: Path) -> None:
    service = KnowledgeService(kb_settings)

    assert service.mode == "keyword"
    assert service.total_chunks == 3

    result = service.search("敏感字段怎么打码")
    assert result.status == "ok"
    assert result.mode == "keyword"
    assert result.hits[0].doc_id == "DA-015-敏感字段打码"
    assert result.hits[0].section == "打码规则"
    assert result.hits[0].line_start == 1 and result.hits[0].line_end == 3
    assert "2-gram" in result.hits[0].why and "BM25" in result.hits[0].why
    assert any("关键词模式" in note for note in result.notes)


def test_unrelated_query_returns_empty(kb_settings: Settings, kb_index: Path) -> None:
    """完全无关的提问不给低分噪声（验收 8）。"""
    result = KnowledgeService(kb_settings).search("推荐几部好看的电影")
    assert result.status == "empty"
    assert result.hits == ()
    assert any("关键词模式" in note for note in result.notes)


def test_low_score_noise_is_filtered_by_threshold(kb_settings: Settings, kb_dir: Path) -> None:
    """只在"所有片段都有的高频 gram"上重合 → 分数低于阈值 → 不返回。

    没有阈值时这类查询会稳定返回一堆无语义的片段（长这样：`数据` 命中 3/3）。
    """
    rows = [
        knowledge_row("a", "d1", "s", "这些数据说明问题", [1.0, 0.0, 0.0, 0.0]),
        knowledge_row("b", "d2", "s", "另一份数据无关", [0.0, 1.0, 0.0, 0.0]),
        knowledge_row("c", "d3", "s", "数据第三份", [0.0, 0.0, 1.0, 0.0]),
    ]
    write_knowledge_artifacts(kb_dir, rows)

    assert KnowledgeService(kb_settings).search("数据").status == "empty"

    # 把阈值放开，同样的查询立刻有结果 —— 证明是阈值在起作用，而不是索引空
    loose = kb_settings.model_copy(update={"knowledge_min_keyword_score": 0.0})
    assert KnowledgeService(loose).search("数据").status == "ok"


def test_top_k_zero_is_clamped_not_rejected(kb_settings: Settings, kb_index: Path) -> None:
    """工具层只校验类型不校验范围；越界的 top_k 在这里收敛到 [1, 上限]。"""
    result = KnowledgeService(kb_settings).search("敏感字段怎么打码", top_k=0)
    assert result.status == "ok"
    assert len(result.hits) == 1

    many = KnowledgeService(kb_settings).search("敏感字段怎么打码", top_k=999)
    assert len(many.hits) <= 999


def test_pure_punctuation_query_skips_embedding(kb_settings: Settings, kb_index: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeQueryEmbedder([1.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(ks, "HttpQueryEmbedder", lambda settings: fake)

    result = KnowledgeService(_hybrid(kb_settings)).search("！！！？？？")
    assert result.status == "empty"
    assert fake.calls == 0, "纯标点查询不该白花一次向量调用"


# ============================================================ 混合模式与降级
def test_hybrid_mode_recovers_what_keyword_misses(kb_settings: Settings, kb_index: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeQueryEmbedder([1.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(ks, "HttpQueryEmbedder", lambda settings: fake)
    service = KnowledgeService(_hybrid(kb_settings))

    assert service.mode == "hybrid"
    result = service.search("红烧肉做法")  # 与语料没有任何 2-gram 重合
    assert result.status == "ok"
    assert result.mode == "hybrid"
    assert fake.calls == 1
    assert result.hits[0].doc_id == "DA-015-敏感字段打码"
    assert "向量#1" in result.hits[0].why and "关键词未进榜" in result.hits[0].why
    assert any("混合" in note for note in result.notes)


def test_query_vector_dim_mismatch_degrades_to_keyword(kb_settings: Settings, kb_index: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """绝不拿不同模型的向量硬比（验收 6）。"""
    fake = FakeQueryEmbedder([0.1, 0.2])  # 2 维 ≠ 工件的 4 维
    monkeypatch.setattr(ks, "HttpQueryEmbedder", lambda settings: fake)

    result = KnowledgeService(_hybrid(kb_settings)).search("敏感字段怎么打码")
    assert result.mode == "keyword"
    assert result.status == "ok" and result.hits  # 降级不影响回答
    assert any("维度" in note and "2" in note and "4" in note for note in result.notes)


@pytest.mark.parametrize(
    "failure",
    [ks._EmbeddingFailed("连接向量服务失败（ConnectError）"), RuntimeError("boom")],
)
def test_vector_failure_degrades_and_never_raises(
    kb_settings: Settings, kb_index: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    class Boom:
        def embed(self, text: str) -> list[float]:
            raise failure

    monkeypatch.setattr(ks, "HttpQueryEmbedder", lambda settings: Boom())

    result = KnowledgeService(_hybrid(kb_settings)).search("敏感字段怎么打码")
    assert result.mode == "keyword"
    assert result.status == "ok" and result.hits
    assert any("降级为" in note for note in result.notes)


def test_chunk_without_vector_forces_keyword_mode(kb_settings: Settings, kb_dir: Path) -> None:
    rows = [
        knowledge_row("c1", "d1", "s", "敏感字段打码", None),  # 缺向量
        knowledge_row("c2", "d2", "s", "客单价口径", [1.0, 0.0, 0.0, 0.0]),
    ]
    write_knowledge_artifacts(kb_dir, rows)

    service = KnowledgeService(_hybrid(kb_settings))
    assert service.mode == "keyword"
    assert any("缺向量" in note for note in service.search("敏感字段打码").notes)


# ============================================================ 工件异常
def test_missing_artifact_is_reported_not_raised(kb_settings: Settings) -> None:
    result = KnowledgeService(kb_settings).search("任意问题")
    assert result.status == "unavailable"
    assert "未构建" in result.detail
    assert "rag_lab.export_index" in result.detail  # 必须给出可执行的修复路径
    assert result.hits == ()


def test_corrupt_artifact_is_reported_not_raised(kb_settings: Settings, kb_dir: Path) -> None:
    kb_dir.mkdir(parents=True, exist_ok=True)
    (kb_dir / "knowledge_index.jsonl").write_text("这不是 JSON\n", encoding="utf-8")

    result = KnowledgeService(kb_settings).search("任意问题")
    assert result.status == "unavailable"
    assert "JSONDecodeError" in result.detail


def test_empty_index_is_reported(kb_settings: Settings, kb_dir: Path) -> None:
    write_knowledge_artifacts(kb_dir, [])
    result = KnowledgeService(kb_settings).search("任意问题")
    assert result.status == "unavailable"
    assert "为空" in result.detail


def test_manifest_is_optional(kb_settings: Settings, kb_dir: Path) -> None:
    """manifest 只用于核对维度：缺了就用工件自身推断，不因此拒绝服务。"""
    write_knowledge_artifacts(kb_dir, KNOWLEDGE_CHUNKS)
    (kb_dir / "manifest.json").unlink()

    service = KnowledgeService(kb_settings)
    assert service.mode == "keyword"
    assert service.search("敏感字段怎么打码").status == "ok"


# ============================================================ 单例与真实工件
def test_service_cache_can_be_cleared_and_rebuilt(kb_settings: Settings, kb_index: Path, patch_settings) -> None:
    patch_settings(kb_settings)
    first = get_knowledge_service()
    assert get_knowledge_service() is first

    get_knowledge_service.cache_clear()
    assert get_knowledge_service() is not first


def test_repo_artifacts_contract() -> None:
    """仓库里真实工件的契约（验收 A）。

    数据在 ``data/`` 下、被 gitignore → 缺失时 ``skip`` 而不是失败：
    工件是可再生的，不该因为"没跑过导出"就让测试挂掉。
    """
    index = BASE_DIR / "data" / "knowledge" / "knowledge_index.jsonl"
    if not index.exists():
        pytest.skip("仓库没有知识库工件（先跑 uv run python -m rag_lab.export_index）")

    rows = [
        json.loads(line)
        for line in index.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    expected = {"chunk_id", "doc_id", "source_path", "section", "text", "line_start", "line_end", "vector"}
    for row in rows:
        assert expected <= set(row), f"片段缺字段: {expected - set(row)}"
        assert isinstance(row["vector"], list) and row["vector"]
    dimensions = {len(row["vector"]) for row in rows}
    assert len(dimensions) == 1, f"向量维度不一致：{dimensions}"

    manifest = json.loads((index.parent / "manifest.json").read_text(encoding="utf-8"))
    assert {
        "built_at",
        "space",
        "provider",
        "model",
        "dim",
        "chunks",
        "ngram",
        "corpus",
    } <= set(manifest)
    assert manifest["chunks"] == len(rows)
    assert manifest["dim"] in dimensions
    assert manifest["ngram"] == 2
