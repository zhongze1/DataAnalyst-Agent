"""RAG-03 向量生成 / 缓存 / 错误处理测试。

**全部离线**：API 路径用 ``httpx.MockTransport`` 编造响应，不触碰任何真实服务；
默认路径用确定性的 ``FakeEmbedder``。

覆盖重点：请求契约、退避重试边界、鉴权错误不重试、响应异常一律报错、
缓存命中不重复算、**中途失败不丢已完成的部分**、以及"Key 不进日志/异常"。
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from rag_lab import embedding as embedding_module
from rag_lab.chunker import LAB_ROOT, Chunk
from rag_lab.embedding import (
    DEFAULT_CHUNKS_FILE,
    ApiEmbedder,
    EmbeddingAuthError,
    EmbeddingCache,
    EmbeddingConfig,
    EmbeddingConfigError,
    EmbeddingRequestError,
    EmbeddingResponseError,
    EmbeddingUnavailableError,
    FakeEmbedder,
    build_embedder,
    config_from_env,
    embed_chunks,
    load_chunks,
    main,
)

API_KEY = "sk-unit-test-key-do-not-leak"


# ============================================================ 夹具与工具
def _config(**overrides) -> EmbeddingConfig:
    base: dict = {
        "provider": "api",
        "base_url": "https://api.example.com/v1",
        "api_key": API_KEY,
        "model": "test-model",
        "batch_size": 2,
        "max_retries": 2,
    }
    base.update(overrides)
    return EmbeddingConfig(**base)


def _ok_response(*vectors: list[float], start: int = 0) -> dict:
    return {
        "data": [
            {"embedding": vector, "index": start + index}
            for index, vector in enumerate(vectors)
        ]
    }


def _embedder(handler, config: EmbeddingConfig | None = None, **kwargs):
    """构造一个走 MockTransport 的 ApiEmbedder，并记录 sleep 调用。"""
    sleeps: list[float] = []
    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)
    embedder = ApiEmbedder(
        config or _config(), client=client, sleep=lambda seconds: sleeps.append(seconds), **kwargs
    )
    return embedder, sleeps


def _chunk(text: str, index: int = 1) -> Chunk:
    import hashlib

    return Chunk(
        chunk_id=f"d#{index:04d}",
        doc_id="d",
        source_path="corpus/d.md",
        section="s",
        text=text,
        line_start=1,
        line_end=1,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
    )


# ============================================================ 配置
def test_config_from_env_reads_env_file_and_env_wins(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# 注释行\nRAG_LAB_EMBEDDING_PROVIDER=api\nRAG_LAB_EMBEDDING_MODEL='from-file'\n"
        "RAG_LAB_EMBEDDING_BATCH_SIZE=8\n",
        encoding="utf-8",
    )
    config = config_from_env(
        {
            "RAG_LAB_EMBEDDING_API_KEY": API_KEY,
            "RAG_LAB_EMBEDDING_BASE_URL": "https://api.example.com/v1",
            "RAG_LAB_EMBEDDING_MODEL": "from-env",
        },
        env_file=env_file,
    )
    assert config.provider == "api"
    assert config.model == "from-env", "进程环境变量优先于 .env"
    assert config.batch_size == 8


def test_env_file_supports_inline_comments_and_quotes(tmp_path: Path) -> None:
    """`.env.example` 里写了行内注释，解析器就得支持它，否则照抄会静默配错。"""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "RAG_LAB_EMBEDDING_PROVIDER=fake   # 行内注释\n"
        'RAG_LAB_EMBEDDING_MODEL="带#的值"\n'
        "\n"
        "# 整行注释\n",
        encoding="utf-8",
    )
    config = config_from_env({}, env_file=env_file)
    assert config.provider == "fake"
    assert config.model == "带#的值", "引号内的 # 不是注释"


def test_config_rejects_missing_api_credentials() -> None:
    with pytest.raises(EmbeddingConfigError, match="BASE_URL"):
        EmbeddingConfig(provider="api", model="m", api_key="k")
    with pytest.raises(EmbeddingConfigError, match="API_KEY"):
        EmbeddingConfig(provider="api", base_url="https://x/v1", model="m")
    with pytest.raises(EmbeddingConfigError, match="MODEL"):
        EmbeddingConfig(provider="api", base_url="https://x/v1", api_key="k", model="")


@pytest.mark.parametrize(
    "overrides",
    [{"provider": "nope"}, {"batch_size": 0}, {"max_retries": -1}],
)
def test_config_rejects_illegal_values(overrides: dict) -> None:
    with pytest.raises(EmbeddingConfigError):
        EmbeddingConfig(**overrides)


def test_api_key_never_appears_in_repr_or_errors() -> None:
    """Key 不能出现在任何可能被打印的地方。"""
    config = _config()
    assert API_KEY not in repr(config)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad key"})

    embedder, _ = _embedder(handler, config)
    with pytest.raises(EmbeddingAuthError) as caught:
        embedder.embed(["x"])
    assert API_KEY not in str(caught.value)


def test_default_batch_size_is_conservative(tmp_path: Path) -> None:
    """默认批大小必须是保守值。

    踩过：默认 32 时，阿里云 text-embedding-v4 直接返回
    ``HTTP 400 InvalidParameter: batch size ... should not be larger than 10``。
    多数国内向量服务单请求上限都在 10 附近，默认值不能太乐观。
    """
    assert EmbeddingConfig().batch_size == 10
    assert config_from_env({}, env_file=tmp_path / "not-exist.env").batch_size == 10
    # 编排函数的默认值也要跟着走（它是关键字专用参数，默认值在 __kwdefaults__）
    assert embed_chunks.__kwdefaults__["batch_size"] == 10


def test_space_changes_with_model_and_dim() -> None:
    assert _config().space != _config(model="other").space
    assert _config().space != _config(dim=8).space, "换维度换空间，缓存不能串用"


# ============================================================ FakeEmbedder
def test_fake_embedder_is_deterministic_and_normalised() -> None:
    embedder = FakeEmbedder(dim=32)
    first = embedder.embed(["同一段文本"])[0]
    assert first == embedder.embed(["同一段文本"])[0], "同文本必须同向量（缓存依赖它）"
    assert first != embedder.embed(["另一段文本"])[0]
    assert len(first) == 32
    assert abs(sum(value * value for value in first) ** 0.5 - 1.0) < 1e-6


def test_fake_embedder_rejects_tiny_dim() -> None:
    with pytest.raises(EmbeddingConfigError):
        FakeEmbedder(dim=4)


def test_build_embedder_picks_implementation() -> None:
    assert isinstance(build_embedder(EmbeddingConfig(provider="fake", dim=16)), FakeEmbedder)
    assert isinstance(build_embedder(_config()), ApiEmbedder)


# ============================================================ API 契约
def test_api_request_shape_and_response_order() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.example.com/v1/embeddings"
        assert request.headers["authorization"] == f"Bearer {API_KEY}"
        seen.append(json.loads(request.content))
        # 故意打乱 index，验证按 index 归位
        return httpx.Response(
            200,
            json={
                "data": [
                    {"embedding": [0.2, 0.2], "index": 1},
                    {"embedding": [0.1, 0.1], "index": 0},
                ]
            },
        )

    embedder, _ = _embedder(handler)
    vectors = embedder.embed(["甲", "乙"])

    assert seen[0]["model"] == "test-model"
    assert seen[0]["input"] == ["甲", "乙"]
    assert vectors == [[0.1, 0.1], [0.2, 0.2]], "必须按 index 还原顺序"
    assert embedder.dim == 2, "未配置维度时应自动推断并锁定"


def test_api_empty_input_does_not_call_the_service() -> None:
    called = []

    def handler(request: httpx.Request) -> httpx.Response:
        called.append(1)
        return httpx.Response(200, json=_ok_response([0.1]))

    embedder, _ = _embedder(handler)
    assert embedder.embed([]) == []
    assert called == []


def test_retries_on_429_then_succeeds() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(429, json={"error": "slow down"})
        return httpx.Response(200, json=_ok_response([0.5, 0.5]))

    embedder, sleeps = _embedder(handler)
    assert embedder.embed(["x"]) == [[0.5, 0.5]]
    assert attempts["n"] == 2
    assert sleeps, "重试之间必须退避等待"


def test_auth_error_is_not_retried() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(403, json={"error": "forbidden"})

    embedder, sleeps = _embedder(handler)
    with pytest.raises(EmbeddingAuthError):
        embedder.embed(["x"])
    assert attempts["n"] == 1, "鉴权失败重试没有意义"
    assert sleeps == []


def test_bad_request_is_not_retried() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(404, json={"error": "model not found"})

    embedder, _ = _embedder(handler)
    with pytest.raises(EmbeddingRequestError):
        embedder.embed(["x"])
    assert attempts["n"] == 1


def test_server_error_exhausts_retries() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(503, json={"error": "down"})

    embedder, sleeps = _embedder(handler, _config(max_retries=2))
    with pytest.raises(EmbeddingUnavailableError):
        embedder.embed(["x"])
    assert attempts["n"] == 3, "max_retries=2 => 共 3 次尝试"
    assert len(sleeps) == 2, "最后一次失败后不再等待"


def test_network_error_is_retried_then_reported() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        raise httpx.ConnectError("connection refused")

    embedder, sleeps = _embedder(handler, _config(max_retries=1))
    with pytest.raises(EmbeddingUnavailableError):
        embedder.embed(["x"])
    assert attempts["n"] == 2 and len(sleeps) == 1


def test_count_mismatch_is_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_response([0.1, 0.2]))  # 只回 1 条

    embedder, _ = _embedder(handler)
    with pytest.raises(EmbeddingResponseError, match="条数不符"):
        embedder.embed(["甲", "乙"])


def test_dimension_mismatch_is_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_response([0.1, 0.2, 0.3]))

    embedder, _ = _embedder(handler, _config(dim=4))
    with pytest.raises(EmbeddingResponseError, match="维度"):
        embedder.embed(["x"])


def test_unresolvable_host_fails_fast_as_config_error() -> None:
    """域名写错时：立刻中止、**不重试**，并指出是 BASE_URL 的问题。

    踩过：本机有系统代理时，这种错误会被报成 "Server disconnected"（像是服务端故障），
    还会白等 3 次退避 —— 排查方向被完全带偏。现在连接层失败会先自己解析一次域名。
    """
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        raise httpx.ConnectError("[SSL: UNEXPECTED_EOF_WHILE_READING] 代理掐断了连接")

    embedder, sleeps = _embedder(
        handler, _config(base_url="https://no-such-host-rag-probe.invalid/v1", max_retries=3)
    )
    with pytest.raises(EmbeddingConfigError) as caught:
        embedder.embed(["x"])

    assert attempts["n"] == 1, "配置错误不该重试"
    assert sleeps == [], "更不该退避等待"
    message = str(caught.value)
    assert "no-such-host-rag-probe.invalid" in message and "BASE_URL" in message
    assert API_KEY not in message


def test_connect_error_to_a_real_host_is_still_retried() -> None:
    """域名能解析时维持原语义：可能是代理抖动/临时网络问题，值得重试。"""
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        raise httpx.ConnectError("connection reset by peer")

    embedder, sleeps = _embedder(handler, _config(max_retries=2))
    with pytest.raises(EmbeddingUnavailableError):
        embedder.embed(["x"])

    assert attempts["n"] == 3 and len(sleeps) == 2


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("true", True), ("1", True), ("YES", True), ("false", False), ("0", False), ("off", False)],
)
def test_trust_env_parsing(raw: str, expected: bool, tmp_path: Path) -> None:
    config = config_from_env(
        {"RAG_LAB_EMBEDDING_TRUST_ENV": raw}, env_file=tmp_path / "none.env"
    )
    assert config.trust_env is expected


def test_trust_env_rejects_garbage(tmp_path: Path) -> None:
    with pytest.raises(EmbeddingConfigError, match="TRUST_ENV"):
        config_from_env(
            {"RAG_LAB_EMBEDDING_TRUST_ENV": "maybe"}, env_file=tmp_path / "none.env"
        )


def test_trust_env_reaches_the_http_client() -> None:
    """开关要真的作用到 httpx 上，而不是只躺在配置里。

    httpx 没有公开的 trust_env 属性，所以这里读它的私有字段 —— 值得为它破例一次。
    """
    embedder = build_embedder(_config(trust_env=False))
    assert embedder._client._trust_env is False  # noqa: SLF001
    assert build_embedder(_config(trust_env=True))._client._trust_env is True  # noqa: SLF001


def test_non_json_body_is_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>网关错误</html>")

    embedder, _ = _embedder(handler)
    with pytest.raises(EmbeddingResponseError):
        embedder.embed(["x"])


# ============================================================ 缓存
def test_cache_roundtrip_and_persistence(tmp_path: Path) -> None:
    path = tmp_path / "emb.jsonl"
    cache = EmbeddingCache(path)
    cache.put("sp", "k1", [0.1, 0.2])

    assert cache.get("sp", "k1") == [0.1, 0.2]
    assert cache.get("other", "k1") is None, "不同空间不得串用"
    assert cache.get("sp", "k2") is None

    reopened = EmbeddingCache(path)
    assert reopened.get("sp", "k1") == [0.1, 0.2], "重启后缓存必须还在"


def test_cache_skips_corrupted_lines(tmp_path: Path) -> None:
    path = tmp_path / "emb.jsonl"
    path.write_text(
        "{ 坏掉的 json\n" + json.dumps({"space": "s", "key": "k", "vector": [1.0]}) + "\n",
        encoding="utf-8",
    )
    cache = EmbeddingCache(path)
    assert cache.get("s", "k") == [1.0]


def test_cache_appends_incrementally(tmp_path: Path) -> None:
    path = tmp_path / "emb.jsonl"
    cache = EmbeddingCache(path)
    cache.put("s", "a", [0.1])
    cache.put("s", "b", [0.2])
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2, "每条算完立即落盘（断点续跑的前提）"


# ============================================================ 编排
class _CountingFake(FakeEmbedder):
    name = "counting"

    def __init__(self, dim: int = 16) -> None:
        super().__init__(dim=dim)
        self.calls = 0
        self.embedded: list[str] = []

    def embed(self, texts):
        self.calls += 1
        self.embedded.extend(texts)
        return super().embed(texts)


def test_embed_chunks_batches_and_caches(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "e.jsonl")
    embedder = _CountingFake()
    chunks = [_chunk(f"文本{i}", i) for i in range(1, 6)]
    progress: list[tuple[int, int]] = []

    stats = embed_chunks(
        chunks,
        embedder,
        cache,
        space="s",
        batch_size=2,
        progress=lambda done, total: progress.append((done, total)),
    )

    assert (stats.total, stats.computed, stats.batches) == (5, 5, 3)
    assert stats.cached == 0 and stats.dim == 16
    assert len(embedder.embedded) == 5
    assert progress == [(2, 5), (4, 5), (5, 5)]
    assert len(cache) == 5


def test_embed_chunks_second_run_hits_cache(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "e.jsonl")
    chunks = [_chunk(f"文本{i}", i) for i in range(1, 4)]

    first = embed_chunks(chunks, _CountingFake(), cache, space="s", batch_size=2)
    billing = _CountingFake()
    second = embed_chunks(chunks, billing, cache, space="s", batch_size=2)

    assert first.computed == 3
    assert second.computed == 0 and second.cached == 3
    assert billing.calls == 0, "缓存命中就绝不能再调服务（这是省钱的关键）"


def test_embed_chunks_respects_limit(tmp_path: Path) -> None:
    stats = embed_chunks(
        [_chunk(f"t{i}", i) for i in range(1, 6)],
        _CountingFake(),
        EmbeddingCache(tmp_path / "e.jsonl"),
        space="s",
        batch_size=2,
        limit=2,
    )
    assert (stats.total, stats.computed) == (2, 2)


def test_mid_run_failure_keeps_finished_batches(tmp_path: Path) -> None:
    """第二批失败时，第一批必须已经落盘 —— 重跑不该重复付费。"""

    class FailOnSecondBatch(_CountingFake):
        def embed(self, texts):
            if self.calls == 1:
                raise EmbeddingUnavailableError("模拟第二批发失败")
            return super().embed(texts)

    cache = EmbeddingCache(tmp_path / "e.jsonl")
    chunks = [_chunk(f"文本{i}", i) for i in range(1, 5)]

    with pytest.raises(EmbeddingUnavailableError):
        embed_chunks(chunks, FailOnSecondBatch(), cache, space="s", batch_size=2)
    assert len(cache) == 2, "已完成的那一批必须已经写进缓存"

    resumed = embed_chunks(chunks, _CountingFake(), cache, space="s", batch_size=2)
    assert (resumed.cached, resumed.computed) == (2, 2), "重跑应只补齐缺的两条"


# ============================================================ 语料与 CLI
def test_load_chunks_roundtrips_the_real_file() -> None:
    chunks = load_chunks(DEFAULT_CHUNKS_FILE)
    assert chunks, "chunks.jsonl 不该为空（先跑 chunker）"
    assert all(chunk.text and chunk.content_hash for chunk in chunks)
    assert all(chunk.source_path.startswith("corpus/") for chunk in chunks)
    lines = [
        line
        for line in DEFAULT_CHUNKS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(chunks) == len(lines)


def test_load_chunks_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(EmbeddingConfigError, match="找不到分块文件"):
        load_chunks(tmp_path / "nope.jsonl")


def _write_chunks(path: Path, texts: list[str]) -> None:
    lines = [json.dumps(chunk.to_dict(), ensure_ascii=False) for chunk in
             [_chunk(text, index) for index, text in enumerate(texts, start=1)]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_cli_fake_provider_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    chunks = tmp_path / "chunks.jsonl"
    cache = tmp_path / "index" / "emb.jsonl"
    _write_chunks(chunks, ["甲", "乙", "丙"])

    code = main(
        ["--provider", "fake", "--chunks", str(chunks), "--cache", str(cache), "--batch-size", "2"]
    )

    assert code == 0
    assert cache.exists()
    out = capsys.readouterr().out
    assert "片段总数: 3" in out and "本次计算: 3" in out and "向量维度: 256" in out


def test_cli_dry_run_does_not_write_cache(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    chunks = tmp_path / "chunks.jsonl"
    cache = tmp_path / "emb.jsonl"
    _write_chunks(chunks, ["甲", "乙"])

    assert main(["--provider", "fake", "--chunks", str(chunks), "--cache", str(cache), "--dry-run"]) == 0
    assert not cache.exists()
    assert "待计算: 2" in capsys.readouterr().out


def test_cli_missing_chunks_file_exits_nonzero(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    code = main(["--provider", "fake", "--chunks", str(tmp_path / "nope.jsonl")])
    assert code == 3
    assert "EMBEDDING_CONFIG" in capsys.readouterr().out


def test_cli_api_provider_without_key_fails_before_any_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    chunks = tmp_path / "chunks.jsonl"
    _write_chunks(chunks, ["甲"])
    for name in (
        "RAG_LAB_EMBEDDING_PROVIDER",
        "RAG_LAB_EMBEDDING_BASE_URL",
        "RAG_LAB_EMBEDDING_API_KEY",
        "RAG_LAB_EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    # 双保险：这条用例关心的是「缺配置必须**在任何网络请求之前**失败」，
    # 所以显式把默认 .env 指空（conftest 里已经兜了一层）——
    # 否则一旦本机配了真实 Key，这条用例会反过来真调一次服务并"通过"。
    monkeypatch.setattr(embedding_module, "DEFAULT_ENV_FILE", tmp_path / "no-such.env")

    code = main(["--provider", "api", "--chunks", str(chunks), "--cache", str(tmp_path / "e.jsonl")])

    assert code == 3
    out = capsys.readouterr().out
    assert "EMBEDDING_CONFIG" in out
    assert "BASE_URL" in out or "API_KEY" in out, "报错必须指出缺的是哪一项"


def test_lab_root_is_referenced_by_default_paths() -> None:
    assert LAB_ROOT.name == "rag_lab"
    assert DEFAULT_CHUNKS_FILE.parent == LAB_ROOT
