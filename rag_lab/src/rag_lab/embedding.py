"""向量生成与缓存（RAG-03）。

三件事，别的都不做（检索留到 RAG-04）：

1. **生成**：把片段文本转成向量。走 **OpenAI 兼容的 ``POST {base_url}/embeddings``**，
   所以换服务商只改配置、不改代码；另外提供一个**确定性离线**的 ``fake`` 实现，
   让测试与"没有 Key 的机器"也能把链路跑通。
2. **缓存**：按 ``内容指纹 + 模型空间`` 落 JSONL。命中就不重复付费；
   **每条算完立即落盘**，所以中途失败也不会白花一次钱（下次接着跑）。
3. **错误处理**：把失败分成"改配置才能好"和"等一等就好"两类 ——
   前者立刻中止（不浪费重试），后者指数退避重试；响应结构/条数/维度不对一律当成错误，
   绝不"凑合着用"。

隐私：API Key 只从环境变量 / ``rag_lab/.env`` 读取，**不写进代码、不落进日志、不进缓存**。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import socket
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

from rag_lab.chunker import LAB_ROOT, Chunk

logger = logging.getLogger(__name__)

DEFAULT_CHUNKS_FILE = LAB_ROOT / "chunks.jsonl"
DEFAULT_CACHE_FILE = LAB_ROOT / "index" / "embeddings.jsonl"
DEFAULT_ENV_FILE = LAB_ROOT / ".env"

# 向量维度只有在拿到第一个响应后才知道时用 0 表示"自动推断"
AUTO_DIM = 0


# --------------------------------------------------------------------- 错误
class EmbeddingError(RuntimeError):
    """向量化相关错误的基类（``code`` 便于日志聚合，``message`` 给人看）。"""

    code = "EMBEDDING_ERROR"


class EmbeddingConfigError(EmbeddingError):
    """配置不完整/不合法（例如选了 api 却没给 Key）。**不要重试**。"""

    code = "EMBEDDING_CONFIG"


class EmbeddingAuthError(EmbeddingError):
    """鉴权失败（401/403）。**重试没有意义**，先检查 Key。"""

    code = "EMBEDDING_AUTH"


class EmbeddingRequestError(EmbeddingError):
    """请求本身被拒绝（400/404 等，例如模型名写错）。**不要重试**。"""

    code = "EMBEDDING_REQUEST"


class EmbeddingUnavailableError(EmbeddingError):
    """暂时不可用（429/5xx/超时/连接失败）。可以退避重试。"""

    code = "EMBEDDING_UNAVAILABLE"


class EmbeddingResponseError(EmbeddingError):
    """响应结构、条数或维度不对 —— 宁可报错，也不拿半个结果继续。"""

    code = "EMBEDDING_RESPONSE"


# --------------------------------------------------------------------- 配置
@dataclass(frozen=True)
class EmbeddingConfig:
    """向量化配置。"""

    provider: str = "fake"  # api | fake
    base_url: str = ""
    # repr=False：Key 不能出现在任何日志/异常输出里
    api_key: str = field(default="", repr=False)
    model: str = "fake-hash-v1"
    dim: int = 0  # 0 = 自动推断（首次响应后锁定）
    # 默认取保守值 10：多数国内向量服务单请求上限就是 10 条
    # （阿里云 text-embedding-v4 实测：batch=32 直接 HTTP 400 InvalidParameter）。
    # 服务商允许更多时，调大它能省往返次数。
    batch_size: int = 10
    timeout: float = 30.0
    max_retries: int = 3
    # 是否读取系统代理（httpx 的 trust_env）。默认 True：很多环境必须走代理。
    # 注意 Windows 上"系统代理"来自注册表，httpx 也会读进来（见 _utils.getproxies）——
    # 留这个开关是为了"某个域名被代理搞坏"时能改配置逃生，而不是改代码。
    trust_env: bool = True
    chunks_path: Path = DEFAULT_CHUNKS_FILE
    cache_path: Path = DEFAULT_CACHE_FILE

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise EmbeddingConfigError(f"batch_size 必须 >= 1，当前 {self.batch_size}")
        if self.max_retries < 0:
            raise EmbeddingConfigError(f"max_retries 不能为负，当前 {self.max_retries}")
        if self.provider not in {"api", "fake"}:
            raise EmbeddingConfigError(
                f"未知的 provider: {self.provider!r}（可选 api / fake）"
            )
        if self.provider == "api":
            if not self.base_url.strip():
                raise EmbeddingConfigError(
                    "provider=api 必须配置 RAG_LAB_EMBEDDING_BASE_URL"
                    "（任意 OpenAI 兼容端点，例如 https://<host>/v1）"
                )
            if not self.api_key.strip():
                raise EmbeddingConfigError(
                    "provider=api 必须配置 RAG_LAB_EMBEDDING_API_KEY"
                    "（放在 rag_lab/.env 里，不要提交到 Git）"
                )
            if not self.model.strip():
                raise EmbeddingConfigError("provider=api 必须配置 RAG_LAB_EMBEDDING_MODEL")

    @property
    def space(self) -> str:
        """缓存命名空间：换 provider / 模型 / 维度 都会自动失效（不会串用旧向量）。"""
        return f"{self.provider}/{self.model}/dim={self.dim or 'auto'}"


def load_env_file(path: Path | None = None) -> dict[str, str]:
    """极简 ``.env`` 解析：``KEY=VALUE``、``#`` 整行注释、空行。

    支持用引号包裹的值，也支持行内注释（``KEY=v  # 说明``，要求 ``#`` 前有空白）。
    刻意不引 python-dotenv：这一层只需要 20 行。
    """
    env_path = Path(path or DEFAULT_ENV_FILE)
    if not env_path.exists():
        return {}
    values: dict[str, str] = {}
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if value[:1] not in {'"', "'"}:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        values[key.strip()] = value.strip('"').strip("'")
    return values


def config_from_env(
    env: dict[str, str] | None = None,
    *,
    env_file: Path | None = None,
) -> EmbeddingConfig:
    """按 ``.env`` + 进程环境变量构造配置（真实环境变量优先）。"""
    merged: dict[str, str] = dict(load_env_file(env_file))
    merged.update(os.environ if env is None else env)

    def get(name: str, default: str = "") -> str:
        return merged.get(f"RAG_LAB_{name}", default).strip()

    def get_int(name: str, default: int) -> int:
        raw = get(name)
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError as exc:
            raise EmbeddingConfigError(f"RAG_LAB_{name} 必须是整数，当前 {raw!r}") from exc

    def get_bool(name: str, default: bool) -> bool:
        raw = get(name).lower()
        if not raw:
            return default
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False
        raise EmbeddingConfigError(
            f"RAG_LAB_{name} 必须是布尔值（true/false），当前 {raw!r}"
        )

    return EmbeddingConfig(
        provider=get("EMBEDDING_PROVIDER", "fake").lower(),
        base_url=get("EMBEDDING_BASE_URL"),
        api_key=get("EMBEDDING_API_KEY"),
        model=get("EMBEDDING_MODEL", "fake-hash-v1"),
        dim=get_int("EMBEDDING_DIM", AUTO_DIM),
        batch_size=get_int("EMBEDDING_BATCH_SIZE", 10),
        timeout=float(get_int("EMBEDDING_TIMEOUT", 30)),
        max_retries=get_int("EMBEDDING_MAX_RETRIES", 3),
        trust_env=get_bool("EMBEDDING_TRUST_ENV", True),
    )


# --------------------------------------------------------------------- 接口
class Embedder(Protocol):
    """把文本转成向量。实现必须保证"同文本同向量"（缓存依赖这一点）。"""

    name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class FakeEmbedder:
    """确定性离线向量：`sha256(文本)` 展开成固定维度并 L2 归一化。

    用途：跑通链路、写测试、在没 Key 的机器上验证缓存与错误处理。
    **它不懂语义**，相似度没有意义，RAG-04 的检索评估必须用真实模型。
    """

    name = "fake"

    def __init__(self, dim: int = 256) -> None:
        if dim < 8:
            raise EmbeddingConfigError(f"fake 维度至少 8，当前 {dim}")
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> list[float]:
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        raw: list[float] = []
        while len(raw) < self.dim:
            seed = hashlib.sha256(seed).digest()
            raw.extend(byte / 255.0 - 0.5 for byte in seed)
        vector = raw[: self.dim]
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]


def _host_is_unresolvable(host: str) -> bool:
    """判断"连不上"到底是因为**域名写错了**，还是网络/代理暂时不通。

    这是踩坑之后加的：本机有系统代理时，把一个不存在的域名交给它，
    代理会先接受连接再掐断，httpx 只能报 ``RemoteProtocolError/UNEXPECTED_EOF``
    —— 看起来像"服务端故障"，真因（域名解析不了）被完全掩盖，
    还会白等 3 次退避。所以连接层失败时先自己解析一次。

    - 解析失败（``gaierror``） → True：配置问题，重试没有意义
    - 解析成功               → False：可能是代理抖动，值得重试
    - 其它异常               → False：宁可多试一次，也不要把可重试的问题误判成配置错误
    """

    if not host:
        return False
    try:
        socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return True
    except OSError:
        return False
    return False


class ApiEmbedder:
    """OpenAI 兼容 ``POST {base_url}/embeddings``。

    只依赖 ``httpx``（主工程已在用），因此 DeepSeek 之外的任意兼容端点都能接。
    """

    name = "api"

    def __init__(
        self,
        config: EmbeddingConfig,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._sleep = sleep
        self._dim = config.dim
        self._client = client or httpx.Client(
            timeout=config.timeout, trust_env=config.trust_env
        )
        self.base_url = config.base_url.rstrip("/")
        self._host = urlparse(self.base_url).hostname or ""

    # 自动推断维度时，用第一条响应锁定
    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {"model": self._config.model, "input": list(texts)}
        data = self._post_with_retries(payload)
        return self._parse(data, expected=len(texts))

    # ----------------------------------------------------------------- 内部
    def _post_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        attempts = self._config.max_retries + 1
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self._client.post(
                    f"{self.base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self._config.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
            except httpx.HTTPError as exc:
                # 连接层失败先分辨原因：域名解析不了 = 配置写错了，立刻中止且不重试
                # （踩过：有系统代理时会被报成 "Server disconnected"，误导排查方向）
                if _host_is_unresolvable(self._host):
                    raise EmbeddingConfigError(
                        f"域名无法解析：{self._host}"
                        " —— 请检查 RAG_LAB_EMBEDDING_BASE_URL"
                        f"（原始错误：{type(exc).__name__}）"
                    ) from exc
                last = EmbeddingUnavailableError(
                    f"连接向量服务失败（{type(exc).__name__}）：{exc}"
                )
                self._backoff(attempt, attempts)
                continue

            status = response.status_code
            if status == 200:
                try:
                    return response.json()
                except ValueError as exc:
                    raise EmbeddingResponseError(
                        f"向量服务返回的不是 JSON（status=200）：{response.text[:200]}"
                    ) from exc
            if status in (401, 403):
                # 重试没有意义：Key 不对/没权限
                raise EmbeddingAuthError(
                    f"向量服务拒绝鉴权（HTTP {status}）。请检查 RAG_LAB_EMBEDDING_API_KEY 是否有效"
                )
            if status == 429 or status >= 500:
                last = EmbeddingUnavailableError(
                    f"向量服务暂时不可用（HTTP {status}）：{response.text[:200]}"
                )
                self._backoff(attempt, attempts)
                continue
            # 其余 4xx：请求/模型名有问题，重试也是白搭
            raise EmbeddingRequestError(
                f"向量服务拒绝请求（HTTP {status}）：{response.text[:200]}"
            )
        raise last or EmbeddingUnavailableError("向量服务不可用，且未记录到具体原因")

    def _backoff(self, attempt: int, attempts: int) -> None:
        if attempt + 1 >= attempts:
            return
        delay = min(1.5 * (2**attempt), 10.0)
        logger.warning("向量服务失败，%.1fs 后重试（第 %d/%d 次）", delay, attempt + 1, attempts)
        self._sleep(delay)

    def _parse(self, data: dict[str, Any], *, expected: int) -> list[list[float]]:
        items = data.get("data")
        if not isinstance(items, list) or len(items) != expected:
            raise EmbeddingResponseError(
                f"向量条数不符：请求 {expected} 条，返回 {len(items) if isinstance(items, list) else '非列表'}"
            )

        ordered = sorted(items, key=lambda item: item.get("index", 0))
        vectors: list[list[float]] = []
        for item in ordered:
            vector = item.get("embedding")
            if not isinstance(vector, list) or not vector:
                raise EmbeddingResponseError("返回项缺少 embedding 字段或为空")
            if not all(isinstance(value, (int, float)) for value in vector):
                raise EmbeddingResponseError("embedding 里含非数值元素")
            vectors.append([float(value) for value in vector])

        if self._dim == AUTO_DIM:
            self._dim = len(vectors[0])  # 首次响应锁定维度
        elif any(len(vector) != self._dim for vector in vectors):
            got = {len(vector) for vector in vectors}
            raise EmbeddingResponseError(
                f"向量维度与配置不符：配置 {self._dim}，返回 {sorted(got)}"
            )
        return vectors


def build_embedder(config: EmbeddingConfig, **kwargs: Any) -> Embedder:
    """按配置构造实现（``api`` 需 Key，``fake`` 离线）。"""
    if config.provider == "fake":
        return FakeEmbedder(dim=config.dim or 256)
    return ApiEmbedder(config, **kwargs)


# --------------------------------------------------------------------- 缓存
class EmbeddingCache:
    """按 ``空间 + 内容指纹`` 缓存的 JSONL 存储。

    - **追加写**：每条算完立即落盘 —— 中途失败/被 Ctrl-C 也不会丢掉已花的钱；
    - **空间隔离**：换 provider/模型/维度后旧向量自动失效，不会串用；
    - 读取时后写覆盖先写（同一个 key 重复出现时以最新为准）。
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = Path(path or DEFAULT_CACHE_FILE)
        self._items: dict[tuple[str, str], list[float]] = {}
        self._load()

    @property
    def path(self) -> Path:
        return self._path

    def __len__(self) -> int:
        return len(self._items)

    def _load(self) -> None:
        if not self._path.exists():
            return
        for raw in self._path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                space, key, vector = item["space"], item["key"], item["vector"]
            except (ValueError, KeyError, TypeError):
                logger.warning("跳过损坏的缓存行: %s", line[:80])
                continue
            if isinstance(vector, list):
                self._items[(str(space), str(key))] = [float(v) for v in vector]

    def get(self, space: str, key: str) -> list[float] | None:
        return self._items.get((space, key))

    def put(self, space: str, key: str, vector: list[float]) -> None:
        self._items[(space, key)] = vector
        self._path.parent.mkdir(parents=True, exist_ok=True)
        record = {"space": space, "key": key, "dim": len(vector), "vector": vector}
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------- 编排
@dataclass
class EmbedStats:
    """一次向量化的账本（用于报告：花了多少、省了多少）。"""

    total: int = 0
    cached: int = 0
    computed: int = 0
    batches: int = 0
    dim: int = 0
    space: str = ""

    def render(self) -> str:
        lines = [
            f"空间: {self.space}",
            f"片段总数: {self.total}",
            f"缓存命中: {self.cached}",
            f"本次计算: {self.computed}（{self.batches} 批）",
            f"向量维度: {self.dim}",
        ]
        return "\n".join(lines)


def embed_chunks(
    chunks: Iterable[Chunk],
    embedder: Embedder,
    cache: EmbeddingCache,
    *,
    space: str,
    batch_size: int = 10,
    limit: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> EmbedStats:
    """为片段生成向量并写入缓存。

    先批量查缓存，只把缺失的送去算；每批算完立即写盘。
    任何一批失败都会**带着已完成的进度**抛出（不会丢），重跑即可续上。
    """
    items = list(chunks)
    if limit is not None:
        items = items[: max(0, limit)]

    stats = EmbedStats(total=len(items), space=space)
    pending: list[tuple[str, str]] = []  # (key, text)
    for chunk in items:
        key = chunk.content_hash
        cached = cache.get(space, key)
        if cached is not None:
            stats.cached += 1
            stats.dim = stats.dim or len(cached)
            continue
        pending.append((key, chunk.text))

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        vectors = embedder.embed([text for _, text in batch])
        if len(vectors) != len(batch):
            raise EmbeddingResponseError(
                f"实现返回的向量条数不符：请求 {len(batch)}，返回 {len(vectors)}"
            )
        for (key, _text), vector in zip(batch, vectors):
            cache.put(space, key, vector)
        stats.computed += len(batch)
        stats.batches += 1
        stats.dim = stats.dim or len(vectors[0])
        if progress is not None:
            progress(stats.computed, len(pending))
    return stats


def load_chunks(path: Path | None = None) -> list[Chunk]:
    """从 chunks.jsonl 读回片段。"""
    source = Path(path or DEFAULT_CHUNKS_FILE)
    if not source.exists():
        raise EmbeddingConfigError(
            f"找不到分块文件: {source}（先跑 uv run python -m rag_lab.chunker）"
        )
    chunks: list[Chunk] = []
    for raw in source.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        item = json.loads(line)
        chunks.append(
            Chunk(
                chunk_id=item["chunk_id"],
                doc_id=item["doc_id"],
                source_path=item["source_path"],
                section=item["section"],
                text=item["text"],
                line_start=int(item["line_start"]),
                line_end=int(item["line_end"]),
                content_hash=item["content_hash"],
                repeated_header=item.get("repeated_header", ""),
            )
        )
    return chunks


# --------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    """命令行入口：``uv run python -m rag_lab.embedding``。

    默认用 ``.env`` 里的配置；``--provider fake`` 可在完全离线的情况下验证链路。
    """
    parser = argparse.ArgumentParser(
        prog="python -m rag_lab.embedding",
        description="为 chunks.jsonl 生成向量并写入缓存（RAG-03，不含检索）",
    )
    parser.add_argument("--provider", choices=["fake", "api"], help="覆盖 .env 里的 provider")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_FILE, help="分块文件")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_FILE, help="缓存文件")
    parser.add_argument("--batch-size", type=int, help="覆盖批大小")
    parser.add_argument("--limit", type=int, help="只处理前 N 条（先小规模试跑，省钱）")
    parser.add_argument("--dry-run", action="store_true", help="只报告将处理多少条，不调用服务")
    args = parser.parse_args(argv)

    try:
        config = config_from_env()
        if args.provider:
            config = replace(config, provider=args.provider)
        config = replace(config, chunks_path=args.chunks, cache_path=args.cache)
        if args.batch_size:
            config = replace(config, batch_size=args.batch_size)
        chunks = load_chunks(config.chunks_path)
        cache = EmbeddingCache(config.cache_path)
        embedder = build_embedder(config)
    except EmbeddingError as exc:
        print(f"[{exc.code}] {exc}", flush=True)
        return 3

    if args.dry_run:
        cached = sum(
            1 for chunk in chunks if cache.get(config.space, chunk.content_hash) is not None
        )
        print(f"空间: {config.space}")
        print(f"片段总数: {len(chunks)}")
        print(f"缓存已有: {cached}")
        print(f"待计算: {len(chunks) - cached}（dry-run，未调用向量服务）")
        return 0

    print(f"provider={config.provider} | model={config.model} | 空间={config.space}")
    try:
        stats = embed_chunks(
            chunks,
            embedder,
            cache,
            space=config.space,
            batch_size=config.batch_size,
            limit=args.limit,
            progress=lambda done, total: print(f"  已处理 {done}/{total}", flush=True),
        )
    except EmbeddingError as exc:
        # 已算完的部分已经落盘，重跑会自动续上
        print(f"[{exc.code}] {exc}", flush=True)
        print(f"提示：已完成的向量已写入 {cache.path}，重跑会自动续上", flush=True)
        return 3

    print(stats.render())
    print(f"缓存文件: {cache.path}（当前共 {len(cache)} 条）")
    return 0


if __name__ == "__main__":  # pragma: no cover - 由 CLI 测试覆盖 main()
    raise SystemExit(main())
