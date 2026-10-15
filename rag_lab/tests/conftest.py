"""rag_lab 测试的公共隔离装置。

三条**兜底铁律**，都由 autouse 夹具强制：

1. **测试不许读 ``rag_lab/.env``**。
   `.env` 里是真实 API Key。踩过一次：加了真实配置之后，那条「没配 Key 应该报错」
   的用例反过来**成功了** —— 因为它读到了真实 Key，并在测试里真调了一次向量服务。
   要读 `.env` 的用例必须自己传 ``env_file=...``。
2. **测试不许建立真实连接**（``socket.create_connection`` 直接失败）。
   API 路径一律用 ``httpx.MockTransport`` 编造响应（它不走 socket，不受影响）。
3. **DNS 解析被换成离线的确定性假实现**（不是拦截掉）。
   原因：``_host_is_unresolvable()`` 这个诊断逻辑本身要调 ``getaddrinfo``，
   如果直接把它禁掉，就没法验证「域名不存在 → 立刻中止」「域名存在 → 继续重试」
   这两条真实分支了。
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from pathlib import Path

import pytest

from rag_lab import embedding as embedding_module

# 假 DNS：这两种一定算解析失败，其余一律解析到 127.0.0.1
_UNRESOLVABLE_MARKERS = (".invalid", "{")


@pytest.fixture(autouse=True)
def isolate_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """把默认 ``.env`` 指到不存在的路径，避免测试读到真实配置。"""
    monkeypatch.setattr(embedding_module, "DEFAULT_ENV_FILE", tmp_path / "no-such.env")
    yield


@pytest.fixture(autouse=True)
def forbid_real_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """隔离网络：禁止真实连接，DNS 换成离线假实现。"""

    def _blocked_connect(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "测试不允许建立真实连接：请用 FakeEmbedder 或 httpx.MockTransport"
        )

    def _fake_getaddrinfo(
        host: object, port: object, *args: object, **kwargs: object
    ) -> list[tuple]:
        text = str(host)
        if any(marker in text for marker in _UNRESOLVABLE_MARKERS):
            raise socket.gaierror(-2, "Name or service not known")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", int(port or 0)))]

    monkeypatch.setattr(socket, "create_connection", _blocked_connect)
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo)
    yield
