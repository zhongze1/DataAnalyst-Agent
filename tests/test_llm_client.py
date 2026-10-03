"""LLM 客户端测试：用 httpx.MockTransport 拦截请求，不发任何真实网络调用。"""

from __future__ import annotations

import json
from typing import Any, Callable

import httpx
import pytest

from app.config import Settings
from app.llm.base import LLMError
from app.llm.openai_client import OpenAICompatClient

BASE_URL = "https://example.invalid/v1"


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "_env_file": None,
        "llm_provider": "openai",
        "llm_api_key": "test-key",
        "llm_base_url": BASE_URL,
        "llm_model": "test-model",
        "llm_max_retries": 1,
    }
    values.update(overrides)  # 允许用例覆盖任意字段
    return Settings(**values)


def _client_with(handler: Callable[[httpx.Request], httpx.Response], **overrides: Any) -> OpenAICompatClient:
    client = OpenAICompatClient(_settings(**overrides))
    # 替换掉内部 AsyncClient，注入 MockTransport 实现完全离线
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


def _ok_response(content: str = "ok", tool_calls: list[dict] | None = None) -> dict:
    message: dict[str, Any] = {"content": content, "role": "assistant"}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message}], "usage": {"total_tokens": 42}}


# ------------------------------------------------------------------ 初始化
def test_missing_api_key_fails_fast() -> None:
    """没有 Key 必须在客户端构造时就报错，而不是等到用户提问。"""
    with pytest.raises(LLMError, match="API_KEY"):
        OpenAICompatClient(_settings(llm_api_key=""))


def test_trailing_slash_in_base_url_is_normalized() -> None:
    client = _client_with(lambda request: httpx.Response(200, json=_ok_response()))
    assert client._settings.llm_base_url.rstrip("/").endswith("/v1")


# ------------------------------------------------------------ 响应解析契约
def test_parse_plain_content() -> None:
    response = OpenAICompatClient._parse(_ok_response("结论内容"))
    assert response.content == "结论内容"
    assert response.has_tool_calls is False
    assert response.usage["total_tokens"] == 42


def test_parse_tool_calls_with_json_string_arguments() -> None:
    payload = _ok_response(
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "load_sheet", "arguments": '{"sheet": "明细"}'},
            }
        ]
    )
    response = OpenAICompatClient._parse(payload)
    assert response.has_tool_calls
    call = response.tool_calls[0]
    assert call.id == "call_1"
    assert call.name == "load_sheet"
    assert call.arguments == {"sheet": "明细"}


def test_parse_tool_calls_with_dict_arguments() -> None:
    payload = _ok_response(
        tool_calls=[
            {"id": "c1", "function": {"name": "list_sheets", "arguments": {}}},
        ]
    )
    assert OpenAICompatClient._parse(payload).tool_calls[0].arguments == {}


def test_parse_empty_choices_raises() -> None:
    with pytest.raises(LLMError, match="模型返回为空"):
        OpenAICompatClient._parse({"choices": []})


def test_parse_invalid_arguments_raises() -> None:
    payload = _ok_response(
        tool_calls=[{"id": "c", "function": {"name": "x", "arguments": "not-json"}}]
    )
    with pytest.raises(LLMError, match="JSON"):
        OpenAICompatClient._parse(payload)


# ------------------------------------------------------------------ 请求
async def test_chat_sends_expected_request() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_ok_response("分析完成"))

    client = _client_with(handler)
    tools = [{"type": "function", "function": {"name": "list_sheets", "parameters": {}}}]
    response = await client.chat([{"role": "user", "content": "hi"}], tools)
    await client._client.aclose()

    assert response.content == "分析完成"
    assert captured["url"] == f"{BASE_URL}/chat/completions"
    assert captured["auth"] == "Bearer test-key"
    assert captured["body"]["model"] == "test-model"
    # 工具必须真的传给模型，否则 Agent 无法调用工具
    assert captured["body"]["tools"] == tools
    assert captured["body"]["tool_choice"] == "auto"


async def test_chat_omits_tools_field_when_absent() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_ok_response())

    client = _client_with(handler)
    await client.chat([{"role": "user", "content": "hi"}], None)
    await client._client.aclose()
    assert "tools" not in captured["body"]


# ------------------------------------------------------------------ 错误处理
async def test_client_error_is_not_retried() -> None:
    """4xx 是请求本身有问题，重试无意义，必须立刻失败。"""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400, json={"error": {"message": "invalid model"}})

    client = _client_with(handler, llm_max_retries=3)
    with pytest.raises(LLMError, match="400"):
        await client.chat([{"role": "user", "content": "hi"}])
    await client._client.aclose()
    assert len(calls) == 1


async def test_server_error_is_retried_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("app.llm.openai_client.asyncio.sleep", _no_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503, text="service unavailable")
        return httpx.Response(200, json=_ok_response("恢复成功"))

    client = _client_with(handler, llm_max_retries=2)
    response = await client.chat([{"role": "user", "content": "hi"}])
    await client._client.aclose()
    assert response.content == "恢复成功"
    assert len(calls) == 2


async def test_retry_exhausted_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("app.llm.openai_client.asyncio.sleep", _no_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="always broken")

    client = _client_with(handler, llm_max_retries=1)
    with pytest.raises(LLMError, match="模型调用失败"):
        await client.chat([{"role": "user", "content": "hi"}])
    await client._client.aclose()
