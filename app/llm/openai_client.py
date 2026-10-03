"""OpenAI 兼容的 HTTP 客户端实现。

只依赖 httpx，不绑定任何厂商 SDK，因此 DeepSeek / Moonshot / OpenRouter /
vLLM / Ollama 等所有 OpenAI 兼容端点都能直接用，改 ``AGENT_LLM_BASE_URL`` 即可。
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from app.config import Settings
from app.llm.base import LLMError, LLMResponse, ToolCall, parse_tool_arguments


class OpenAICompatClient:
    """调用 ``POST {base_url}/chat/completions``。"""

    def __init__(self, settings: Settings) -> None:
        if not settings.llm_api_key:
            raise LLMError(
                "未配置 AGENT_LLM_API_KEY。复制 .env.example 为 .env 填入 Key，"
                "或设置 AGENT_LLM_PROVIDER=mock 用本地假模型跑通流程。"
            )
        self._settings = settings
        self._client = httpx.AsyncClient(timeout=settings.llm_timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": messages,
            "temperature": self._settings.llm_temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        last_error: Exception | None = None
        for attempt in range(self._settings.llm_max_retries + 1):
            try:
                response = await self._client.post(
                    f"{self._settings.llm_base_url.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._settings.llm_api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                if response.status_code >= 400:
                    # 4xx 多为参数/鉴权问题，重试无意义
                    if 400 <= response.status_code < 500 and response.status_code != 429:
                        raise LLMError(
                            f"模型接口返回 {response.status_code}: {response.text[:500]}"
                        )
                    raise LLMError(f"模型接口返回 {response.status_code}: {response.text[:500]}")
                return self._parse(response.json())
            except (httpx.HTTPError, LLMError) as exc:
                last_error = exc
                if isinstance(exc, LLMError) and "返回 4" in str(exc):
                    break
                if attempt < self._settings.llm_max_retries:
                    await asyncio.sleep(1.5 * (attempt + 1))

        raise LLMError(f"模型调用失败: {last_error}") from last_error

    @staticmethod
    def _parse(data: dict[str, Any]) -> LLMResponse:
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"模型返回为空: {data}")
        message = choices[0].get("message") or {}

        tool_calls: list[ToolCall] = []
        for item in message.get("tool_calls") or []:
            function = item.get("function") or {}
            name = function.get("name")
            if not name:
                continue
            tool_calls.append(
                ToolCall(
                    id=item.get("id") or f"call_{len(tool_calls)}",
                    name=name,
                    arguments=parse_tool_arguments(function.get("arguments")),
                )
            )

        return LLMResponse(
            content=(message.get("content") or "").strip(),
            tool_calls=tool_calls,
            usage=data.get("usage") or {},
        )
