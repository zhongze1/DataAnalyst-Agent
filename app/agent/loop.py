"""单 Agent Loop —— 本项目的核心。

流程（ReAct / function-calling 范式）：

    用户问题
      ↓
    ┌─→ 调模型 ─→ 模型要调工具？ ──是──→ 执行工具（pandas）→ 结果回传 → 调模型（循环）
      │                    │
      │                   否
      │                    ↓
      └────────────  最终答案返回前端

关键设计：
- 循环上限 ``max_agent_steps``，防止模型无限调工具；
- 工具异常不中断循环，错误信息回传给模型让它自我纠正；
- 每一步都记录成 ``ToolStep``，前端可完整展示「Agent 的思考过程」。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app.agent.prompts import build_system_prompt
from app.config import Settings, get_settings
from app.llm.base import LLMClient, LLMError
from app.storage.session_store import Session
from app.tools.base import ToolContext
from app.tools.registry import ToolRegistry, registry

logger = logging.getLogger(__name__)


@dataclass
class ToolStep:
    step: int
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    result: str = ""
    error: str | None = None


@dataclass
class AgentResult:
    answer: str
    steps: list[ToolStep] = field(default_factory=list)
    stop_reason: str = "final"
    usage: dict[str, Any] = field(default_factory=dict)


class DataAnalystAgent:
    """把「模型 + 工具 + 会话」编排成一个可循环执行的 Agent。"""

    def __init__(
        self,
        llm: LLMClient,
        store: Any,
        tools: ToolRegistry | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._llm = llm
        self._store = store
        self._tools = tools or registry
        self._settings = settings or get_settings()

    # ------------------------------------------------------------------ 对外
    async def run(self, session: Session | None, question: str) -> AgentResult:
        """处理一次用户提问。``session`` 为 None 表示用户还没上传文件。"""
        started = time.perf_counter()
        if session is None:
            return self._finish(
                AgentResult(
                    answer="还没有检测到已上传的 Excel 文件。请先上传文件，我才能开始分析。",
                    stop_reason="no_file",
                ),
                started,
            )

        logger.info(
            "Agent Loop 开始 | 问题 %d 字 | 可用工具 %d 个",
            len(question),
            len(self._tools.names()),
        )

        ctx = ToolContext(
            session_id=session.session_id,
            file_path=str(session.path),
            file_name=session.filename,
            sheets=list(session.sheets),
            frames=session.frames,  # 复用会话内缓存的 DataFrame
        )

        system = build_system_prompt(session, self._tools.names())
        session.messages.append({"role": "user", "content": question})
        working: list[dict[str, Any]] = [{"role": "system", "content": system}, *session.messages]

        steps: list[ToolStep] = []
        usage: dict[str, Any] = {}
        stop_reason = "max_steps"

        for turn in range(1, self._settings.max_agent_steps + 1):
            try:
                # 模型推理（网络 IO，异步）
                response = await self._llm.chat(working, self._tools.openai_schemas())
            except LLMError as exc:
                logger.warning("模型调用失败: %s", exc)
                session.messages = working[1:]  # 仅保留完整的历史 + 本次提问
                return self._finish(
                    AgentResult(
                        answer=f"模型调用失败：{exc}",
                        steps=steps,
                        stop_reason="llm_error",
                    ),
                    started,
                )

            usage = response.usage or usage
            working.append(response_to_assistant_message(response))

            if not response.has_tool_calls:
                session.messages = working[1:]
                return self._finish(
                    AgentResult(
                        answer=response.content or "（模型未返回文本内容）",
                        steps=steps,
                        stop_reason="final",
                        usage=usage,
                    ),
                    started,
                )

            # 执行工具：pandas 是阻塞型 CPU 任务，放到线程池避免卡住事件循环
            for call in response.tool_calls:
                ok, output = await asyncio.to_thread(
                    self._tools.run, call.name, call.arguments, ctx
                )
                steps.append(
                    ToolStep(
                        step=len(steps) + 1,
                        tool=call.name,
                        arguments=call.arguments,
                        ok=ok,
                        result=output,
                        error=None if ok else output,
                    )
                )
                working.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": output,
                    }
                )

        # 达到步数上限：让模型基于已有结果收尾，而不是直接丢弃
        try:
            final = await self._llm.chat(
                working
                + [
                    {
                        "role": "user",
                        "content": "已达到最大工具调用次数，请停止继续调用工具，"
                        "基于目前已获得的数据给出阶段性结论，并说明哪些部分未完成。",
                    }
                ]
            )
            answer = final.content or "已达到最大调用次数，无法完成分析。"
            session.messages = working[1:]
        except LLMError as exc:  # pragma: no cover
            # 收尾也失败：只保留用户提问，丢弃半截的工具轨迹，避免污染下一轮
            answer = f"已达到最大调用次数，且收尾请求失败：{exc}"
            session.messages = working[1 : 1 + len(session.messages)]

        return self._finish(
            AgentResult(answer=answer, steps=steps, stop_reason=stop_reason, usage=usage),
            started,
        )

    # ------------------------------------------------------------------ 内部
    @staticmethod
    def _finish(result: AgentResult, started: float) -> AgentResult:
        """统一的收尾出口：所有 return 都经过这里，保证一定留下耗时与用量日志。"""
        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        failed = sum(1 for step in result.steps if not step.ok)
        logger.info(
            "Agent Loop 结束 | 工具 %d 次(失败 %d) | 终止原因 %s | %.1fms",
            len(result.steps),
            failed,
            result.stop_reason,
            duration_ms,
            extra={
                "steps": len(result.steps),
                "failed_steps": failed,
                "stop_reason": result.stop_reason,
                "duration_ms": duration_ms,
                "usage": result.usage or None,
            },
        )
        return result

    # ------------------------------------------------------------------ 内部


def response_to_assistant_message(response: Any) -> dict[str, Any]:
    """把 LLMResponse 转回 OpenAI assistant 消息格式。"""
    message: dict[str, Any] = {"role": "assistant", "content": response.content or None}
    if response.has_tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in response.tool_calls
        ]
    return message
