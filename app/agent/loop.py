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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from app.agent.events import (
    EVENT_START,
    EVENT_STEP,
    EVENT_TOKEN,
    EVENT_TOOL_ARGS,
    AgentEvent,
)
from app.agent.prompts import build_system_prompt
from app.config import Settings, get_settings
from app.llm.base import LLMClient, LLMError, LLMResponse
from app.logging_setup import describe_arguments
from app.security.redact import redact_text, redact_value
from app.storage.session_store import Session
from app.tools.base import ToolContext, humanize_tool_error, split_tool_error
from app.tools.registry import ToolRegistry, registry

logger = logging.getLogger(__name__)

# 事件回调：Agent 只管调用，是否吞掉异常由调用方决定
EventEmitter = Callable[[AgentEvent], Awaitable[None]]


def _as_steps(steps: list[ToolStep]) -> list[dict[str, Any]]:
    """ToolStep -> 可 JSON 序列化的 dict（落盘与接口共用一份形状）。"""
    return [
        {
            "step": s.step,
            "tool": s.tool,
            "arguments": s.arguments,
            "ok": s.ok,
            "result": s.result,
            "error": s.error,
        }
        for s in steps
    ]


USAGE_TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


def _merge_usage(acc: dict[str, Any], raw: dict[str, Any]) -> None:
    """把一次模型调用的 usage 累加进 ``acc``（原地修改）。

    只累加**确实出现且是非负整数**的 token 字段：厂商给字符串、浮点或
    干脆不给，都视为「没有这个字段」。宁可少算，也不把估算值
    伪装成精确 usage —— 费用口径失真比少几个数字危险得多。
    """
    for key in USAGE_TOKEN_KEYS:
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        acc[key] = acc.get(key, 0) + max(0, value)


def _finalize_usage(acc: dict[str, Any], calls: int, missing: int) -> dict[str, Any]:
    """收尾成对外可见的 usage。

    一轮 usage 都没拿到时返回 ``{}``（``usage or None`` 之后前端仍显示「暂无用量」，
    与改造前完全一致，不改变既有契约）；只要有一轮回报过，就带上轮次统计，
    让调用方知道这份数字的**完整度**。
    """
    if not acc:
        return {}
    return {
        **acc,
        "calls": calls,
        "usage_reported_calls": calls - missing,
        "usage_missing_calls": missing,
    }


async def _emit(on_event: EventEmitter | None, event: AgentEvent) -> None:
    """发送一个事件；没人在听时静默跳过。"""
    if on_event is None:
        return
    await on_event(event)

# 任务漂移告警阈值：工具调用次数超过此值，且最终答案里找不到提问的任何一个词，
# 说明模型很可能在答别的问题。只告警不阻断。
_DRIFT_STEP_THRESHOLD = 8
_TOKEN_MIN_LEN = 2


def _is_cjk(char: str) -> bool:
    """是否为汉字。

    不能简单用「码位 >= 0x2E80」：那会把全角标点（？！，U+FF00–U+FFEF）
    和 CJK 标点（U+3000–U+303F）也算进来，产出「？？」这类无意义候选词。
    这里只认 CJK 统一表意文字与扩展 A。
    """
    code = ord(char)
    return 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF


def _question_tokens(question: str) -> list[str]:
    """从提问里抽出可用于比对的候选片段。

    中文没有空格，无法按空白切词（整句会被当成一个 token，提取不出「张三」）。
    因此中文走 **2 字滑窗**：一个 11 字的问题能得到 10 个候选片段，
    只要答案里出现其中任何一个，就认为回答触及了问题。
    代价是可能误报（答案里偶然出现「一下」这类片段）—— 所以只写日志不阻断。
    """
    tokens: list[str] = []
    ascii_buf: list[str] = []
    cjk_buf: list[str] = []

    def flush_ascii() -> None:
        if len(ascii_buf) >= _TOKEN_MIN_LEN:
            tokens.append("".join(ascii_buf))
        ascii_buf.clear()

    def flush_cjk() -> None:
        for i in range(len(cjk_buf) - _TOKEN_MIN_LEN + 1):
            tokens.append("".join(cjk_buf[i : i + _TOKEN_MIN_LEN]))
        cjk_buf.clear()

    for char in question:
        if char.isascii() and (char.isalnum() or char == "_"):
            flush_cjk()
            ascii_buf.append(char)
        elif _is_cjk(char):
            flush_ascii()
            cjk_buf.append(char)
        else:
            flush_ascii()
            flush_cjk()
    flush_ascii()
    flush_cjk()
    return tokens


def detect_task_drift(question: str, answer: str, step_count: int) -> str | None:
    """检测「答非所问」：返回告警描述，正常时返回 None。

    这是观测手段，不是正确性保证 —— 关键词匹配会有误报，所以只写日志。
    """
    if step_count < _DRIFT_STEP_THRESHOLD:
        return None
    tokens = _question_tokens(question)
    if not tokens:
        return None
    if any(token in answer for token in tokens):
        return None
    return (
        f"疑似任务漂移：调用 {step_count} 次工具，但答案里没有出现提问中的任何一个词"
        f"（关键词: {', '.join(tokens[:6])}）"
    )


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
    async def run(
        self,
        session: Session | None,
        question: str,
        on_event: EventEmitter | None = None,
    ) -> AgentResult:
        """处理一次用户提问。``session`` 为 None 表示用户还没上传文件。

        Args:
            on_event: 可选回调，用于流式输出。Agent 只负责产出事件、不关心谁在
                消费；LLM 不支持流式时自动降级为一次性返回，行为完全一致。
        """
        started = time.perf_counter()
        if session is None:
            return self._finish(
                AgentResult(
                    answer="还没有检测到已上传的 Excel 文件。请先上传文件，我才能开始分析。",
                    stop_reason="no_file",
                ),
                started,
                session,
                [],
                question,
            )

        logger.info(
            "Agent Loop 开始 | 问题 %d 字 | 可用工具 %d 个",
            len(question),
            len(self._tools.names()),
        )
        await _emit(
            on_event,
            AgentEvent(
                EVENT_START,
                {"question_len": len(question), "tools": len(self._tools.names())},
            ),
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
        # 用量必须**逐轮累加**：改造前只保留最后一次模型调用的 usage，
        # 而一次提问往往要调用 3~12 轮模型，核算费用时会严重少算
        usage_acc: dict[str, Any] = {}
        model_calls = 0
        usage_missing = 0
        stop_reason = "max_steps"

        for turn in range(1, self._settings.max_agent_steps + 1):
            try:
                # 模型推理（网络 IO，异步）。有人在听事件时才走流式。
                response = await self._call_model(working, on_event, streaming=on_event is not None)
            except LLMError as exc:
                # 原始异常可能含厂商响应体/端点等内部信息，只进日志；
                # 用户看到的是脱敏文案，排查靠前端展示的 request_id
                logger.error(
                    "模型调用失败 | %s: %s",
                    type(exc).__name__,
                    exc,
                    extra={"error_code": "LLM_SERVICE_ERROR"},
                )
                session.messages = working[1:]  # 仅保留完整的历史 + 本次提问
                return self._finish(
                    AgentResult(
                        answer=(
                            "模型服务调用失败，暂时无法完成分析。"
                            "服务端已记录详细原因，请稍后重试。"
                        ),
                        steps=steps,
                        stop_reason="llm_error",
                        # 失败前已经花掉的用量也要如实上报，不能因为它失败了就吞掉
                        usage=_finalize_usage(usage_acc, model_calls, usage_missing),
                    ),
                    started,
                    session,
                    _as_steps(steps),
                    question,
                )

            model_calls += 1
            if response.usage:
                _merge_usage(usage_acc, response.usage)
            else:
                usage_missing += 1
            working.append(response_to_assistant_message(response))

            if not response.has_tool_calls:
                session.messages = working[1:]
                return self._finish(
                    AgentResult(
                        answer=response.content or "（模型未返回文本内容）",
                        steps=steps,
                        stop_reason="final",
                        usage=_finalize_usage(usage_acc, model_calls, usage_missing),
                    ),
                    started,
                    session,
                    _as_steps(steps),
                    question,
                )

            # 执行工具：pandas 是阻塞型 CPU 任务，放到线程池避免卡住事件循环
            for call in response.tool_calls:
                step_no = len(steps) + 1
                # 先发 tool_args：UI 立刻显示「正在执行 X」，不必等它跑完
                await _emit(
                    on_event,
                    AgentEvent(
                        EVENT_TOOL_ARGS,
                        {
                            "step": step_no,
                            "tool": call.name,
                            "args_summary": describe_arguments(call.arguments),
                        },
                    ),
                )
                # 工具可能需要按步骤记录产物（如导出用的结构化结果）
                ctx.current_step = step_no
                started_at = time.perf_counter()
                ok, output = await asyncio.to_thread(
                    self._tools.run, call.name, call.arguments, ctx
                )
                # 双通路：模型拿到带细节的原始错误（能自纠），用户拿人话。
                # 原始错误只在这里落日志，不再进 HTTP 响应。
                error_text: str | None = None
                if not ok:
                    code, _raw = split_tool_error(output)
                    error_text = humanize_tool_error(code)
                    logger.info(
                        "工具失败已转为人话 | tool=%s | code=%s",
                        call.name,
                        code,
                    )
                steps.append(
                    ToolStep(
                        step=len(steps) + 1,
                        tool=call.name,
                        # DA-015：对外展示与落库的都是**脱敏值**（界面 steps、
                        # SSE done 事件、导出轨迹共用这一个来源）。
                        # 注意喂给模型的 output 仍是原文 —— 见下面 working.append
                        arguments=redact_value(call.arguments),
                        ok=ok,
                        result=redact_text(output),
                        error=error_text,
                    )
                )
                await _emit(
                    on_event,
                    AgentEvent(
                        EVENT_STEP,
                        {
                            "step": step_no,
                            "tool": call.name,
                            "ok": ok,
                            "duration_ms": round((time.perf_counter() - started_at) * 1000, 1),
                            "result_chars": len(output),
                            "error_code": None if ok else split_tool_error(output)[0],
                        },
                    ),
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
            # 收尾调用同样真实产生费用，必须一起计入，否则每轮触顶的会话都会少算一轮
            model_calls += 1
            if final.usage:
                _merge_usage(usage_acc, final.usage)
            else:
                usage_missing += 1
            answer = final.content or "已达到最大调用次数，无法完成分析。"
            # 收尾结论必须**一起落进会话历史**：早前只写 working[1:]，而 working 里
            # 并没有 final 这条 assistant 消息 —— 于是用户刚看到的结论在刷新/导出/
            # 下一轮追问时全都不见了，且历史以 role=tool 结尾（部分厂商下一轮直接 400）。
            session.messages = working[1:] + [
                {"role": "assistant", "content": answer}
            ]
        except LLMError as exc:  # pragma: no cover
            # 收尾也失败：丢弃半截的工具轨迹，避免污染下一轮。
            # 注意**不能**把 exc 拼进答案：它可能带着厂商端点、响应体等内部信息，
            # 与上面 274-276 行「原始异常只进日志」的策略保持一致。
            logger.error(
                "收尾调用失败 | %s: %s",
                type(exc).__name__,
                exc,
                extra={"error_code": "LLM_SERVICE_ERROR"},
            )
            answer = "已达到最大调用次数，且未能取得阶段性结论。请重试或换一种问法。"
            session.messages = working[1 : 1 + len(session.messages)]

        return self._finish(
            AgentResult(
                answer=answer,
                steps=steps,
                stop_reason=stop_reason,
                usage=_finalize_usage(usage_acc, model_calls, usage_missing),
            ),
            started,
            session,
            _as_steps(steps),
            question,
        )

    # ------------------------------------------------------------------ 内部
    def _save(self, session: Session | None, steps: list[dict[str, Any]] | None) -> None:
        """把这一轮对话落盘。落库失败只记日志，不能让用户白等一场。"""
        if session is None or not session.messages:
            return
        try:
            self._store.save_turn(session, steps)
        except Exception:
            logger.exception("会话落盘失败 | session=%s", session.session_id)

    async def _call_model(
        self,
        messages: list[dict[str, Any]],
        on_event: EventEmitter | None,
        *,
        streaming: bool,
    ) -> LLMResponse:
        """调用模型，必要时流式推送增量文本。

        流式能力是**可选**的：客户端不实现 ``chat_stream`` 时退化为一次性调用，
        并补发一个 token 事件，保证前端在两种路径下表现一致。
        """
        schemas = self._tools.openai_schemas()
        stream = getattr(self._llm, "chat_stream", None) if streaming else None

        if stream is None:
            response = await self._llm.chat(messages, schemas)
            if on_event is not None and response.content:
                await _emit(on_event, AgentEvent(EVENT_TOKEN, {"text": response.content}))
            return response

        final: LLMResponse | None = None
        async for event in stream(messages, schemas):
            if event.kind == "delta":
                await _emit(on_event, AgentEvent(EVENT_TOKEN, {"text": event.text}))
            elif event.kind == "done":
                final = event.response
        if final is None:
            raise LLMError("流式响应结束但未返回完整结果")
        return final

    def _finish(
        self,
        result: AgentResult,
        started: float,
        session: Session | None = None,
        steps: list[dict[str, Any]] | None = None,
        question: str = "",
    ) -> AgentResult:
        """统一的收尾出口：所有 return 都经过这里，保证一定留下耗时与用量日志。"""
        self._save(session, steps)
        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        failed = sum(1 for step in result.steps if not step.ok)
        if question:
            drift = detect_task_drift(question, result.answer, len(result.steps))
            if drift:
                logger.warning("任务漂移告警 | %s", drift, extra={"error_code": "TASK_DRIFT"})
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
        # DA-015：模型可能把手机号直接写进结论，对外的答案必须脱敏。
        # 放在最后一步做：上面的漂移检测要拿模型**真实输出的原文**去比对提问关键词，
        # 提前打码会把它变成误判。session.messages 里存的也仍是原文（下一轮要喂给模型）。
        result.answer = redact_text(result.answer)
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
