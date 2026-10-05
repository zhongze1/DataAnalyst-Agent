"""Agent 事件与 SSE 序列化。

事件是「Loop 内部状态」与「HTTP 流」之间的唯一契约。Agent 只管产出
:class:`AgentEvent`，不关心谁在消费；SSE 的格式化集中在这里。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# 事件类型。保持精简：前端只需按类型分发。
EVENT_START = "start"        # 会话开始
EVENT_TOOL_ARGS = "tool_args"  # 即将执行某个工具（让 UI 提前显示"正在…"）
EVENT_STEP = "step"          # 某个工具执行完毕
EVENT_TOKEN = "token"        # 最终答案的增量文本
EVENT_DONE = "done"          # 全部结束，携带完整快照
EVENT_ERROR = "error"        # 流中途失败


@dataclass
class AgentEvent:
    """一次事件。``kind`` 决定前端如何处理，``data`` 是 JSON 载荷。"""

    kind: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_sse(self) -> bytes:
        """序列化为一个 SSE 帧。

        载荷里的换行必须转义，否则会破坏帧边界（``data:`` 是逐行协议）。
        """
        payload = json.dumps(self.data, ensure_ascii=False, default=str)
        body = f"event: {self.kind}\ndata: {payload}\n\n"
        return body.encode("utf-8")


def done_event(
    *,
    answer: str,
    steps: list[dict[str, Any]],
    stop_reason: str,
    usage: dict[str, Any] | None = None,
) -> AgentEvent:
    """收尾事件：携带完整快照。

    流可能因网络抖动丢事件，前端以这个快照为准做最终校正。
    """
    return AgentEvent(
        kind=EVENT_DONE,
        data={
            "answer": answer,
            "steps": steps,
            "stop_reason": stop_reason,
            "usage": usage or None,
        },
    )


def error_event(code: str, message: str) -> AgentEvent:
    return AgentEvent(kind=EVENT_ERROR, data={"code": code, "message": message})
