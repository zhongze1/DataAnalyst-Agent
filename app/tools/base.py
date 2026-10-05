"""工具抽象：Tool 协议、执行上下文、结果格式化。

一个「工具」= 名字 + 描述 + JSON Schema 参数 + 同步执行函数。
工具执行结果统一格式化为**字符串**回传给模型（模型只能读文本）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from app.config import get_settings


class ToolError(Exception):
    """工具执行失败。

    注意：这类异常**不会中断 Agent Loop**，错误信息会作为工具结果回传给模型，
    让模型自行纠正（例如 sheet 名写错时改用 list_sheets 重新选择）。

    ``code`` 是错误分类。原始 message 只给模型（它需要细节才能自纠），
    用户看到的是 :func:`humanize_tool_error` 映射出的人话。
    """

    def __init__(self, message: str, *, code: str = "TOOL_ERROR") -> None:
        super().__init__(message)
        self.code = code


@dataclass
class ToolContext:
    """一次提问过程中，Agent 持有的数据上下文。

    ``frames`` 以「工作表名」为键缓存已加载的 DataFrame，避免重复读盘。

    ``active_sheet`` 是「模型当前正在分析哪张表」。它**只应由显式意图改变**
    （``load_sheet`` 或 ``run_pandas(sheet=...)``）——只读工具若偷偷改写它，
    模型会在体检完其他表后发现自己写的代码作用在错的表上。
    """

    session_id: str
    file_path: str | None = None
    file_name: str | None = None
    sheets: list[str] = field(default_factory=list)
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    active_sheet: str | None = None

    def get_frame(self, sheet: str | None = None, *, activate: bool = False) -> pd.DataFrame:
        """取出（必要时加载）某个工作表的 DataFrame。

        Args:
            sheet: 目标工作表；``None`` 表示用当前工作表。
            activate: 是否把该表设为「当前工作表」。只读工具必须传 ``False``，
                否则会产生副作用：连续体检三张表后，模型后续的 ``df`` 会指向
                最后一张，而它以为还是第一张。
        """
        from app.storage.session_store import SessionStore  # 延迟导入避免循环依赖

        name = sheet or self.active_sheet or (self.sheets[0] if self.sheets else None)
        if not name:
            raise ToolError("尚未加载任何数据，请先调用 load_sheet。", code="NO_DATA_LOADED")
        if name not in self.frames:
            if self.sheets and name not in self.sheets:
                raise ToolError(
                    f"工作表 {name!r} 不存在。可用工作表: {', '.join(self.sheets) or '无'}",
                    code="SHEET_NOT_FOUND",
                )
            self.frames[name] = SessionStore.read_frame(self.file_path, name)
        # 首次使用时初始化是必要的（否则模型无从知道 df 是哪张表）；
        # 之后只有显式 activate 才允许切换。
        if activate or self.active_sheet is None:
            self.active_sheet = name
        return self.frames[name]

    def describe_active_frame(self) -> str:
        """生成「当前 df」标记，追加到工具输出里。

        没有它，模型无法知道 ``run_pandas`` 里的 ``df`` 到底指向哪张表 ——
        这正是「KeyError: 销售额」这类误判的根源。
        """
        name = self.active_sheet
        if name is None:
            return "[当前 df = 未加载]"
        frame = self.frames.get(name)
        if frame is None:
            return f"[当前 df = {name}]"
        rows, cols = frame.shape
        return f"[当前 df = {name}（{rows:,} 行 × {cols} 列）]"


@dataclass
class Tool:
    """可被模型调用的函数。

    ``uses_active_frame`` 标记该工具是否读取「当前 df」。为 True 时，
    ``run()`` 会自动在输出首行加上当前工作表标记 —— 由框架保证，
    不依赖每个工具作者自觉。
    """

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    uses_active_frame: bool = False

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def run(self, ctx: ToolContext, arguments: dict[str, Any] | None = None) -> str:
        """校验参数 → 执行 → 格式化结果。任何失败都转成 ToolError 文案。"""
        arguments = arguments or {}
        self._validate(arguments)
        value = self.func(ctx, **arguments)
        text = format_value(value)
        if self.uses_active_frame:
            return f"{ctx.describe_active_frame()}\n{text}"
        return text

    def _validate(self, arguments: dict[str, Any]) -> None:
        schema = self.parameters or {}
        properties: dict[str, Any] = schema.get("properties", {})
        required: list[str] = schema.get("required", [])

        unknown = set(arguments) - set(properties)
        if unknown:
            raise ToolError(
                f"工具 {self.name} 不支持参数 {sorted(unknown)}；"
                f"可用参数: {sorted(properties) or '无'}"
            )
        missing = [key for key in required if arguments.get(key) in (None, "")]
        if missing:
            raise ToolError(f"工具 {self.name} 缺少必填参数: {missing}")

        for key, spec in properties.items():
            if key not in arguments:
                continue
            expected = spec.get("type")
            value = arguments[key]
            if expected == "string" and not isinstance(value, str):
                raise ToolError(f"参数 {key} 应为字符串，收到 {type(value).__name__}")
            if expected == "integer" and isinstance(value, bool):
                raise ToolError(f"参数 {key} 应为整数")


# ----------------------------------------------------------------- 错误双通路
# 工具失败时，模型收到带细节的原始错误（能自我纠正），
# 用户收到这里映射出的人话（看得懂、能行动）。原始错误只进日志。
_USER_HINTS: dict[str, str] = {
    "SHEET_NOT_FOUND": "数据里没有这个工作表。换一个表名，或先用「工作表列表」看看有哪些表。",
    "COLUMN_NOT_FOUND": "数据里没有这个列名。请对照「列画像」里的实际列名后再试。",
    "COLUMN_MISSING": "代码里引用的列在当前表中不存在。请看工具输出首行的「当前 df」与列名清单。",
    "NO_COLUMNS": "没有指定任何列，无法分析。",
    "NO_DATA_LOADED": "还没有加载数据。请先上传 Excel 文件。",
    "SANDBOX_VIOLATION": "生成的代码触及安全限制（不能读写文件），已阻止执行。",
    "CODE_TOO_LONG": "生成的代码太长，请拆成几步分别计算。",
    "SYNTAX_ERROR": "生成的代码有语法错误，请检查括号与引号是否配对。",
    "UNKNOWN_TOOL": "请求了一个不存在的工具。",
    "RUNTIME_ERROR": "这一步计算出错了。可以换个问法或补充说明后重试。",
    "TOOL_ERROR": "这一步没能完成。可以换个问法或补充说明后重试。",
}

_ERROR_PREFIX = re.compile(r"^\[([A-Z_]+)\]\s*(.*)$", re.DOTALL)


def encode_tool_error(code: str, detail: str) -> str:
    """给工具错误加机器可读前缀。模型看到无伤（多一个分类标签）。"""
    return f"[{code}] {detail}"


def split_tool_error(text: str) -> tuple[str, str]:
    """把 ``[CODE] detail`` 拆回 (code, detail)；不是该格式则原样归类。"""
    match = _ERROR_PREFIX.match(text)
    if match:
        return match.group(1), match.group(2)
    return "TOOL_ERROR", text


def humanize_tool_error(code: str) -> str:
    """内部错误码 -> 面向用户的中文提示。"""
    return _USER_HINTS.get(code, _USER_HINTS["TOOL_ERROR"])


def format_value(value: Any) -> str:
    """把任意执行结果转成适合喂给模型的文本（并做长度截断）。"""
    if isinstance(value, str):
        text = value
    elif isinstance(value, pd.DataFrame):
        text = _format_frame(value)
    elif isinstance(value, pd.Series):
        text = f"[Series] len={len(value)}\n{value.head(20).to_string()}"
    elif isinstance(value, dict):
        text = "\n".join(f"{k}: {v}" for k, v in value.items())
    else:
        text = str(value)
    return _truncate(text)


def _format_frame(df: pd.DataFrame) -> str:
    rows = df.head(20)
    header = f"[DataFrame] shape={df.shape} | columns={list(df.columns)}"
    if df.empty:
        return f"{header}\n(空表)"
    return f"{header}\n{rows.to_string()}"


def _truncate(text: str) -> str:
    limit = get_settings().max_tool_output_chars
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... [内容过长已截断，共 {len(text)} 字符]"
