"""DA-018 / RAG-05：把知识库检索注册成 Agent 工具 ``knowledge_search``。

为什么**不用 ``@tool`` 装饰器**（本工程唯一的例外）：

工具注册是「导入即副作用注册到模块级单例」。而 ``AGENT_KNOWLEDGE_ENABLED=false``
要求工具**根本不注册**（既不出现在 schema 里，也不出现在提示词里）—— 装饰器必须
在函数定义处无条件执行，做不到这件事。所以这里改成显式的
:func:`register_knowledge_tool`，由 ``app/tools/__init__.py`` 在导入时调用。

注册路径里**只允许返回 None 或正常返回，不得做任何 IO / 抛异常**：
它跑在 ``app.tools`` 的 import 期，抛错等于整个应用起不来。

复用既有横切设施（本工具有意"什么都不做"）：
- 注册进 ``ToolRegistry`` → 自动获得 DA-008 工具审计（只记形状不记内容）；
- 返回字符串 → 由 Agent Loop 统一出口套 DA-015 打码（喂模型的仍是原文）；
- 请求级配额与并发（DA-017）自动生效。
"""

from __future__ import annotations

from app.config import Settings, get_settings
from app.services.knowledge_service import KnowledgeSearchResult, get_knowledge_service
from app.tools.base import Tool, ToolContext
from app.tools.registry import ToolRegistry

TOOL_NAME = "knowledge_search"

# 输出预算给 base.format_value 的截断留的余量：宁可自己收敛，
# 也不要让截断把「来源」那一行切掉（模型失去出处就去编答案了）
_OUTPUT_RESERVE = 120
# 省略提示那一行先扣出来，这样最终输出才是**硬上限**（不是"大概不超过"）
_OMIT_RESERVE = 60
# 单条正文的最低配额；配额算到比它还小时，剩下的命中就不再展开正文
_MIN_SNIPPET = 80

_DESCRIPTION = (
    "检索**本项目自己的文档**（指标口径、架构说明、测试指南、DA-xxx 设计工单），"
    "返回带来源（文档名 + 章节 + 行号）的原文片段。\n"
    "**问题涉及本项目自身的口径 / 规则 / 架构 / 指标定义 / 已知限制 时，必须优先用它**"
    "（例如「销售额怎么定义」「敏感字段怎么打码」「配额规则」「为什么这么设计」）——"
    "它是项目文档的唯一权威来源，别用 run_pandas 去猜口径，也别让用户复述文档。\n"
    "它**只查项目文档，不查用户上传的 Excel 数据**（查数据请用其它工具）。\n"
    "返回「知识库中没有相关内容」时如实说明「文档里没查到」，严禁编造；"
    "引用时给出来源，不要粘贴大段原文。"
)


def build_knowledge_tool(settings: Settings) -> Tool:
    """按配置构造工具（schema 里的 top_k 上限取真实生效值）。"""
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "要查的问题或关键词，尽量用文档里的术语"
                    "（如「客单价口径」「敏感字段打码」「配额规则」）"
                ),
            },
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "maximum": settings.knowledge_top_k_max,
                "description": (
                    f"返回条数，默认 {settings.knowledge_top_k}，"
                    f"最多 {settings.knowledge_top_k_max}"
                ),
            },
        },
        "required": ["query"],
    }
    return Tool(
        name=TOOL_NAME,
        description=_DESCRIPTION,
        parameters=parameters,
        func=_run_knowledge_search,
        uses_active_frame=False,
    )


def register_knowledge_tool(
    registry: ToolRegistry, settings: Settings | None = None
) -> Tool | None:
    """按 ``AGENT_KNOWLEDGE_ENABLED`` 决定是否注册；返回注册好的工具或 None。"""
    resolved = settings or get_settings()
    if not resolved.knowledge_enabled:
        return None
    existing = registry.get(TOOL_NAME)
    if existing is not None:  # 幂等：重复 import 不该炸在「工具名重复」上
        return existing
    return registry.add(build_knowledge_tool(resolved))


# --------------------------------------------------------------------- 执行
def _run_knowledge_search(ctx: ToolContext, query: str, top_k: int | None = None) -> str:
    """工具入口。

    刻意**不读 ``ctx``**：知识库是全局的，与分析哪张表无关；也因此
    ``uses_active_frame=False``（不会偷偷改「当前 df」）。
    """
    settings = get_settings()
    result = get_knowledge_service().search(query, top_k=top_k)
    return render_result(result, settings)


# --------------------------------------------------------------------- 渲染
def _snippet(text: str, limit: int) -> str:
    """把片段正文压成一行引用，超出上限即省略 —— 保住来源比保住全文重要。"""
    flattened = " ".join(text.split())
    if len(flattened) <= limit:
        return f"> {flattened}"
    return f"> {flattened[:limit]} …（正文共 {len(flattened)} 字，已省略）"


def render_result(result: KnowledgeSearchResult, settings: Settings) -> str:
    """把检索结果渲染成给模型看的文本。

    两条硬约束：
    1. **来源清单前置**：即使正文被省略，模型也拿得到全部出处；
    2. **总长是硬上限**：正文配额按"剩余预算 ÷ 剩余命中数"分摊，
       不会出现"第一条太长把预算顶穿"（实测踩过：top_k=10 时超出 7%）。
    """
    if result.status == "unavailable":
        return f"[知识库检索 · 不可用] {result.detail}"

    mode_label = "混合模式" if result.mode == "hybrid" else "关键词模式"
    head = f"[知识库检索 · {mode_label}] 查询「{result.query}」"

    if result.status == "empty":
        return (
            f"{head} | 命中 0/{result.total_chunks} 片段\n"
            f"结论: 知识库中没有相关内容（{result.detail}）。\n"
            "不要编造：如实说明「文档里没查到」，再给出文档中能回答的最接近内容。"
        )

    sources = "；".join(
        f"{index}) {hit.doc_id} §{hit.section} 行{hit.line_start}-{hit.line_end}"
        for index, hit in enumerate(result.hits, 1)
    )
    lines = [f"{head} | 命中 {len(result.hits)}/{result.total_chunks} 片段", f"来源: {sources}"]
    lines.extend(f"说明: {note}" for note in result.notes)

    budget = max(
        200,
        min(
            settings.knowledge_max_chars,
            settings.max_tool_output_chars - _OUTPUT_RESERVE,
        ),
    )
    body_budget = budget - _OMIT_RESERVE
    used = sum(len(line) + 1 for line in lines)
    rendered = 0
    for index, hit in enumerate(result.hits, 1):
        remaining = body_budget - used
        if remaining <= 0:
            break
        hits_left = len(result.hits) - index + 1
        allowance = min(
            settings.knowledge_snippet_chars, max(_MIN_SNIPPET, remaining // hits_left)
        )
        block = (
            f"### {index}. [{hit.doc_id}] §{hit.section}"
            f" | 行 {hit.line_start}-{hit.line_end} | 分数 {hit.score:.4f}\n"
            f"为什么: {hit.why}\n"
            f"{_snippet(hit.text, allowance)}"
        )
        # 一条都还没写进去时兜底放第一条（否则模型什么都拿不到）；之后严格守预算
        if rendered and len(block) + 1 > remaining:
            break
        lines.append(block)
        used += len(block) + 1
        rendered += 1

    if rendered < len(result.hits):
        lines.append(
            f"…（为控制输出长度，其余 {len(result.hits) - rendered} 条命中的正文已省略；"
            "完整出处见上方「来源」；需要补全时针对具体小节再查一次即可）"
        )
    return "\n".join(lines)
