"""DA-018：``knowledge_search`` 工具本身（全程离线）。

覆盖：schema 与 top_k 上限（验收 9）、关闭开关时根本不注册（验收 14）、
混合 / 关键词两种输出形态（验收 4、5）、工件缺失与损坏时不打断 Loop（验收 7）、
无关提问的空结果（验收 8）、输出预算与来源前置，以及
**DA-008 审计 + DA-015 打码两条既有设施自动生效**（验收 11）。
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.agent.loop import DataAnalystAgent
from app.config import Settings
from app.services import knowledge_service as ks
from app.tools.base import ToolContext
from app.tools.knowledge_tools import TOOL_NAME, register_knowledge_tool
from app.tools.registry import ToolRegistry
from stubs import (
    KNOWLEDGE_CHUNKS,
    FakeQueryEmbedder,
    KnowledgeLLM,
    knowledge_row,
    write_knowledge_artifacts,
)

RAW_PHONE = "13800138000"
MASKED_PHONE = "138****8000"


# ----------------------------------------------------------------- 夹具
@pytest.fixture(autouse=True)
def forbid_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """验收 17：任何用例真去联网都必须立刻炸出来。"""
    real_client = httpx.Client

    def guarded(*args: object, **kwargs: object) -> httpx.Client:
        if "transport" not in kwargs:
            raise AssertionError("测试禁止构造会联网的 httpx.Client")
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", guarded)


@pytest.fixture
def artifacts(settings: Settings) -> Path:
    """把合成索引写到 conftest 指定的 tmp 路径（绝不碰仓库 data/）。"""
    return write_knowledge_artifacts(Path(settings.knowledge_index_path).parent, KNOWLEDGE_CHUNKS)


def _registry(settings: Settings) -> ToolRegistry:
    registry = ToolRegistry()
    assert register_knowledge_tool(registry, settings) is not None
    return registry


def _run(registry: ToolRegistry, **arguments: object) -> tuple[bool, str]:
    return registry.run(TOOL_NAME, arguments, ToolContext(session_id="kb-test"))


# ============================================================ schema（验收 9）
def test_schema_exposes_query_and_topk_ceiling(settings: Settings) -> None:
    entry = next(
        item
        for item in _registry(settings).openai_schemas()
        if item["function"]["name"] == TOOL_NAME
    )
    assert entry["type"] == "function"
    assert "项目自己的文档" in entry["function"]["description"]
    assert "不查用户上传的 Excel" in entry["function"]["description"]

    params = entry["function"]["parameters"]
    assert params["required"] == ["query"]
    assert params["properties"]["query"]["type"] == "string"
    top_k = params["properties"]["top_k"]
    assert top_k["type"] == "integer"
    assert top_k["minimum"] == 1
    assert top_k["maximum"] == settings.knowledge_top_k_max


# ============================================================ 开关（验收 14）
def test_disabled_switch_registers_nothing(settings: Settings) -> None:
    off = settings.model_copy(update={"knowledge_enabled": False})
    registry = ToolRegistry()
    assert register_knowledge_tool(registry, off) is None
    assert registry.names() == []
    assert registry.openai_schemas() == []


def test_registration_is_idempotent(settings: Settings) -> None:
    """重复注册不该炸在「工具名重复」上（import 期重复触发过一次就完了）。"""
    registry = ToolRegistry()
    first = register_knowledge_tool(registry, settings)
    assert register_knowledge_tool(registry, settings) is first
    assert registry.names() == [TOOL_NAME]


# ============================================================ 正常输出（验收 4、5）
def test_keyword_mode_is_labelled_with_source_and_why(settings: Settings, artifacts: Path) -> None:
    ok, text = _run(_registry(settings), query="敏感字段怎么打码")
    assert ok is True
    assert text.startswith("[知识库检索 · 关键词模式] 查询「敏感字段怎么打码」")
    assert "命中 1/3" in text
    assert "来源: 1) DA-015-敏感字段打码 §打码规则 行1-3" in text
    assert "为什么:" in text and "BM25" in text
    assert "分数" in text
    # 关键词降级必须**明说**（验收 5）
    assert "关键词模式" in text
    assert RAW_PHONE in text, "喂给模型的必须是原文（打码只发生在 Loop 出口）"


def test_hybrid_mode_output(settings: Settings, artifacts: Path, monkeypatch: pytest.MonkeyPatch, patch_settings) -> None:
    hybrid = settings.model_copy(
        update={
            "embedding_base_url": "https://vector.example/v1",
            "embedding_api_key": "test-key",
            "embedding_model": "test-embedding",
        }
    )
    fake = FakeQueryEmbedder([1.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(ks, "HttpQueryEmbedder", lambda settings: fake)
    patch_settings(hybrid)

    ok, text = _run(_registry(hybrid), query="红烧肉做法")
    assert ok is True
    assert "[知识库检索 · 混合模式]" in text
    assert fake.calls == 1
    assert "RRF" in text
    assert "DA-015-敏感字段打码" in text
    assert "向量#1" in text


def test_output_stays_within_budget_and_keeps_sources(settings: Settings) -> None:
    """长正文只截片段，来源清单必须完整留在被截掉之前。"""
    long_text = "敏感字段怎么打码：" + "细节说明" * 200 + "（完）"
    write_knowledge_artifacts(
        Path(settings.knowledge_index_path).parent,
        [knowledge_row("c1", "长文档", "打码", long_text, [1.0, 0.0, 0.0, 0.0])],
    )
    budget = max(200, min(settings.knowledge_max_chars, settings.max_tool_output_chars - 120))

    ok, text = _run(_registry(settings), query="敏感字段怎么打码")
    assert ok is True
    assert len(text) <= budget
    assert "来源: 1) 长文档 §打码 行1-3" in text  # 来源前置，先于正文
    assert "已省略" in text
    assert settings.knowledge_snippet_chars < len(long_text)


def test_budget_is_a_hard_cap_even_at_top_k_max(settings: Settings, patch_settings) -> None:
    """命中多、来源清单长时，也不能把输出预算顶穿（实测踩过：超出 7%）。"""
    rows = [
        knowledge_row(
            f"c{index}",
            f"文档{index}",
            f"第 {index} 节",
            "配额规则说明：" + "细节" * 300,
            [1.0, 0.0, 0.0, 0.0],
        )
        for index in range(1, 11)
    ]
    write_knowledge_artifacts(Path(settings.knowledge_index_path).parent, rows)
    # 阈值降到 0：本用例要的是"命中很多条"，不是"分数够不够"
    loose = settings.model_copy(
        update={"knowledge_min_keyword_score": 0.0, "knowledge_top_k_max": 10}
    )
    patch_settings(loose)
    budget = max(200, min(loose.knowledge_max_chars, loose.max_tool_output_chars - 120))

    ok, text = _run(_registry(loose), query="配额规则", top_k=10)
    assert ok is True
    assert len(text) <= budget
    assert "命中 10/10" in text, "来源清单必须完整（就是要它能撑过正文省略）"
    assert "### 1." in text, "至少要有第一条的正文，否则模型什么都拿不到"
    assert "为控制输出长度" in text or text.count("### ") == 10


# ============================================================ 空结果与异常（验收 7、8）
def test_irrelevant_query_says_no_content_and_forbids_fabrication(settings: Settings, artifacts: Path) -> None:
    ok, text = _run(_registry(settings), query="推荐几部好看的电影")
    assert ok is True, "「没查到」是有价值的答案，不该当成工具错误"
    assert "命中 0/3" in text
    assert "没有相关内容" in text
    assert "不要编造" in text


def test_missing_artifact_tells_you_how_to_build_it(settings: Settings) -> None:
    ok, text = _run(_registry(settings), query="任意问题")
    assert ok is True
    assert "不可用" in text
    assert "未构建" in text
    assert "rag_lab.export_index" in text


def test_corrupt_index_is_contained(settings: Settings) -> None:
    Path(settings.knowledge_index_path).parent.mkdir(parents=True, exist_ok=True)
    Path(settings.knowledge_index_path).write_text("{ 坏掉的 json\n", encoding="utf-8")

    ok, text = _run(_registry(settings), query="任意问题")
    assert ok is True
    assert "不可用" in text


def test_bad_arguments_still_fail_loudly(settings: Settings, artifacts: Path) -> None:
    """参数错误走既有校验通道（ok=False），与"知识库不可用"是两回事。"""
    registry = _registry(settings)
    ok, text = _run(registry, top_k=3)
    assert ok is False
    assert "缺少必填参数" in text

    ok, text = _run(registry, query="打码", unknown=1)
    assert ok is False
    assert "不支持参数" in text


# ============================================================ 既有设施自动生效（验收 11）
async def test_audit_records_shape_and_step_output_is_redacted(
    settings: Settings, artifacts: Path, session, store
) -> None:
    """一次真实会话提问：审计只记形状，步骤结果已打码，模型仍拿到原文。"""
    local = ToolRegistry()
    register_knowledge_tool(local, settings)
    llm = KnowledgeLLM("敏感字段怎么打码")
    agent = DataAnalystAgent(llm=llm, store=store, tools=local, settings=settings)

    result = await agent.run(session, "项目文档里敏感字段是怎么打码的？")

    assert result.steps and result.steps[0].tool == TOOL_NAME
    step = result.steps[0]
    assert step.ok is True

    # ① 对外（界面 / SSE / 落库）：敏感值已打码（DA-015 在 Loop 统一出口生效）
    assert RAW_PHONE not in step.result
    assert MASKED_PHONE in step.result

    # ② 对模型：工具输出仍是原文，否则它会拿着 138****8000 去比对
    assert any(RAW_PHONE in content for content in llm.tool_contents)

    # ③ 审计：调用被记录，但**只记形状不记内容**
    audit_files = list(Path(settings.audit_dir).glob("*.jsonl"))
    assert audit_files, "审计文件应当被创建"
    records = [
        json.loads(line)
        for path in audit_files
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    calls = [r for r in records if r.get("event") == "tool_call" and r.get("tool") == TOOL_NAME]
    assert calls, "knowledge_search 的调用必须进审计"
    assert calls[0]["ok"] is True
    assert calls[0]["args_summary"].startswith("query(str,")
    assert "敏感字段怎么打码" not in json.dumps(records, ensure_ascii=False)
