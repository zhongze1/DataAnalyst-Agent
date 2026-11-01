"""DA-018：系统提示词里的知识库用法（验收 10 / 14）。

插入与否**只看 ``available_tools``** —— 提示词拿不到也不该拿到配置，
这条结构性约束由 ``test_sandbox_security`` 守着（本文件不去破坏它）。
"""

from __future__ import annotations

from app.agent.prompts import build_system_prompt


def test_knowledge_usage_is_documented_when_tool_is_available(session) -> None:
    prompt = build_system_prompt(
        session, ["list_sheets", "run_pandas", "knowledge_search"]
    )
    assert "## 项目文档检索（knowledge_search）" in prompt
    assert "优先调用 knowledge_search" in prompt
    assert "不要用 run_pandas 去猜口径" in prompt
    assert "先 knowledge_search" in prompt  # 顺序要求
    assert "不要用同义改写对同一问题反复检索" in prompt  # 防止把步数耗在一次提问上


def test_knowledge_section_is_omitted_when_tool_is_not_available(session) -> None:
    """关闭开关时工具根本没注册，提示词里也必须一个字都不提（验收 14）。"""
    prompt = build_system_prompt(session, ["list_sheets", "run_pandas"])
    assert "项目文档检索" not in prompt
    assert "knowledge_search" not in prompt


def test_knowledge_section_is_omitted_before_any_upload() -> None:
    prompt = build_system_prompt(None, ["list_sheets"])
    assert "knowledge_search" not in prompt
    assert "请先在左侧上传 Excel 文件" in prompt
