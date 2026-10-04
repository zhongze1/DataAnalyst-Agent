"""DA-001 配置校验测试。

校验函数是纯函数（不打印、不抛异常），所以每条规则都能独立验证。
最后两项是端到端验证：lifespan 真的会因为坏配置拒绝启动。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import (
    ConfigProblem,
    Settings,
    format_problems,
    runtime_summary,
    validate_settings,
)

VALID_KEY = "sk-" + "x" * 32


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "_env_file": None,
        "llm_provider": "openai",
        "llm_api_key": VALID_KEY,
        "llm_base_url": "https://api.deepseek.com/v1",
        "llm_model": "deepseek-chat",
        "upload_dir": tmp_path / "uploads",
    }
    values.update(overrides)
    return Settings(**values)


def _keys(problems: list[ConfigProblem]) -> set[str]:
    return {p.key for p in problems}


# ============================================================ 正常配置
def test_valid_config_has_no_problems(tmp_path: Path) -> None:
    assert validate_settings(_settings(tmp_path)) == []


def test_mock_provider_needs_no_key_or_url(tmp_path: Path) -> None:
    """离线演示模式不该被 Key / base_url 拦住。"""
    settings = _settings(
        tmp_path, llm_provider="mock", llm_api_key="", llm_base_url="", llm_model=""
    )
    assert validate_settings(settings) == []


# ============================================================ LLM 配置
def test_missing_api_key_is_reported(tmp_path: Path) -> None:
    settings = _settings(tmp_path, llm_api_key="")
    problems = validate_settings(settings)
    assert "AGENT_LLM_API_KEY" in _keys(problems)
    hint = next(p.hint for p in problems if p.key == "AGENT_LLM_API_KEY")
    assert "mock" in hint  # 必须给出离线模式的出路


def test_whitespace_only_key_counts_as_missing(tmp_path: Path) -> None:
    assert "AGENT_LLM_API_KEY" in _keys(validate_settings(_settings(tmp_path, llm_api_key="   ")))


@pytest.mark.parametrize(
    "base_url",
    [
        "api.deepseek.com/v1",     # 缺协议
        "https://",                # 缺主机名
        "ftp://api.deepseek.com",  # 协议不支持
        "https://api deepseek.com",  # 含空格
    ],
)
def test_malformed_base_url_is_reported(tmp_path: Path, base_url: str) -> None:
    assert "AGENT_LLM_BASE_URL" in _keys(
        validate_settings(_settings(tmp_path, llm_base_url=base_url))
    )


def test_empty_model_is_reported(tmp_path: Path) -> None:
    assert "AGENT_LLM_MODEL" in _keys(validate_settings(_settings(tmp_path, llm_model="")))


def test_local_llm_endpoint_is_accepted(tmp_path: Path) -> None:
    """自建服务（vLLM / Ollama）不应被误判为非法 URL。"""
    settings = _settings(tmp_path, llm_base_url="http://localhost:11434/v1")
    assert validate_settings(settings) == []


# ============================================================ 数值与白名单
def test_invalid_agent_budget_is_reported(tmp_path: Path) -> None:
    settings = _settings(tmp_path, max_agent_steps=0, max_tool_output_chars=100)
    keys = _keys(validate_settings(settings))
    assert "AGENT_MAX_AGENT_STEPS" in keys
    assert "AGENT_MAX_TOOL_OUTPUT_CHARS" in keys


def test_empty_suffix_allowlist_is_reported(tmp_path: Path) -> None:
    settings = _settings(tmp_path, allowed_suffixes=())
    assert "allowed_suffixes" in _keys(validate_settings(settings))


# ============================================================ 上传目录
def test_valid_upload_dir_passes_and_leaves_no_probe_file(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert validate_settings(settings) == []
    # 写探测文件必须清理干净，不能污染用户目录
    assert list(settings.upload_dir.iterdir()) == []


def test_unwritable_upload_dir_is_reported(tmp_path: Path) -> None:
    """upload_dir 指向一个已存在的普通文件时，mkdir 必然失败。"""
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x", encoding="utf-8")
    problems = validate_settings(_settings(tmp_path, upload_dir=blocker))
    assert "upload_dir" in _keys(problems)
    assert "不可写" in problems[0].message


# ============================================================ 展示辅助
def test_format_problems_lists_keys_and_hints() -> None:
    text = format_problems(
        [
            ConfigProblem("AGENT_LLM_API_KEY", "未配置 API Key", "去 .env 里填"),
            ConfigProblem("AGENT_LLM_MODEL", "未配置模型名"),
        ]
    )
    assert "共 2 项" in text
    assert "[AGENT_LLM_API_KEY] 未配置 API Key" in text
    assert "建议: 去 .env 里填" in text
    # 没有 hint 的问题不应输出空白的建议行
    assert text.count("建议:") == 1


def test_runtime_summary_shows_endpoint_without_leaking_key(tmp_path: Path) -> None:
    summary = runtime_summary(_settings(tmp_path))
    assert "端点=https://api.deepseek.com/v1/chat/completions" in summary
    assert "模型=deepseek-chat" in summary
    assert f"Key=已加载({len(VALID_KEY)} 字符)" in summary
    assert VALID_KEY not in summary  # 绝不能打印 Key 内容


def test_runtime_summary_for_mock_provider(tmp_path: Path) -> None:
    summary = runtime_summary(_settings(tmp_path, llm_provider="mock"))
    assert "provider=mock" in summary
    assert "离线演示" in summary
    assert "端点" not in summary


def test_runtime_summary_reports_strict_flag(tmp_path: Path) -> None:
    assert "严格配置=开" in runtime_summary(_settings(tmp_path, strict_config=True))
    assert "严格配置=关" in runtime_summary(_settings(tmp_path, strict_config=False))


# ============================================================ 端到端 fail-fast
def test_lifespan_refuses_to_start_with_bad_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DA-001 的核心诉求：坏配置必须起不来，而不是等用户提问才 500。"""
    from app.main import app

    bad = _settings(tmp_path, llm_api_key="", strict_config=True)
    monkeypatch.setattr("app.main.get_settings", lambda: bad)

    with pytest.raises(RuntimeError) as excinfo:
        with TestClient(app):
            pass  # pragma: no cover - 正常情况下不会执行到这里
    assert "AGENT_LLM_API_KEY" in str(excinfo.value)


def test_lifespan_starts_when_problems_tolerated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, store
) -> None:
    """strict_config=false 时允许带病启动 —— 便于临时调试。"""
    from app.main import app

    bad = _settings(tmp_path, llm_api_key="", strict_config=False)
    monkeypatch.setattr("app.main.get_settings", lambda: bad)
    # 避免 lifespan 调用真实的 get_store() 触碰仓库内 data/uploads
    monkeypatch.setattr("app.main.get_store", lambda: store)

    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
