"""API 层特征测试：锁定 v0.1 的对外行为契约。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.testclient import TestClient

from app.schemas import ChatResponse
from stubs import AlwaysToolLLM, WrongSheetLLM

if TYPE_CHECKING:
    from app.storage.session_store import Session


# ------------------------------------------------------------------ 元信息
def test_health(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["provider"] == "mock"
    assert "version" in body


def test_index_page_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "DataAnalyst Agent" in response.text


# ------------------------------------------------------------------ 上传
def test_upload_excel(client: TestClient, sample_xlsx: bytes) -> None:
    response = client.post(
        "/api/files/upload",
        files={"file": ("销售数据.xlsx", sample_xlsx, "application/vnd.ms-excel")},
    )
    assert response.status_code == 200
    info = response.json()["file"]
    assert info["filename"] == "销售数据.xlsx"
    assert info["sheets"] == ["明细", "摘要"]
    assert info["size_bytes"] == len(sample_xlsx)
    assert info["session_id"]


def test_upload_rejects_unsupported_type(client: TestClient) -> None:
    response = client.post(
        "/api/files/upload",
        files={"file": ("note.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400
    assert "支持" in response.json()["detail"]


def test_upload_rejects_non_excel_content(client: TestClient, garbage_bytes: bytes) -> None:
    """解析失败必须返回 400，且不留脏会话。"""
    response = client.post(
        "/api/files/upload",
        files={"file": ("fake.xlsx", garbage_bytes, "application/vnd.ms-excel")},
    )
    assert response.status_code == 400
    assert "无法解析" in response.json()["detail"]


def test_upload_empty_file_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/files/upload",
        files={"file": ("empty.xlsx", b"", "application/vnd.ms-excel")},
    )
    assert response.status_code == 400


def test_get_file_info(client: TestClient, session: Session) -> None:
    response = client.get(f"/api/files/{session.session_id}")
    assert response.status_code == 200
    assert response.json()["filename"] == "测试数据.xlsx"


def test_get_file_info_unknown_session(client: TestClient) -> None:
    assert client.get("/api/files/deadbeef").status_code == 404


def test_delete_session(client: TestClient, session: Session) -> None:
    assert client.delete(f"/api/files/{session.session_id}").status_code == 200
    assert client.get(f"/api/files/{session.session_id}").status_code == 404


# ------------------------------------------------------------------ 对话
def test_chat_without_file_returns_guidance(client: TestClient) -> None:
    """没有上传文件时也要给出可执行的下一步，而不是报错。"""
    response = client.post("/api/chat", json={"message": "帮我分析"})
    assert response.status_code == 200
    body = response.json()
    assert body["stop_reason"] == "no_file"
    assert "上传" in body["answer"]
    assert body["steps"] == []


def test_chat_with_file_runs_agent_loop(client: TestClient, session: Session) -> None:
    response = client.post(
        "/api/chat",
        json={"session_id": session.session_id, "message": "整体情况怎么样？"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["stop_reason"] == "final"
    assert body["answer"].strip()
    # Mock 剧本固定三步：list_sheets -> load_sheet -> run_pandas
    assert [step["tool"] for step in body["steps"]] == [
        "list_sheets",
        "load_sheet",
        "run_pandas",
    ]
    assert all(step["ok"] for step in body["steps"])


def test_chat_steps_carry_arguments_and_result(client: TestClient, session: Session) -> None:
    """前端要展示「每步调了什么、返回了什么」，字段必须齐全。"""
    body = client.post(
        "/api/chat", json={"session_id": session.session_id, "message": "体检"}
    ).json()
    load_step = next(s for s in body["steps"] if s["tool"] == "load_sheet")
    assert load_step["arguments"] == {"sheet": "明细"}
    assert load_step["result"]
    assert load_step["error"] is None
    assert load_step["step"] == 2


def test_chat_survives_tool_failure(make_client, session) -> None:
    """工具报错不能让整个 Loop 崩掉，错误要回传给模型让它自己纠正。"""
    client = make_client(WrongSheetLLM())
    body = client.post(
        "/api/chat", json={"session_id": session.session_id, "message": "分析一下"}
    ).json()
    assert [s["tool"] for s in body["steps"]] == ["load_sheet"]
    failed = body["steps"][0]
    assert failed["ok"] is False
    assert "不存在" in failed["error"]
    # 模型拿到错误后自行修正并给出结论
    assert body["stop_reason"] == "final"
    assert body["answer"] == "已根据实际可用信息给出结论。"


def test_chat_hits_step_limit(make_client, session) -> None:
    """永远调工具的模型必须被步数上限拦住，且给出收尾说明。"""
    client = make_client(AlwaysToolLLM(), max_agent_steps=2)
    body = client.post(
        "/api/chat", json={"session_id": session.session_id, "message": "分析"}
    ).json()
    assert body["stop_reason"] == "max_steps"
    assert len(body["steps"]) == 2
    assert "最大" in body["answer"]


def test_chat_with_unknown_session_returns_404(client: TestClient) -> None:
    response = client.post("/api/chat", json={"session_id": "nope", "message": "hi"})
    assert response.status_code == 404


def test_chat_rejects_empty_message(client: TestClient) -> None:
    assert client.post("/api/chat", json={"message": ""}).status_code == 422


# ------------------------------------------------------- 契约锁定：响应字段
def test_chat_response_field_contract() -> None:
    """锁定 ChatResponse 字段集合：新增/删除字段必须同步改前端与本用例。"""
    assert set(ChatResponse.model_fields) == {
        "session_id",
        "answer",
        "steps",
        "stop_reason",
        "usage",
    }


def test_chat_response_keys_match_contract(client: TestClient, session: Session) -> None:
    body = client.post(
        "/api/chat", json={"session_id": session.session_id, "message": "结构"}
    ).json()
    assert set(body) == set(ChatResponse.model_fields)
    assert set(body["steps"][0]) == {
        "step",
        "tool",
        "arguments",
        "ok",
        "result",
        "error",
    }


# ------------------------------------------------------------------ 重置
def test_reset_clears_history_but_keeps_data(client: TestClient, session: Session) -> None:
    client.post("/api/chat", json={"session_id": session.session_id, "message": "第一轮"})
    assert session.messages, "对话历史应被记录"

    response = client.post(f"/api/chat/reset?session_id={session.session_id}")
    assert response.status_code == 200
    assert response.json() == {"reset": True}
    assert session.messages == []
    # 数据仍在，可以继续提问
    assert client.get(f"/api/files/{session.session_id}").status_code == 200


def test_reset_unknown_session_returns_404(client: TestClient) -> None:
    assert client.post("/api/chat/reset?session_id=nope").status_code == 404


def test_multi_turn_keeps_context(client: TestClient, session: Session) -> None:
    """多轮对话：第二轮的消息历史里应带上前一轮的问答。"""
    client.post("/api/chat", json={"session_id": session.session_id, "message": "第一轮问题"})
    client.post("/api/chat", json={"session_id": session.session_id, "message": "第二轮问题"})
    roles = [m["role"] for m in session.messages]
    assert roles.count("user") == 2
    assert roles[-1] == "assistant"
