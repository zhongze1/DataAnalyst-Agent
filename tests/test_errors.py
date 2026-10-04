"""DA-003 统一错误响应测试。

核心不变量：
1. 所有错误响应体形如 ``{"error": {code, message, request_id}}``；
2. **响应体里绝不出现内部异常细节**（openpyxl 报错、厂商响应体、本地路径）；
3. 详细信息只出现在日志里。
"""

from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.errors import (
    AppError,
    FileEmptyError,
    FileParseError,
    FileTooLargeError,
    FileTypeError,
    LLMServiceError,
    SessionNotFoundError,
)
from app.middleware import REQUEST_ID_HEADER, RequestContextMiddleware
from stubs import BrokenLLM

SECRET = "D:\\secret\\config.yaml"


# ============================================================ 异常类契约
def test_app_error_defaults() -> None:
    error = AppError()
    assert error.message == AppError.default_message
    assert error.to_payload() == {
        "error": {"code": "APP_ERROR", "message": AppError.default_message}
    }
    assert error.to_payload("rid-1")["error"]["request_id"] == "rid-1"


def test_app_error_keeps_cause_out_of_message() -> None:
    """核心纪律：cause 只用于日志，绝不出现在 message 里。"""
    cause = ValueError("openpyxl 内部栈：D:\\secret\\path\\file.xlsx")
    error = FileParseError(cause=cause)
    assert error.cause is cause
    assert "secret" not in error.message
    assert "openpyxl" not in error.message
    assert "secret" not in str(error.to_payload())


def test_subclasses_have_expected_status_and_code() -> None:
    cases = [
        (FileTypeError(), 400, "FILE_TYPE_UNSUPPORTED"),
        (FileEmptyError(), 400, "FILE_EMPTY"),
        (FileTooLargeError(50 * 1024 * 1024), 413, "FILE_TOO_LARGE"),
        (FileParseError(), 400, "FILE_PARSE_ERROR"),
        (SessionNotFoundError(), 404, "SESSION_NOT_FOUND"),
        (LLMServiceError(), 502, "LLM_SERVICE_ERROR"),
    ]
    for error, status, code in cases:
        assert error.http_status == status
        assert error.code == code
        assert error.message  # 每个都必须有面向用户的中文文案


def test_too_large_error_message_contains_limit() -> None:
    assert "50MB" in FileTooLargeError(50 * 1024 * 1024).message


# ============================================================ HTTP 响应体
def _assert_envelope(response, status: int, code: str) -> dict:
    assert response.status_code == status
    body = response.json()
    assert set(body) == {"error"}
    assert set(body["error"]) >= {"code", "message", "request_id"}
    assert body["error"]["code"] == code
    return body["error"]


def test_unsupported_file_type_error(client) -> None:
    response = client.post("/api/files/upload", files={"file": ("note.txt", b"hi")})
    error = _assert_envelope(response, 400, "FILE_TYPE_UNSUPPORTED")
    assert ".xlsx" in error["message"]


def test_empty_file_error(client) -> None:
    response = client.post("/api/files/upload", files={"file": ("e.xlsx", b"")})
    _assert_envelope(response, 400, "FILE_EMPTY")


def test_oversized_file_error(client, sample_xlsx: bytes, settings) -> None:
    settings.max_upload_bytes = 16
    response = client.post("/api/files/upload", files={"file": ("big.xlsx", sample_xlsx)})
    error = _assert_envelope(response, 413, "FILE_TOO_LARGE")
    assert "MB" in error["message"]


def test_unparsable_file_does_not_leak_internals(client, garbage_bytes: bytes) -> None:
    """最关键的一条：openpyxl 的报错细节不能出现在响应体里。"""
    response = client.post("/api/files/upload", files={"file": ("fake.xlsx", garbage_bytes)})
    error = _assert_envelope(response, 400, "FILE_PARSE_ERROR")
    assert "opener" not in error["message"]
    assert "zipfile" not in error["message"].lower()
    assert "Traceback" not in error["message"]


def test_parse_failure_details_go_to_log(client, garbage_bytes: bytes, caplog) -> None:
    caplog.set_level(logging.WARNING)
    client.post("/api/files/upload", files={"file": ("fake.xlsx", garbage_bytes)})
    # 真实原因必须留痕，否则排障无从下手
    assert "Excel 解析失败" in caplog.text


def test_session_not_found_error(client) -> None:
    _assert_envelope(client.get("/api/files/nope"), 404, "SESSION_NOT_FOUND")
    _assert_envelope(
        client.post("/api/chat", json={"session_id": "nope", "message": "hi"}),
        404,
        "SESSION_NOT_FOUND",
    )
    _assert_envelope(client.post("/api/chat/reset?session_id=nope"), 404, "SESSION_NOT_FOUND")
    _assert_envelope(client.delete("/api/files/nope"), 404, "SESSION_NOT_FOUND")


def test_validation_error_uses_envelope_and_drops_input(client) -> None:
    """422 也要统一外壳，且 details 里不能带 input（可能是文件内容/用户原文）。"""
    response = client.post("/api/chat", json={"message": ""})
    error = _assert_envelope(response, 422, "VALIDATION_ERROR")
    assert error["details"], "应保留校验细节供 API 调用方排查"
    for item in error["details"]:
        assert set(item) == {"loc", "msg", "type"}
        assert "input" not in item


def test_unknown_route_uses_envelope(client) -> None:
    _assert_envelope(client.get("/api/does-not-exist"), 404, "NOT_FOUND")


def test_error_response_carries_request_id(client) -> None:
    response = client.get("/api/files/nope", headers={REQUEST_ID_HEADER: "trace-abc"})
    error = response.json()["error"]
    assert error["request_id"] == "trace-abc"
    assert response.headers[REQUEST_ID_HEADER] == "trace-abc"


@pytest.fixture
def crash_app() -> FastAPI:
    """装有全局异常处理器的最小应用，用于验证 500 兜底。"""
    from app.main import register_exception_handlers

    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError(f"内部实现细节 {SECRET}")

    return app


def test_unexpected_exception_returns_generic_message_and_logs_detail(
    crash_app: FastAPI, caplog
) -> None:
    """未捕获异常：客户端只拿通用文案，traceback 进日志。"""
    caplog.set_level(logging.ERROR)
    with TestClient(crash_app, raise_server_exceptions=False) as client:
        response = client.get("/boom")

    assert response.status_code == 500
    error = response.json()["error"]
    assert error["code"] == "INTERNAL_ERROR"
    assert "secret" not in error["message"]
    assert "服务器内部错误" in error["message"]
    # request_id 必须能从 scope 取到（此时 ContextVar 已被中间件重置）
    assert error["request_id"] == response.headers[REQUEST_ID_HEADER]
    # 详情必须在日志里，否则排障无从下手
    assert "secret\\config.yaml" in caplog.text
    assert "Traceback" in caplog.text


def test_openapi_documents_error_shape(client) -> None:
    """错误响应结构要出现在 /docs 里，调用方才知道怎么解析。"""
    schema = client.get("/openapi.json").json()
    assert "ErrorResponse" in schema["components"]["schemas"]
    responses = schema["paths"]["/api/files/{session_id}"]["get"]["responses"]
    assert "404" in responses
    assert responses["404"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "ErrorResponse"
    )


# ============================================================ LLM 故障脱敏
def test_llm_failure_answer_is_sanitized(make_client, session) -> None:
    """模型故障时，厂商返回体不能出现在 answer 里。"""
    client = make_client(BrokenLLM())
    response = client.post(
        "/api/chat", json={"session_id": session.session_id, "message": "分析"}
    )
    body = response.json()
    assert body["stop_reason"] == "llm_error"
    assert "模拟的模型故障" not in body["answer"]
    assert "模型服务调用失败" in body["answer"]


def test_llm_failure_details_go_to_log(make_client, session, caplog) -> None:
    caplog.set_level(logging.ERROR)
    client = make_client(BrokenLLM())
    client.post("/api/chat", json={"session_id": session.session_id, "message": "分析"})
    assert "模拟的模型故障" in caplog.text
    # error_code 是结构化字段，不在消息文本里，需从 record 上读
    record = next(r for r in caplog.records if r.name == "app.agent.loop")
    assert record.error_code == "LLM_SERVICE_ERROR"


def test_tool_errors_keep_detail_for_model(session) -> None:
    """工具错误不能被脱敏：模型需要细节才能自我纠正。"""
    from app.tools import registry
    from app.tools.base import ToolContext

    ctx = ToolContext(
        session_id=session.session_id,
        file_path=str(session.path),
        sheets=list(session.sheets),
    )
    ok, text = registry.run("load_sheet", {"sheet": "不存在"}, ctx)
    assert ok is False
    assert "可用工作表" in text
