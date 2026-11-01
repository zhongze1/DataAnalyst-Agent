"""DA-002 结构化日志与请求追踪测试。"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.context import (
    current_request_id,
    current_session_id,
    request_id_var,
    session_id_var,
    set_session_id,
)
from app.logging_setup import (
    JsonFormatter,
    TextFormatter,
    describe_arguments,
    short_hash,
)
from app.middleware import REQUEST_ID_HEADER, RequestContextMiddleware


# ==================================================================== 格式化
def _record(**extra: Any) -> logging.LogRecord:
    record = logging.LogRecord(
        name="app.test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="测试消息 %s", args=("参数",), exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_emits_valid_json() -> None:
    payload = json.loads(JsonFormatter().format(_record()))
    assert payload["level"] == "INFO"
    assert payload["logger"] == "app.test"
    assert payload["msg"] == "测试消息 参数"
    assert payload["ts"].endswith("Z")


def test_json_formatter_includes_request_id() -> None:
    token = request_id_var.set("abc123def456")
    try:
        payload = json.loads(JsonFormatter().format(_record()))
    finally:
        request_id_var.reset(token)
    assert payload["request_id"] == "abc123def456"


def test_json_formatter_omits_session_id_when_unset() -> None:
    payload = json.loads(JsonFormatter().format(_record()))
    assert "session_id" not in payload


def test_json_formatter_includes_session_id_when_set() -> None:
    token = set_session_id("sess-777")
    try:
        payload = json.loads(JsonFormatter().format(_record()))
    finally:
        session_id_var.reset(token)
    assert payload["session_id"] == "sess-777"


def test_json_formatter_carries_extras_but_not_reserved_attrs() -> None:
    payload = json.loads(
        JsonFormatter().format(_record(tool="load_sheet", duration_ms=12.5, pathname="X"))
    )
    assert payload["tool"] == "load_sheet"
    assert payload["duration_ms"] == 12.5
    # LogRecord 自带属性不能混进 payload
    assert "pathname" not in payload
    assert "lineno" not in payload
    assert "args" not in payload


def test_json_formatter_includes_traceback() -> None:
    try:
        raise ValueError("内部细节 /secret/path")
    except ValueError:
        import sys

        record = logging.LogRecord(
            "app.test", logging.ERROR, __file__, 1, "失败了", (), sys.exc_info()
        )
    payload = json.loads(JsonFormatter().format(record))
    assert "ValueError" in payload["exc"]
    assert "Traceback" in payload["exc"]


def test_json_formatter_handles_non_serializable_extra() -> None:
    payload = json.loads(JsonFormatter().format(_record(usage=object())))
    assert isinstance(payload["usage"], str)


def test_text_formatter_appends_request_id() -> None:
    token = request_id_var.set("rid-42")
    try:
        text = TextFormatter().format(_record())
    finally:
        request_id_var.reset(token)
    assert "req=rid-42" in text
    assert "测试消息 参数" in text


def test_text_formatter_omits_placeholder_without_request_id() -> None:
    assert "req=" not in TextFormatter().format(_record())


# ================================================================ 参数脱敏
def test_describe_arguments_does_not_leak_content() -> None:
    """工具参数可能含客户名/手机号/整段代码，日志只能记形状 + 指纹。"""
    described = describe_arguments({
        "code": "df[df['手机号'] == '13800138000']",
        "sheet": "客户明细",
        "top": 5,
    })
    assert "13800138000" not in described
    assert "客户明细" not in described
    # 只记「名称(类型,长度,指纹)」，长度随内容变化，不写死具体数字
    assert "code(str," in described
    assert "sheet(str,4," in described
    assert "top(int)" in described


def test_describe_arguments_includes_short_hash() -> None:
    """DA-009 方案 B：记录内容指纹，使「两次调用参数是否相同」可被追查。"""
    described = describe_arguments({"sheet": "销售明细"})
    assert described == f"sheet(str,4,#{short_hash('销售明细')})"
    # 同名同参 → 指纹相同
    assert describe_arguments({"sheet": "销售明细"}) == described
    # 内容不同 → 指纹不同
    assert describe_arguments({"sheet": "库存快照"}) != described


def test_short_hash_is_stable_and_short() -> None:
    a = short_hash("张三")
    assert a == short_hash("张三")
    assert a != short_hash("李四")
    assert len(a) == 8
    assert all(char in "0123456789abcdef" for char in a)


def test_hash_cannot_reveal_short_values() -> None:
    """指纹是单向的：知道长度+指纹也无法反推出原文（彩虹表不在防护范围内，
    但足以避免日志被直接读到业务数据）。"""
    import hashlib

    assert short_hash("abc") == hashlib.sha256(b"abc").hexdigest()[:8]


def test_describe_arguments_handles_containers_and_empty() -> None:
    assert describe_arguments({}) == "-"
    assert describe_arguments({"columns": ["a", "b"]}) == "columns(list,2)"
    assert describe_arguments({"opts": {"k": 1}}) == "opts(dict,1)"


# ================================================================ 中间件
@pytest.fixture
def probe_app() -> FastAPI:
    """最小应用：暴露当前上下文，供中间件断言使用。"""
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get("/probe")
    async def probe() -> dict:
        return {"request_id": current_request_id(), "session_id": current_session_id()}

    @app.get("/bind-session")
    async def bind_session() -> dict:
        set_session_id("sess-from-handler")
        return {"session_id": current_session_id()}

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("内部实现细节: /secret/path")

    return app


def test_request_id_is_generated_when_absent(probe_app: FastAPI) -> None:
    with TestClient(probe_app) as client:
        response = client.get("/probe")
    body = response.json()
    assert body["request_id"] != "-"
    assert len(body["request_id"]) == 12
    # 响应头必须回写同一个 ID，前端才能展示给用户
    assert response.headers[REQUEST_ID_HEADER] == body["request_id"]


def test_client_provided_request_id_is_echoed(probe_app: FastAPI) -> None:
    with TestClient(probe_app) as client:
        response = client.get("/probe", headers={REQUEST_ID_HEADER: "trace-from-gateway"})
    assert response.json()["request_id"] == "trace-from-gateway"
    assert response.headers[REQUEST_ID_HEADER] == "trace-from-gateway"


def test_non_ascii_request_id_is_replaced_instead_of_crashing() -> None:
    """非 ASCII 的 request_id 必须被丢弃重生成，而不是让请求 500。

    HTTP 头只能是 ASCII：早前直接原样透传并 ``.encode("ascii")``，一个中文
    ``X-Request-ID`` 就能让整个请求在中间件里抛 UnicodeEncodeError —— 而且
    这个值是外部可控的，等于给了一个廉价的拒绝服务入口。

    这里直接测净化函数：httpx 客户端自身就不让发非 ASCII 头，
    但真实 ASGI 服务器按 latin-1 解码头字节，值照样能带着怪字符进来。
    """
    from app.middleware import _sanitize_request_id

    for bad in ["请求-123", "tr\u00e4ce", "a" * 200, "has space", 'q"x', "semi;colon", ""]:
        generated = _sanitize_request_id(bad)
        assert generated != bad or bad == "", bad
        assert len(generated) == 12, bad
        # 必须能安全地进响应头
        generated.encode("ascii")


def test_safe_request_id_is_preserved() -> None:
    """合法 ID（网关常见形态）必须原样保留，否则跨服务串联就断了。"""
    from app.middleware import _sanitize_request_id

    for good in ["trace-from-gateway", "abc123", "a.b_c:d/e-f", "A" * 128]:
        assert _sanitize_request_id(good) == good


def test_gateway_request_id_is_echoed_end_to_end(probe_app: FastAPI) -> None:
    """合法 ID 端到端透传（响应头与响应体一致）。"""
    with TestClient(probe_app) as client:
        response = client.get("/probe", headers={REQUEST_ID_HEADER: "gw-001"})
    assert response.status_code == 200
    assert response.json()["request_id"] == "gw-001"
    assert response.headers[REQUEST_ID_HEADER] == "gw-001"


def test_session_id_does_not_leak_between_requests(probe_app: FastAPI) -> None:
    """关键隔离性：第二个请求不能看到第一个请求绑定的 session_id。"""
    with TestClient(probe_app) as client:
        assert client.get("/bind-session").json()["session_id"] == "sess-from-handler"
        assert client.get("/probe").json()["session_id"] == "-"


def test_unhandled_exception_is_logged_with_traceback(
    probe_app: FastAPI, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    with TestClient(probe_app, raise_server_exceptions=False) as client:
        response = client.get("/boom")

    assert response.status_code == 500
    assert "未捕获异常" in caplog.text
    assert "RuntimeError" in caplog.text
    assert "secret/path" in caplog.text  # 异常详情进日志（不是进响应体）


def test_access_log_records_method_status_and_duration(
    probe_app: FastAPI, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    with TestClient(probe_app) as client:
        client.get("/probe")

    record = next(r for r in caplog.records if r.name == "app.middleware")
    assert "GET" in record.getMessage()
    assert record.status_code == 200
    assert record.method == "GET"
    assert record.path == "/probe"
    assert record.duration_ms >= 0


# ================================================================ 格式配置
def test_setup_logging_is_idempotent() -> None:
    from app.logging_setup import setup_logging

    root = logging.getLogger()
    original = list(root.handlers)
    try:
        setup_logging("text")
        first = list(root.handlers)
        setup_logging("json")
        second = list(root.handlers)
        # 重复调用不能叠加 handler，否则日志会重复输出
        assert len(first) == len(second)
        assert isinstance(second[0].formatter, JsonFormatter)
    finally:
        root.handlers = original


def test_log_format_setting_is_validated() -> None:
    from app.config import Settings

    assert Settings(_env_file=None, log_format="json").log_format == "json"
    with pytest.raises(ValueError):
        Settings(_env_file=None, log_format="xml")


# ==================================================== 埋点不得制造运行时故障
def test_upload_log_extra_avoids_reserved_attribute(
    client, sample_xlsx: bytes, caplog: pytest.LogCaptureFixture
) -> None:
    """回归测试：extra 用 filename 会撞 LogRecord 内置属性，导致上传直接 500。

    这个坑能一路逃过测试，是因为默认 WARNING 级别下 logger.info 根本不会
    构造 LogRecord —— 所以 conftest 强制开启了 INFO。
    """
    caplog.set_level(logging.INFO)
    response = client.post(
        "/api/files/upload", files={"file": ("销售数据.xlsx", sample_xlsx)}
    )
    assert response.status_code == 200
    record = next(r for r in caplog.records if "上传成功" in r.getMessage())
    assert record.file_name == "销售数据.xlsx"
    assert record.size_bytes == len(sample_xlsx)


def test_agent_loop_emits_structured_completion_log(client, session) -> None:
    """端到端验证整条链路：中间件注入 request_id → 路由绑定 session_id → Loop 埋点。"""
    captured: list[dict] = []

    class CaptureHandler(logging.Handler):
        """在请求上下文中同步格式化，验证 ContextVar 注入效果。"""

        def emit(self, record: logging.LogRecord) -> None:
            captured.append(json.loads(JsonFormatter().format(record)))

    handler = CaptureHandler()
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        response = client.post(
            "/api/chat", json={"session_id": session.session_id, "message": "整体情况"}
        )
    finally:
        root.removeHandler(handler)

    assert response.status_code == 200
    payload = next(p for p in captured if "Agent Loop 结束" in p["msg"])
    assert payload["stop_reason"] == "final"
    assert payload["steps"] == 3
    assert payload["failed_steps"] == 0
    assert payload["duration_ms"] >= 0
    # 两层上下文都是自动注入的，调用方没有手工传
    assert payload["session_id"] == session.session_id
    assert payload["request_id"] != "-"
    # 工具参数只记形状，不落内容
    tool_payload = next(p for p in captured if "工具完成" in p["msg"])
    assert "明细" not in tool_payload["msg"] or "sheet(str," in tool_payload["msg"]
