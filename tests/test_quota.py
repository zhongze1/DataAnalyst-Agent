"""DA-017 配额与并发保护测试。

覆盖五类不变量：
1. **请求次数硬配额**：用户每分钟 / 会话每分钟 / 用户每日，且被拒绝的请求不计数；
2. **并发保护**：用户级与全局级独立，超限立即拒绝、不排队；
3. **释放**：正常、异常、取消、客户端断开四条路径都不泄漏并发占用；
4. **用量统计**：usage 逐轮累加（含收尾轮与失败前的用量），缺失时不估算；
5. **契约**：429 走统一错误体 + ``Retry-After``，SSE 在推流前就返回 JSON 错误。

测试全部离线（Mock LLM）、全部写 ``tmp_path``，不碰真实 ``data/`` 目录。
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from app.config import Settings, validate_settings
from app.errors import QuotaExceededError, ServerBusyError, TooManyConcurrentError
from app.llm.mock_client import MockLLMClient
from app.main import app as fastapi_app
from app.services.quota_service import LOCAL_USER_KEY, QuotaService, resolve_tz
from app.storage.sqlite_quota_repo import SqliteQuotaRepository
from stubs import (
    BlockingLLM,
    BrokenLLM,
    CountingLLM,
    UsageLLM,
    UsageThenBrokenLLM,
)

TZ = ZoneInfo("Asia/Shanghai")


# ============================================================ 夹具与工具
class FakeClock:
    """可手动推进的时钟：窗口滚动、跨日重置这类行为必须能确定性验证。"""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


@pytest.fixture
def clock() -> FakeClock:
    # 固定在「北京时间 10:30」，离零点足够远，避免用例在跨日边界上随机失败
    return FakeClock(datetime(2026, 1, 1, 2, 30, tzinfo=timezone.utc))


@pytest.fixture
def make_quota(settings: Settings):
    """按需构造 QuotaService，可覆盖任意配额项；repo 指向用例自己的 tmp 数据库。"""
    repos: list[SqliteQuotaRepository] = []

    def _make(clock: FakeClock | None = None, **overrides) -> QuotaService:
        config = settings.model_copy(update=overrides)
        repo = SqliteQuotaRepository(config.db_path)
        repos.append(repo)
        return QuotaService(config, repo=repo, clock=clock)

    yield _make
    # 显式关连接：漏句柄会在 GC 时刷出一屏 ResourceWarning
    for repo in repos:
        repo.close()


def _day_of(moment: datetime, tz: ZoneInfo = TZ) -> str:
    return moment.astimezone(tz).strftime("%Y-%m-%d")


def _daily(settings: Settings, user_key: str, day: str):
    """直读存储层（DAO 是唯一真相源），避免依赖服务的私有方法。"""
    repo = SqliteQuotaRepository(settings.db_path)
    try:
        return repo.get_daily(user_key, day)
    finally:
        repo.close()


def _use_quota(quota: QuotaService) -> None:
    """把当前用例的配额服务挂到应用上（覆盖 conftest 里的默认实例）。"""
    from app.dependencies import get_quota_service

    fastapi_app.dependency_overrides[get_quota_service] = lambda: quota


def _unique_settings(tmp_path, **overrides) -> Settings:
    return Settings(
        _env_file=None,
        llm_provider="mock",
        upload_dir=tmp_path / "uploads",
        db_path=tmp_path / "cfg.db",
        auth_enabled=False,
        **overrides,
    )


# ============================================================ 分钟窗口 / 会话额度
def test_user_minute_limit_rejects_and_does_not_count_the_rejection(make_quota, clock, settings):
    quota = make_quota(clock, quota_user_per_minute=2)
    for _ in range(2):
        quota.release(quota.reserve(user_key="u1"))

    with pytest.raises(QuotaExceededError) as caught:
        quota.reserve(user_key="u1")

    assert caught.value.code == "QUOTA_EXCEEDED"
    assert 1 <= caught.value.retry_after <= 60
    # 被拒绝的请求不能吃掉额度
    assert _daily(settings, "u1", _day_of(clock())).requests == 2


def test_minute_window_rolls_over_after_a_minute(make_quota, clock):
    quota = make_quota(clock, quota_user_per_minute=1)
    quota.release(quota.reserve(user_key="u1"))
    with pytest.raises(QuotaExceededError):
        quota.reserve(user_key="u1")

    clock.advance(61)
    quota.release(quota.reserve(user_key="u1"))  # 新窗口，额度恢复


def test_session_limit_is_independent_from_user_limit(make_quota, clock):
    quota = make_quota(clock, quota_session_per_minute=1, quota_user_per_minute=10)
    quota.release(quota.reserve(user_key="u1", session_id="s1"))

    with pytest.raises(QuotaExceededError):
        quota.reserve(user_key="u1", session_id="s1")

    # 换一个会话：用户维度没用满，必须放行（两个维度不能混成一个计数）
    quota.release(quota.reserve(user_key="u1", session_id="s2"))


def test_user_limit_counts_across_sessions(make_quota, clock):
    quota = make_quota(clock, quota_user_per_minute=2, quota_session_per_minute=5)
    quota.release(quota.reserve(user_key="u1", session_id="s1"))
    quota.release(quota.reserve(user_key="u1", session_id="s2"))

    with pytest.raises(QuotaExceededError):
        quota.reserve(user_key="u1", session_id="s3")


def test_rejection_leaves_no_partial_occupation(make_quota, clock, settings):
    """被拒绝的请求既不加每日计数，也不占并发。"""
    quota = make_quota(
        clock,
        quota_user_per_minute=10,
        quota_session_per_minute=1,
        quota_max_concurrency_global=1,
    )
    reservation = quota.reserve(user_key="u1", session_id="s1")

    # 这次是「会话额度」拒绝：检查顺序在并发之前，所以它不会因为并发而被误拒
    with pytest.raises(QuotaExceededError) as caught:
        quota.reserve(user_key="u1", session_id="s1")
    assert "会话" in caught.value.message

    quota.release(reservation)
    # 若被拒绝的那次偷偷占了并发，这里的全局并发就已是 1，会被判繁忙
    quota.release(quota.reserve(user_key="u2", session_id="s2"))
    assert _daily(settings, "u1", _day_of(clock())).requests == 1


# ============================================================ 每日额度
def test_daily_limit_message_matches_actual_reset_time(make_quota, clock, settings):
    quota = make_quota(clock, quota_user_per_day=2, quota_user_per_minute=10)
    for _ in range(2):
        quota.release(quota.reserve(user_key="u1"))

    with pytest.raises(QuotaExceededError) as caught:
        quota.reserve(user_key="u1")

    error = caught.value
    # 文案里的重置时刻与 retry_after 必须指向同一时刻（口径一致）
    reset_at = (clock().astimezone(TZ) + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    assert reset_at.strftime("%m-%d %H:%M") in error.message
    assert error.retry_after == int((reset_at - clock()).total_seconds())
    assert error.retry_after > 0


def test_daily_counter_persists_across_restart(make_quota, clock, settings):
    quota = make_quota(clock, quota_user_per_day=1)
    quota.release(quota.reserve(user_key="u1"))

    # 模拟重启：全新的服务 + 全新的数据库连接，指向同一个文件
    restarted = QuotaService(settings.model_copy(update={"quota_user_per_day": 1}), clock=clock)
    with pytest.raises(QuotaExceededError):
        restarted.reserve(user_key="u1")


def test_daily_counter_uses_quota_timezone(settings):
    """换时区会换「自然日」的边界，说明计费口径真的按配置走。"""
    # 取一个 UTC 还是当天、但在 UTC+14 已是次日的时刻
    late = FakeClock(datetime(2026, 1, 1, 20, 0, tzinfo=timezone.utc))
    config = settings.model_copy(update={"quota_timezone": "Pacific/Kiritimati"})  # UTC+14
    quota = QuotaService(config, repo=SqliteQuotaRepository(config.db_path), clock=late)
    reservation = quota.reserve(user_key="u1")
    quota.release(reservation)

    assert reservation.day == _day_of(late(), ZoneInfo("Pacific/Kiritimati"))
    assert reservation.day != _day_of(late(), ZoneInfo("UTC"))


# ============================================================ 并发保护
def test_user_concurrency_rejects_second_request_immediately(make_quota, clock):
    quota = make_quota(clock, quota_max_concurrency_per_user=1)
    reservation = quota.reserve(user_key="u1")

    with pytest.raises(TooManyConcurrentError) as caught:
        quota.reserve(user_key="u1")
    assert caught.value.code == "TOO_MANY_CONCURRENT"

    quota.release(reservation)
    quota.release(quota.reserve(user_key="u1"))  # 释放后可再次进入


def test_global_concurrency_blocks_other_users(make_quota, clock):
    quota = make_quota(clock, quota_max_concurrency_global=1)
    quota.reserve(user_key="u1")

    with pytest.raises(ServerBusyError) as caught:
        quota.reserve(user_key="u2")
    assert caught.value.code == "SERVER_BUSY"
    assert caught.value.retry_after >= 1


def test_release_is_idempotent(make_quota, clock):
    quota = make_quota(clock, quota_max_concurrency_per_user=1)
    reservation = quota.reserve(user_key="u1")

    quota.release(reservation)
    quota.release(reservation)  # 重复释放不能把计数打到负数

    quota.release(quota.reserve(user_key="u1"))


def test_quota_disabled_bypasses_everything(make_quota, clock, settings):
    quota = make_quota(clock, quota_enabled=False)
    for _ in range(30):
        reservation = quota.reserve(user_key="u1")
        assert reservation.counted is False
    quota.settle(reservation, {"total_tokens": 10})
    quota.release(reservation)

    # 关闭时不该留下任何计数（连统计也不写）
    assert _daily(settings, "u1", _day_of(clock())) is None


# ============================================================ 用量结算
def test_settle_persists_reported_tokens(make_quota, clock, settings):
    quota = make_quota(clock)
    reservation = quota.reserve(user_key="u1")
    quota.release(reservation)
    quota.settle(
        reservation,
        {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "usage_reported_calls": 3,
            "usage_missing_calls": 1,
        },
    )

    record = _daily(settings, "u1", reservation.day)
    assert (record.requests, record.prompt_tokens, record.completion_tokens) == (1, 100, 20)
    assert record.total_tokens == 120
    assert record.usage_reported_calls == 3
    assert record.usage_missing_calls == 1

    # 再结算一轮：必须累加而不是覆盖
    quota.settle(reservation, {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3})
    assert _daily(settings, "u1", reservation.day).total_tokens == 123


def test_settle_without_usage_records_no_tokens(make_quota, clock, settings):
    quota = make_quota(clock)
    reservation = quota.reserve(user_key="u1")
    quota.release(reservation)
    quota.settle(reservation, {})

    record = _daily(settings, "u1", reservation.day)
    # 缺失 usage 时**不估算**：token 保持 0，只记「有一轮没回报」
    assert record.total_tokens == 0
    assert record.prompt_tokens == 0
    assert record.usage_missing_calls == 1
    assert record.usage_reported_calls == 0


def test_settle_ignores_bogus_token_values(make_quota, clock, settings):
    """厂商给了字符串/浮点/负数，一律当作「没有这个字段」，不能猜。"""
    quota = make_quota(clock)
    reservation = quota.reserve(user_key="u1")
    quota.release(reservation)
    quota.settle(
        reservation,
        {"prompt_tokens": "100", "completion_tokens": 12.5, "total_tokens": -3},
    )

    record = _daily(settings, "u1", reservation.day)
    assert record.total_tokens == 0
    assert record.prompt_tokens == 0
    assert record.completion_tokens == 0


def test_settle_failure_does_not_break_the_request(settings, clock):
    """统计写入失败只记日志：它是观测，不是准入。"""

    class BoomRepo:
        def get_daily(self, user_key, day):
            return None

        def increment_requests(self, user_key, day):
            return 1

        def add_tokens(self, *args, **kwargs):
            raise OSError("模拟磁盘故障")

    quota = QuotaService(settings, repo=BoomRepo(), clock=clock)
    reservation = quota.reserve(user_key="u1")
    quota.release(reservation)
    quota.settle(reservation, {"total_tokens": 10})  # 不抛异常即通过


# ============================================================ 时区与配置
def test_resolve_tz_handles_utc_and_unknown_names(settings):
    assert resolve_tz("UTC") == timezone.utc
    assert resolve_tz("") == timezone.utc
    # 未知时区只告警回退，绝不抛异常（否则一个写错的时区名会让服务起不来）
    assert resolve_tz("Mars/Olympus") == timezone.utc


def test_quota_timezone_fallback_keeps_service_alive(settings, clock):
    config = settings.model_copy(update={"quota_timezone": "Mars/Olympus"})
    quota = QuotaService(config, repo=SqliteQuotaRepository(config.db_path), clock=clock)
    assert quota.timezone == timezone.utc
    quota.release(quota.reserve(user_key="u1"))


def test_defaults_are_backward_compatible(tmp_path):
    """没有任何配额环境变量时，默认值必须能通过校验（升级不会起不来）。"""
    fresh = _unique_settings(tmp_path)
    assert validate_settings(fresh) == []
    assert fresh.quota_enabled is True
    assert fresh.quota_user_per_minute == 10
    assert fresh.quota_user_per_day == 50
    assert fresh.quota_session_per_minute == 5
    assert fresh.quota_max_concurrency_per_user == 1
    assert fresh.quota_max_concurrency_global == 4
    assert fresh.quota_timezone == "Asia/Shanghai"


def test_invalid_quota_values_are_reported(tmp_path):
    problems = validate_settings(_unique_settings(tmp_path, quota_user_per_minute=0))
    assert [p.key for p in problems] == ["AGENT_QUOTA_USER_PER_MINUTE"]

    problems = validate_settings(_unique_settings(tmp_path, quota_max_concurrency_global=-1))
    assert [p.key for p in problems] == ["AGENT_QUOTA_MAX_CONCURRENCY_GLOBAL"]

    # 每分钟上限大于每日上限：额度会在一分钟内被抽干，通常是写反了
    problems = validate_settings(
        _unique_settings(tmp_path, quota_user_per_minute=100, quota_user_per_day=10)
    )
    assert [p.key for p in problems] == ["AGENT_QUOTA_USER_PER_MINUTE"]


def test_quota_validation_is_skipped_when_disabled(tmp_path):
    fresh = _unique_settings(tmp_path, quota_enabled=False, quota_user_per_minute=0)
    assert validate_settings(fresh) == []


# ============================================================ 审计
def test_rejection_is_audited_without_sensitive_content(make_quota, clock, settings):
    quota = make_quota(clock, quota_user_per_minute=1)
    quota.release(quota.reserve(user_key="u1"))
    with pytest.raises(QuotaExceededError):
        quota.reserve(user_key="u1")

    files = list(settings.audit_dir.glob("*.jsonl"))
    assert files, "配额拒绝必须留下审计记录"
    raw = files[0].read_text(encoding="utf-8")
    events = [json.loads(line) for line in raw.splitlines()]
    rejected = [event for event in events if event["event"] == "quota_rejected"]
    assert rejected and rejected[0]["dimension"] == "user_per_minute"
    assert rejected[0]["limit"] == 1

    # 审计里不能出现提示词、口令、Cookie 这类内容
    for secret in ("password", "cookie", "prompt", "Authorization"):
        assert secret.lower() not in raw.lower()


def test_settlement_is_audited(make_quota, clock, settings):
    quota = make_quota(clock)
    reservation = quota.reserve(user_key="u1")
    quota.release(reservation)
    quota.settle(reservation, {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10})

    files = list(settings.audit_dir.glob("*.jsonl"))
    events = [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines()]
    settled = [event for event in events if event["event"] == "quota_settled"]
    assert settled and settled[0]["usage"]["total_tokens"] == 10


# ============================================================ HTTP 端到端
def test_chat_returns_429_envelope_with_retry_after(client, make_quota, clock, session):
    _use_quota(make_quota(clock, quota_user_per_minute=1))
    body = {"session_id": session.session_id, "message": "整体情况"}

    assert client.post("/api/chat", json=body).status_code == 200
    rejected = client.post("/api/chat", json=body)

    assert rejected.status_code == 429
    error = rejected.json()["error"]
    assert error["code"] == "QUOTA_EXCEEDED"
    assert error["message"]  # 人话文案，非空
    assert error["request_id"]  # 报障能定位到日志
    assert 1 <= int(rejected.headers["Retry-After"]) <= 60
    # 错误体契约不变：仍然只有 code / message / request_id
    assert set(error) == {"code", "message", "request_id"}


def test_rejected_request_never_reaches_the_model(make_client, make_quota, clock, session):
    llm = CountingLLM()
    client = make_client(llm)
    _use_quota(make_quota(clock, quota_user_per_minute=1))
    body = {"session_id": session.session_id, "message": "整体情况"}

    assert client.post("/api/chat", json=body).status_code == 200
    calls_before = llm.chat_calls
    assert calls_before > 0

    assert client.post("/api/chat", json=body).status_code == 429
    assert llm.chat_calls == calls_before, "被拒绝的请求不能打到模型（那才是真的烧钱）"


def test_chat_stream_precheck_returns_json_429(client, make_quota, clock, session):
    """SSE 的预检必须在推流之前：推一半再中断，前端拿不到可展示的错误。"""
    _use_quota(make_quota(clock, quota_user_per_minute=1))
    body = {"session_id": session.session_id, "message": "整体情况"}

    first = client.post("/api/chat/stream", json=body)
    assert first.status_code == 200
    assert first.headers["content-type"].startswith("text/event-stream")

    rejected = client.post("/api/chat/stream", json=body)
    assert rejected.status_code == 429
    assert rejected.headers["content-type"].startswith("application/json")
    assert rejected.json()["error"]["code"] == "QUOTA_EXCEEDED"
    assert "event:" not in rejected.text


def test_upload_then_chat_end_to_end_respects_quota(client, make_quota, clock, sample_xlsx):
    """端到端：上传真实 Excel → 正常提问 → 再问被 429 拒绝。

    与手工验收脚本走同一条路径（multipart 上传 + 统一错误体），
    确保配额不是在「绕过 HTTP」的层面才成立。
    """
    _use_quota(make_quota(clock, quota_user_per_minute=1))
    upload = client.post(
        "/api/files/upload",
        files={
            "file": (
                "测试数据.xlsx",
                sample_xlsx,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )
    assert upload.status_code == 200
    session_id = upload.json()["file"]["session_id"]

    body = {"session_id": session_id, "message": "整体情况"}
    ok = client.post("/api/chat", json=body)
    assert ok.status_code == 200
    assert ok.json()["answer"].strip()

    rejected = client.post("/api/chat", json=body)
    assert rejected.status_code == 429
    assert rejected.json()["error"]["code"] == "QUOTA_EXCEEDED"
    assert rejected.headers["Retry-After"]


def test_local_mode_is_protected_with_a_fixed_key(client, make_quota, clock, session, settings):
    """认证关闭的本机模式同样要限流，且用固定的本机用户键。"""
    assert settings.auth_enabled is False
    _use_quota(make_quota(clock, quota_user_per_minute=1))
    body = {"session_id": session.session_id, "message": "整体情况"}

    assert client.post("/api/chat", json=body).status_code == 200
    assert client.post("/api/chat", json=body).status_code == 429

    record = _daily(settings, LOCAL_USER_KEY, _day_of(clock()))
    assert record is not None and record.requests == 1


def test_concurrency_is_released_after_each_request(client, make_quota, clock, session):
    """连续请求都必须成功：任何一次结束都要释放并发占用。"""
    _use_quota(make_quota(clock, quota_max_concurrency_per_user=1))
    body = {"session_id": session.session_id, "message": "整体情况"}

    for _ in range(3):
        assert client.post("/api/chat", json=body).status_code == 200


def test_concurrency_is_released_after_llm_error(make_client, make_quota, clock, session):
    client = make_client(BrokenLLM())
    _use_quota(make_quota(clock, quota_max_concurrency_per_user=1))
    body = {"session_id": session.session_id, "message": "整体情况"}

    first = client.post("/api/chat", json=body)
    assert first.status_code == 200
    assert first.json()["stop_reason"] == "llm_error"
    # 失败路径同样要释放，否则用户会被永久锁在自己的并发限制里
    assert client.post("/api/chat", json=body).status_code == 200


async def test_two_concurrent_requests_only_one_passes(
    store, make_agent, make_quota, clock, session
):
    """同一用户的两个并发请求：只能有一个通过。

    这是本工单最关键的一条 —— 检查与占用必须原子，
    否则两个请求会同时读到「并发 0」而一起放行。
    """
    from httpx import ASGITransport, AsyncClient

    from app.dependencies import get_agent, get_quota_service, get_store

    llm = BlockingLLM(delay=0.3)
    quota = make_quota(clock)
    fastapi_app.dependency_overrides[get_store] = lambda: store
    fastapi_app.dependency_overrides[get_agent] = lambda: make_agent(llm)
    fastapi_app.dependency_overrides[get_quota_service] = lambda: quota
    body = {"session_id": session.session_id, "message": "并发提问"}
    try:
        async with AsyncClient(
            transport=ASGITransport(app=fastapi_app), base_url="http://test"
        ) as client:
            first, second = await asyncio.gather(
                client.post("/api/chat", json=body),
                client.post("/api/chat", json=body),
            )
    finally:
        fastapi_app.dependency_overrides.clear()

    statuses = sorted([first.status_code, second.status_code])
    assert statuses == [200, 429], f"只应有一个通过，实际 {statuses}"
    rejected = first if first.status_code == 429 else second
    assert rejected.json()["error"]["code"] == "TOO_MANY_CONCURRENT"


async def test_concurrency_is_released_when_client_disconnects(
    store, make_agent, make_quota, clock, session, settings
):
    """客户端断开 → 任务被取消 → 并发必须释放，否则用户永久被锁。"""
    from app.routers.chat_stream import chat_stream
    from app.schemas import ChatRequest

    llm = BlockingLLM(delay=30)  # 故意挂住，直到被取消
    quota = make_quota(clock, quota_user_per_minute=10)

    response = await chat_stream(
        ChatRequest(session_id=session.session_id, message="慢查询"),
        agent=make_agent(llm),
        user=None,
        store=store,
        quota=quota,
    )
    iterator = response.body_iterator
    assert b"event: start" in await iterator.__anext__()
    await asyncio.wait_for(llm.entered.wait(), timeout=2)

    await iterator.aclose()  # 模拟「停止生成」/ 关闭页面

    for _ in range(100):
        await asyncio.sleep(0.01)
        try:
            probe = quota.reserve(user_key=LOCAL_USER_KEY, session_id=session.session_id)
        except TooManyConcurrentError:
            continue  # 取消还没传播完，继续等
        else:
            break
    else:
        pytest.fail("客户端断开后并发占用没有释放（配额被永久锁死）")
    quota.release(probe)

    # 断开前的那次占用已经计入当日次数（费用确实发生了），探针再 +1
    assert _daily(settings, LOCAL_USER_KEY, _day_of(clock())).requests == 2


# ============================================================ usage 逐轮累加
async def test_usage_is_accumulated_across_rounds(make_agent, session):
    llm = UsageLLM(tool_rounds=2, prompt=10, completion=5)
    result = await make_agent(llm).run(session, "整体情况")

    assert result.stop_reason == "final"
    assert result.usage["prompt_tokens"] == 30  # 3 轮 × 10
    assert result.usage["completion_tokens"] == 15
    assert result.usage["total_tokens"] == 45
    assert result.usage["calls"] == 3
    assert result.usage["usage_missing_calls"] == 0


async def test_usage_includes_max_steps_wrapup_call(make_agent, session, settings):
    """触顶后的收尾调用同样花钱，必须计入。"""
    llm = UsageLLM(tool_rounds=99)  # 永远调工具 → 触发 max_steps 与收尾
    result = await make_agent(llm).run(session, "整体情况")

    assert result.stop_reason == "max_steps"
    expected_calls = settings.max_agent_steps + 1
    assert result.usage["calls"] == expected_calls
    assert result.usage["total_tokens"] == 15 * expected_calls


async def test_usage_counts_missing_rounds_without_estimating(make_agent, session):
    llm = UsageLLM(tool_rounds=2, missing_rounds=(1,))
    result = await make_agent(llm).run(session, "整体情况")

    assert result.usage["calls"] == 3
    assert result.usage["usage_reported_calls"] == 2
    assert result.usage["usage_missing_calls"] == 1
    # 缺失的那一轮**没有**被估算成任何 token
    assert result.usage["total_tokens"] == 30


async def test_usage_is_empty_when_provider_never_reports(make_agent, session):
    """一轮都没拿到 usage 时保持空字典，前端仍显示「暂无用量」（契约不变）。"""
    result = await make_agent(MockLLMClient()).run(session, "整体情况")
    assert result.usage == {}


async def test_usage_is_kept_when_a_later_round_fails(make_agent, session):
    llm = UsageThenBrokenLLM()
    result = await make_agent(llm).run(session, "整体情况")

    assert result.stop_reason == "llm_error"
    # 失败前那一轮真实花掉的用量不能因为失败而被吞掉
    assert result.usage["total_tokens"] == 15
    assert result.usage["calls"] == 1


async def test_streaming_usage_is_counted_at_stream_end(make_agent, session):
    """厂商在流结束时才给 usage（OpenAI 兼容接口的常见行为）。"""
    from stubs import StreamingUsageLLM

    llm = StreamingUsageLLM(tool_rounds=1, prompt=7, completion=3)
    result = await make_agent(llm).run(session, "整体情况", on_event=_noop_event)

    assert result.usage["total_tokens"] == 20  # 2 轮 × 10
    assert result.usage["calls"] == 2


async def _noop_event(event) -> None:  # pragma: no cover - 只为触发流式分支
    return None
