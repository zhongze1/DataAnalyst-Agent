"""配额与并发保护（DA-017 / PRD W2-3）。

**为什么所有方法都是同步的**（这是本模块最重要的设计决定）：

1. *原子性*：决策要求「同一用户的并发检查与占用必须原子」。
   asyncio 只在 ``await`` 处切换协程，同步方法内部不会被切走，
   于是「读计数 → 判断 → 占用」天然是一个不可分割的临界区；
   若写成 ``async def`` 并在中间 ``await`` 一次数据库，两个请求就会
   同时通过检查 —— 这正是决策里点名要避免的。
2. *取消安全*：请求被取消（用户点「停止生成」/ 断开连接）时
   ``CancelledError`` 会在传播路径上让任何 ``await`` 立即失败，
   于是「释放并发」与「结算用量」就会丢掉，配额被永久锁死。
   同步方法不受取消影响，一定执行完。

分层：
- **分钟窗口**与**当前并发**是进程内状态（短期、可容忍重启清零）；
- **每日请求计数**与**token 用量**落 SQLite（需跨重启保留）。
- 只支持单进程部署：进程内状态不跨 worker 共享。
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Any
from zoneinfo import ZoneInfo

from app.audit import record_quota_event
from app.config import Settings
from app.errors import QuotaExceededError, ServerBusyError, TooManyConcurrentError
from app.storage.repository import QuotaRepository
from app.storage.sqlite_quota_repo import SqliteQuotaRepository

logger = logging.getLogger(__name__)

# 认证关闭（本机模式）时使用的固定配额键：没有它，本机模式就等于没有限流
LOCAL_USER_KEY = "local"

# 只认这三个 OpenAI 通用字段；厂商返回别的字段一律忽略，绝不猜测
_TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")

Clock = Callable[[], datetime]


def resolve_tz(name: str) -> tzinfo:
    """把配置里的时区名解析成 tzinfo。

    **解析失败只告警并回退 UTC，绝不抛异常**：否则一个写错的时区名
    就会让服务起不来，违背「升级后不会无法启动」的约束。
    （Windows 上 zoneinfo 依赖 tzdata，缺失时也走这条回退路径。）
    """
    if not name or name.strip().upper() == "UTC":
        return timezone.utc
    try:
        return ZoneInfo(name.strip())
    except Exception:  # noqa: BLE001 - KeyError/ValueError/OSError 都视为不可用
        logger.warning(
            "配额时区 %r 无法解析，回退 UTC（装上 tzdata 或改用 IANA 时区名，如 Asia/Shanghai）",
            name,
            extra={"error_code": "QUOTA_TZ_FALLBACK"},
        )
        return timezone.utc


@dataclass
class QuotaReservation:
    """一次成功占用的凭据。

    ``counted=False`` 表示配额功能已关闭、本次没有占用任何计数，
    此时 :meth:`QuotaService.release` / :meth:`QuotaService.settle` 直接跳过。
    """

    user_key: str
    session_id: str | None = None
    # 占用时所在的自然日：结算必须写回**同一天**，哪怕请求跨过了零点
    day: str = ""
    counted: bool = False
    released: bool = False


class QuotaService:
    """请求次数硬配额 + 并发保护 + 用量结算。"""

    def __init__(
        self,
        settings: Settings,
        repo: QuotaRepository | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._settings = settings
        self._repo = repo if repo is not None else SqliteQuotaRepository(settings.db_path)
        self._clock: Clock = clock or (lambda: datetime.now(timezone.utc))
        self._tz = resolve_tz(settings.quota_timezone)
        # 防御性加锁：正常路径都在事件循环线程里调用，同步方法本身已原子；
        # 但若将来被线程池/同步路由复用，这把锁能保证同一件事依然成立
        self._lock = threading.Lock()
        # (维度, 键, 分钟桶) -> 次数；bucket 让窗口随分钟自然滚动
        self._minute: dict[tuple[str, str, int], int] = {}
        self._active_users: dict[str, int] = {}
        self._active_global = 0

    @property
    def timezone(self) -> tzinfo:
        """每日额度使用的时区（文案与判定共用它，保证口径一致）。"""
        return self._tz

    # ------------------------------------------------------------ 预检 + 占用
    def reserve(self, *, user_key: str, session_id: str | None = None) -> QuotaReservation:
        """预检所有限制，全部通过后**一次性**占用计数与并发。

        任何一项不通过都直接抛异常，且**不留部分占用**（不计数、不占并发），
        所以被拒绝的请求不会吃掉额度，也不会让并发泄漏。

        Raises:
            QuotaExceededError: 次数超限（用户分钟 / 会话分钟 / 用户每日）。
            TooManyConcurrentError: 该用户已有请求在执行。
            ServerBusyError: 全局并发已满。
        """
        if not self._settings.quota_enabled:
            return QuotaReservation(user_key=user_key, session_id=session_id)

        settings = self._settings
        now = self._clock()
        bucket = self._bucket(now)
        day = self._day(now)

        with self._lock:
            self._prune(bucket)

            # ---- 只读检查：先查内存（便宜），再查库（每日额度）----
            used_minute = self._minute.get((_USER, user_key, bucket), 0)
            if used_minute >= settings.quota_user_per_minute:
                wait = self._seconds_to_next_minute(now)
                raise self._reject(
                    "user_per_minute",
                    user_key,
                    settings.quota_user_per_minute,
                    wait,
                    f"提问过于频繁（每分钟最多 {settings.quota_user_per_minute} 次），"
                    f"请 {wait} 秒后再试",
                )

            if session_id:
                session_used = self._minute.get((_SESSION, session_id, bucket), 0)
                if session_used >= settings.quota_session_per_minute:
                    wait = self._seconds_to_next_minute(now)
                    raise self._reject(
                        "session_per_minute",
                        user_key,
                        settings.quota_session_per_minute,
                        wait,
                        f"该会话提问过于频繁（每分钟最多 {settings.quota_session_per_minute} 次），"
                        f"请 {wait} 秒后再试",
                    )

            if self._active_users.get(user_key, 0) >= settings.quota_max_concurrency_per_user:
                raise self._reject(
                    "user_concurrent",
                    user_key,
                    settings.quota_max_concurrency_per_user,
                    settings.quota_retry_after_seconds,
                    "你有一个分析正在进行，请等它结束后再提问",
                    error_cls=TooManyConcurrentError,
                )

            if self._active_global >= settings.quota_max_concurrency_global:
                raise self._reject(
                    "global_busy",
                    user_key,
                    settings.quota_max_concurrency_global,
                    settings.quota_retry_after_seconds,
                    "服务器当前繁忙，请稍后重试",
                    error_cls=ServerBusyError,
                )

            record = self._repo.get_daily(user_key, day)
            if record is not None and record.requests >= settings.quota_user_per_day:
                reset_at = self._next_day_start(now)
                wait = self._seconds_until(now, reset_at)
                raise self._reject(
                    "user_per_day",
                    user_key,
                    settings.quota_user_per_day,
                    wait,
                    f"今日提问额度已用完（每天最多 {settings.quota_user_per_day} 次），"
                    f"将于 {reset_at.strftime('%m-%d %H:%M')} 重置",
                )

            # ---- 全部通过，一次性占用 ----
            # 先落库：自增是准入的真相来源，写失败就不该让请求继续跑（fail-closed）
            total = self._repo.increment_requests(user_key, day)
            self._minute[(_USER, user_key, bucket)] = used_minute + 1
            if session_id:
                key = (_SESSION, session_id, bucket)
                self._minute[key] = self._minute.get(key, 0) + 1
            self._active_users[user_key] = self._active_users.get(user_key, 0) + 1
            self._active_global += 1

        logger.debug(
            "配额占用 | 当日第 %d 次 | 全局并发 %d",
            total,
            self._active_global,
            extra={"quota_dimension": "reserve"},
        )
        return QuotaReservation(user_key=user_key, session_id=session_id, day=day, counted=True)

    # ------------------------------------------------------------ 释放
    def release(self, reservation: QuotaReservation | None) -> None:
        """释放并发占用。**幂等**，且可以在取消路径上安全调用。

        必须在 ``finally`` 里调用：正常结束、抛异常、被取消、客户端断开
        四条路径都要走到，否则并发会被永久锁死。
        """
        if reservation is None or not reservation.counted:
            return
        with self._lock:
            if reservation.released:
                return
            reservation.released = True
            remaining = self._active_users.get(reservation.user_key, 0) - 1
            if remaining > 0:
                self._active_users[reservation.user_key] = remaining
            else:
                # 归零即删键：长期运行下字典不能无限增长
                self._active_users.pop(reservation.user_key, None)
                if remaining < 0:
                    logger.warning(
                        "并发计数出现负值，疑似重复释放 | user=%s",
                        reservation.user_key,
                        extra={"error_code": "QUOTA_RELEASE_UNDERFLOW"},
                    )
            self._active_global = max(0, self._active_global - 1)

    # ------------------------------------------------------------ 结算
    def settle(self, reservation: QuotaReservation | None, usage: dict[str, Any] | None) -> None:
        """把本轮真实获得的 usage 累加落库。**失败只记日志**。

        与 ``reserve`` 相反，这里是**观测**而不是准入：统计写失败绝不能
        连带用户的请求失败。缺失的 usage 一律**不估算**，只记轮次。
        """
        if reservation is None or not reservation.counted:
            return
        numbers = _usage_numbers(usage)
        reported, missing = _usage_call_counts(usage)
        try:
            self._repo.add_tokens(
                reservation.user_key,
                reservation.day,
                prompt_tokens=numbers.get("prompt_tokens", 0),
                completion_tokens=numbers.get("completion_tokens", 0),
                total_tokens=numbers.get("total_tokens", 0),
                reported_calls=reported,
                missing_calls=missing,
            )
        except Exception:
            logger.exception("用量结算失败（不影响本次请求） | user=%s", reservation.user_key)
            return

        logger.info(
            "用量结算 | prompt=%d completion=%d total=%d | 回报轮次=%d 缺失轮次=%d",
            numbers.get("prompt_tokens", 0),
            numbers.get("completion_tokens", 0),
            numbers.get("total_tokens", 0),
            reported,
            missing,
        )
        record_quota_event(
            event="quota_settled",
            dimension="usage",
            user_key=reservation.user_key,
            usage={
                "prompt_tokens": numbers.get("prompt_tokens", 0),
                "completion_tokens": numbers.get("completion_tokens", 0),
                "total_tokens": numbers.get("total_tokens", 0),
                "reported_calls": reported,
                "missing_calls": missing,
            },
        )

    # ------------------------------------------------------------ 内部
    def _reject(
        self,
        dimension: str,
        user_key: str,
        limit: int,
        retry_after: int,
        message: str,
        *,
        error_cls: type[QuotaExceededError] = QuotaExceededError,
    ) -> QuotaExceededError:
        """构造拒绝异常，并留下日志 + 审计。返回异常而不是抛出，便于 ``raise``。

        文案里的等待时间与 ``retry_after`` 由同一个时刻算出，
        保证用户看到的「请 N 秒后再试」与响应头完全一致。
        """
        logger.warning(
            "配额拒绝 | 维度=%s | 上限=%d | 建议等待 %ss",
            dimension,
            limit,
            retry_after,
            extra={"error_code": error_cls.code, "quota_dimension": dimension},
        )
        record_quota_event(
            event="quota_rejected",
            dimension=dimension,
            user_key=user_key,
            limit=limit,
            retry_after=retry_after,
        )
        return error_cls(message, retry_after=retry_after)

    @staticmethod
    def _bucket(now: datetime) -> int:
        """分钟桶：同一分钟内的请求共享一个桶，桶一变窗口就滚过去了。"""
        return int(now.timestamp() // 60)

    def _day(self, now: datetime) -> str:
        """配额时区下的自然日（``YYYY-MM-DD``）。"""
        return now.astimezone(self._tz).strftime("%Y-%m-%d")

    def _prune(self, current_bucket: int) -> None:
        """丢掉过期的分钟桶，避免长期运行后内存只增不减。"""
        stale = [key for key in self._minute if key[2] < current_bucket]
        for key in stale:
            del self._minute[key]

    @staticmethod
    def _seconds_to_next_minute(now: datetime) -> int:
        return max(1, math.ceil(60 - now.timestamp() % 60))

    def _next_day_start(self, now: datetime) -> datetime:
        """下一个自然日的零点（配额时区，仍是 tz-aware）。"""
        local = now.astimezone(self._tz)
        return (local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    @staticmethod
    def _seconds_until(now: datetime, moment: datetime) -> int:
        return max(1, math.ceil((moment - now).total_seconds()))


_USER = "user"
_SESSION = "session"


def _usage_numbers(usage: dict[str, Any] | None) -> dict[str, int]:
    """抽出可用的 token 数字。

    只接受非负整数：厂商给字符串/浮点/None 一律视为「没有这个字段」，
    拒绝猜测 —— 统计宁可少，也不能把估算值伪装成精确 usage。
    """
    numbers: dict[str, int] = {}
    source = usage or {}
    for key in _TOKEN_KEYS:
        value = source.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        numbers[key] = max(0, value)
    return numbers


def _usage_call_counts(usage: dict[str, Any] | None) -> tuple[int, int]:
    """返回 ``(回报了 usage 的轮次, 未回报的轮次)``。

    整轮都没有 usage 时无从得知实际轮次，按「至少 1 轮未回报」计数：
    这是**轮次**统计，不是 token 估算，不会污染费用口径。
    """
    source = usage or {}

    def _int(key: str) -> int:
        value = source.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            return 0
        return max(0, value)

    reported, missing = _int("usage_reported_calls"), _int("usage_missing_calls")
    if reported == 0 and missing == 0:
        missing = 1
    return reported, missing
