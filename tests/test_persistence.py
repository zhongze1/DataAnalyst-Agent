"""W2-1 会话持久化测试。

核心不变量：
1. 一轮对话结束后消息落盘，**重启可恢复**；
2. 历史列表按最近活跃倒序，且带提问摘要；
3. 旧数据（meta.json）自动迁移且幂等；
4. 存储层不碰用户真实数据目录。
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from app.storage.migrate import META_NAME, migrate_meta_files, needs_migration
from app.storage.repository import MessageRecord, SessionRecord
from app.storage.session_store import SessionStore
from app.storage.sqlite_repo import SqliteSessionRepository


# ============================================================ repository 层
def test_upsert_and_get_session(store, settings) -> None:
    repo = store.repo
    record = SessionRecord(
        session_id="abc123",
        filename="a.xlsx",
        file_path="C:/x/a.xlsx",
        sheets=["S1"],
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )
    repo.upsert_session(record)
    got = repo.get_session("abc123")
    assert got is not None
    assert got.filename == "a.xlsx"
    assert got.sheets == ["S1"]


def test_upsert_is_idempotent(store) -> None:
    repo = store.repo
    for turn in (1, 2, 3):
        repo.upsert_session(
            SessionRecord(
                session_id="dup",
                filename="a.xlsx",
                file_path="C:/x/a.xlsx",
                updated_at=f"2026-01-0{turn}T00:00:00+00:00",
                turn_count=turn,
            )
        )
    assert repo.count_sessions() == 1
    assert repo.get_session("dup").turn_count == 3


def test_messages_roundtrip_preserves_payload(store) -> None:
    """payload 必须无损 —— tool_calls / tool_call_id 丢了，重放时模型接口会报错。"""
    repo = store.repo
    messages = [
        MessageRecord("m1", "s1", "user", {"role": "user", "content": "问题"}),
        MessageRecord(
            "m2",
            "s1",
            "assistant",
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "function": {"name": "x"}}]},
        ),
        MessageRecord(
            "m3",
            "s1",
            "tool",
            {"role": "tool", "tool_call_id": "c1", "name": "x", "content": "结果"},
        ),
    ]
    repo.replace_messages("s1", messages)
    got = repo.get_messages("s1")
    assert [m.role for m in got] == ["user", "assistant", "tool"]
    assert got[1].payload["tool_calls"][0]["id"] == "c1"
    assert got[2].payload["tool_call_id"] == "c1"


def test_replace_messages_is_idempotent(store) -> None:
    """整体替换不应累积重复行。"""
    repo = store.repo
    for _ in range(3):
        repo.replace_messages(
            "s1", [MessageRecord("m1", "s1", "user", {"role": "user", "content": "x"})]
        )
    assert len(repo.get_messages("s1")) == 1


def test_replace_messages_rolls_back_on_error(store) -> None:
    """写入中途失败不能留下半截数据。"""
    repo = store.repo
    repo.replace_messages("s1", [MessageRecord("m1", "s1", "user", {"role": "user", "content": "x"})])
    with pytest.raises(sqlite3.IntegrityError):
        repo.replace_messages(
            "s1",
            [
                MessageRecord("m2", "s1", "user", {"role": "user", "content": "y"}),
                # 主键重复：第二条插入必然失败，用于验证事务整体回滚
                MessageRecord("m2", "s1", "user", {"role": "user", "content": "z"}),
            ],
        )
    # 旧数据应完好
    assert len(repo.get_messages("s1")) == 1


def test_delete_cascades_messages(store) -> None:
    repo = store.repo
    repo.upsert_session(SessionRecord("s1", "a.xlsx", "C:/x/a.xlsx"))
    repo.replace_messages("s1", [MessageRecord("m1", "s1", "user", {"role": "user", "content": "x"})])
    assert repo.delete_session("s1") is True
    assert repo.get_messages("s1") == []
    assert repo.get_session("s1") is None


def test_list_sessions_sorted_by_recency(store) -> None:
    repo = store.repo
    for index, day in enumerate(["01", "02", "03"]):
        repo.upsert_session(
            SessionRecord(
                session_id=f"s{day}",
                filename="a.xlsx",
                file_path="C:/x/a.xlsx",
                updated_at=f"2026-02-{day}T00:00:00+00:00",
            )
        )
    assert [r.session_id for r in repo.list_sessions()] == ["s03", "s02", "s01"]


def test_last_user_messages_batches(store) -> None:
    """历史列表要一次查完所有会话的最后提问，不能 N+1。"""
    repo = store.repo
    for index, sid in enumerate(("a", "b")):
        repo.replace_messages(
            sid,
            [
                MessageRecord(f"{index}-1", sid, "user", {"role": "user", "content": "第一问"}),
                MessageRecord(f"{index}-2", sid, "user", {"role": "user", "content": "第二问"}),
            ],
        )
    previews = repo.last_user_messages(["a", "b", "missing"])
    assert previews["a"] == "第二问"
    assert previews["b"] == "第二问"
    assert "missing" not in previews


# ============================================================ 落盘与恢复
async def test_conversation_survives_restart(make_agent, store, session, settings, sample_xlsx) -> None:
    """核心验收：对话历史落盘，重启（重建 store）后可恢复。"""
    await make_agent().run(session, "第一轮问题")
    await make_agent().run(session, "第二轮问题")
    session_id = session.session_id

    # 模拟重启：全新 repository + 全新 store
    fresh_repo = SqliteSessionRepository(settings.db_path)
    fresh = SessionStore(root=settings.upload_dir, repo=fresh_repo)
    recovered = fresh.get(session_id)

    assert recovered is not None
    assert recovered.turn_count == 2
    user_contents = [m["content"] for m in recovered.messages if m["role"] == "user"]
    assert user_contents == ["第一轮问题", "第二轮问题"]


async def test_tool_calls_survive_restart(make_agent, store, session, settings) -> None:
    """assistant 的 tool_calls 与 tool 的 tool_call_id 必须一起活下来。"""
    await make_agent().run(session, "整体情况")
    fresh = SessionStore(
        root=settings.upload_dir, repo=SqliteSessionRepository(settings.db_path)
    )
    recovered = fresh.get(session.session_id)
    roles = [m["role"] for m in recovered.messages]
    assert "tool" in roles
    tool_messages = [m for m in recovered.messages if m["role"] == "tool"]
    assert all("tool_call_id" in m for m in tool_messages)
    assert any(m.get("tool_calls") for m in recovered.messages if m["role"] == "assistant")


async def test_steps_snapshot_is_persisted(make_agent, store, session) -> None:
    """历史列表要显示「上次用了几步工具」，靠 assistant 消息上的 steps 快照。"""
    await make_agent().run(session, "整体情况")
    records = store.repo.get_messages(session.session_id)
    assistant = [r for r in records if r.role == "assistant"]
    assert assistant and assistant[-1].steps
    assert len(assistant[-1].steps) == 3
    assert assistant[-1].steps[0]["tool"] == "list_sheets"


async def test_reset_clears_persisted_messages(make_agent, store, session, settings) -> None:
    await make_agent().run(session, "问题")
    store.reset_messages(session)
    fresh = SessionStore(
        root=settings.upload_dir, repo=SqliteSessionRepository(settings.db_path)
    )
    recovered = fresh.get(session.session_id)
    assert recovered.messages == []
    assert recovered.turn_count == 0


def test_delete_removes_from_database(store, session, settings) -> None:
    session_id = session.session_id
    store.delete(session_id)
    fresh = SessionStore(
        root=settings.upload_dir, repo=SqliteSessionRepository(settings.db_path)
    )
    assert fresh.get(session_id) is None


# ============================================================ 历史列表
def test_list_sessions_includes_preview(store, session) -> None:
    session.messages = [{"role": "user", "content": "帮我看看销售趋势"}]
    store.save_turn(session)
    listed = store.list_sessions()
    assert listed[0].session_id == session.session_id
    assert listed[0].last_question() == "帮我看看销售趋势"


def test_list_does_not_pollute_cached_session(store, session) -> None:
    """列表页的摘要不能写进缓存里的真实会话对象。"""
    session.messages = [{"role": "user", "content": "真实问题"}]
    store.save_turn(session)
    store.list_sessions()
    assert session.messages == [{"role": "user", "content": "真实问题"}]


def test_get_preserves_object_identity(store, session) -> None:
    """回归：get() 曾经用新对象替换缓存，导致 loop 写进了没人引用的对象。"""
    first = store.get(session.session_id)
    second = store.get(session.session_id)
    assert first is second
    assert first is session  # 与 fixture 持有的是同一个对象


# ============================================================ 旧数据迁移
def test_migration_imports_legacy_meta(tmp_path, settings) -> None:
    upload_dir = tmp_path / "uploads"
    legacy = upload_dir / "oldsession"
    legacy.mkdir(parents=True)
    (legacy / "旧文件.xlsx").write_bytes(b"x")
    (legacy / META_NAME).write_text(
        json.dumps(
            {
                "session_id": "oldsession",
                "filename": "旧文件.xlsx",
                "sheets": ["表一"],
                "created_at": "2026-01-01T00:00:00+00:00",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    repo = SqliteSessionRepository(tmp_path / "m.db")
    assert needs_migration(repo, upload_dir) is True
    assert migrate_meta_files(repo, upload_dir) == 1
    record = repo.get_session("oldsession")
    assert record is not None
    assert record.filename == "旧文件.xlsx"
    assert record.sheets == ["表一"]


def test_migration_is_idempotent(tmp_path) -> None:
    upload_dir = tmp_path / "uploads"
    legacy = upload_dir / "oldsession"
    legacy.mkdir(parents=True)
    (legacy / "旧文件.xlsx").write_bytes(b"x")
    (legacy / META_NAME).write_text(
        json.dumps({"session_id": "oldsession", "filename": "旧文件.xlsx"}), encoding="utf-8"
    )
    repo = SqliteSessionRepository(tmp_path / "m.db")
    assert migrate_meta_files(repo, upload_dir) == 1
    assert migrate_meta_files(repo, upload_dir) == 0  # 第二次无事可做
    assert repo.count_sessions() == 1


def test_migration_skips_when_db_not_empty(tmp_path, settings) -> None:
    upload_dir = tmp_path / "uploads"
    (upload_dir / "x").mkdir(parents=True)
    repo = SqliteSessionRepository(tmp_path / "m.db")
    repo.upsert_session(SessionRecord("existing", "a.xlsx", "C:/a.xlsx"))
    assert needs_migration(repo, upload_dir) is False


def test_migration_skips_broken_meta(tmp_path) -> None:
    upload_dir = tmp_path / "uploads"
    broken = upload_dir / "broken"
    broken.mkdir(parents=True)
    (broken / META_NAME).write_text("{ 坏掉的 json", encoding="utf-8")
    repo = SqliteSessionRepository(tmp_path / "m.db")
    assert migrate_meta_files(repo, upload_dir) == 0  # 不抛异常


def test_migration_skips_folder_without_file(tmp_path) -> None:
    upload_dir = tmp_path / "uploads"
    empty = upload_dir / "empty"
    empty.mkdir(parents=True)
    (empty / META_NAME).write_text(json.dumps({"session_id": "empty"}), encoding="utf-8")
    repo = SqliteSessionRepository(tmp_path / "m.db")
    assert migrate_meta_files(repo, upload_dir) == 0


# ============================================================ 数据隔离
def test_runtime_paths_stay_out_of_repo(settings) -> None:
    """回归：db / audit / uploads 都必须落在 tmp_path，不能写进仓库的 data/。

    这条曾经真实发生过 —— 审计功能上线时，测试往用户的 data/audit
    写了 89KB 垃圾数据。
    """
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    for path in (settings.db_path, settings.audit_dir, settings.upload_dir):
        resolved = Path(path).resolve()
        assert repo_root not in resolved.parents, f"{resolved} 落在仓库内"
        assert resolved != repo_root
