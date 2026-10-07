"""存储层测试：格式校验、原子性、持久化恢复。"""

from __future__ import annotations

import pandas as pd
import pytest

from app.errors import FileEmptyError, FileParseError, FileTooLargeError, FileTypeError
from app.storage.session_store import META_NAME, SessionStore


def test_create_writes_file_and_record(store: SessionStore, sample_xlsx: bytes) -> None:
    """W2-1：元数据进数据库，不再写 meta.json（那只是旧数据迁移的产物）。"""
    session = store.create("销售数据.xlsx", sample_xlsx)
    assert session.path.exists()
    assert session.path.read_bytes() == sample_xlsx
    assert session.sheets == ["明细", "摘要"]
    assert session.size_bytes == len(sample_xlsx)

    record = store.repo.get_session(session.session_id)
    assert record is not None
    assert record.filename == "销售数据.xlsx"
    assert record.sheets == ["明细", "摘要"]
    assert record.turn_count == 0
    assert record.file_path == str(session.path)
    # 新代码不再产生 meta.json
    assert not (session.path.parent / META_NAME).exists()


def test_rejects_unsupported_suffix(store: SessionStore, sample_xlsx: bytes) -> None:
    with pytest.raises(FileTypeError, match="暂不支持的文件类型"):
        store.create("data.csv", sample_xlsx)


def test_rejects_empty_content(store: SessionStore) -> None:
    with pytest.raises(FileEmptyError):
        store.create("empty.xlsx", b"")


def test_rejects_oversized_file(store: SessionStore, settings, sample_xlsx: bytes) -> None:
    settings.max_upload_bytes = 10
    with pytest.raises(FileTooLargeError):
        store.create("big.xlsx", sample_xlsx)


def test_parse_failure_leaves_no_dirty_session(store: SessionStore, garbage_bytes: bytes) -> None:
    """解析失败必须清理干净，不留半截目录污染会话列表。"""
    with pytest.raises(FileParseError):
        store.create("fake.xlsx", garbage_bytes)
    assert store.list_sessions() == []


def test_get_and_delete(store: SessionStore, sample_xlsx: bytes) -> None:
    session = store.create("a.xlsx", sample_xlsx)
    assert store.get(session.session_id) is session
    assert store.get("不存在") is None
    assert store.get(None) is None

    folder = session.path.parent
    assert store.delete(session.session_id) is True
    assert not folder.exists()
    assert store.delete(session.session_id) is False


def test_reset_messages_keeps_data(store: SessionStore, sample_xlsx: bytes) -> None:
    session = store.create("a.xlsx", sample_xlsx)
    session.messages.append({"role": "user", "content": "hi"})
    store.reset_messages(session)
    assert session.messages == []
    assert session.path.exists()


def test_sessions_survive_restart(store: SessionStore, settings, sample_xlsx: bytes) -> None:
    """新进程重建 SessionStore 后应能恢复文件索引（对话历史不恢复）。"""
    created = store.create("a.xlsx", sample_xlsx)
    reopened = SessionStore(root=settings.upload_dir)
    restored = reopened.get(created.session_id)
    assert restored is not None
    assert restored.filename == "a.xlsx"
    assert restored.sheets == ["明细", "摘要"]
    assert restored.messages == []
    assert list(reopened.list_sessions())[0].created_at == created.created_at


def test_load_all_skips_broken_dirs(store: SessionStore, settings) -> None:
    """损坏的 meta.json 或空目录应被跳过，不能让启动崩溃。"""
    (settings.upload_dir / "broken").mkdir(parents=True, exist_ok=True)
    (settings.upload_dir / "broken" / META_NAME).write_text("{ not json", encoding="utf-8")
    (settings.upload_dir / "no_meta").mkdir(parents=True, exist_ok=True)
    (settings.upload_dir / "no_meta" / "x.xlsx").write_bytes(b"whatever")

    reopened = SessionStore(root=settings.upload_dir)
    assert reopened.list_sessions() == []


def test_read_frame_and_sheet_names(store: SessionStore, sample_xlsx: bytes) -> None:
    session = store.create("a.xlsx", sample_xlsx)
    assert SessionStore.list_sheet_names(session.path) == ["明细", "摘要"]
    frame = SessionStore.read_frame(session.path, "明细")
    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == (22, 6)
    assert "备注" in frame.columns


def test_list_sessions_sorted_by_time(store: SessionStore, sample_xlsx: bytes) -> None:
    first = store.create("a.xlsx", sample_xlsx)
    second = store.create("b.xlsx", sample_xlsx)
    ordered = store.list_sessions()
    assert {s.session_id for s in ordered} == {first.session_id, second.session_id}
    assert ordered[0].created_at >= ordered[-1].created_at
