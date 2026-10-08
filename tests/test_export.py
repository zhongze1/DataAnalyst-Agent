"""W2-2 分析报告导出测试。

**所有涉及文件系统的用例都走 tmp_path**（由 settings.upload_dir /
db_path 夹具隔离），绝不碰仓库的 data/ 目录 —— 这条纪律已经被违反过两次
（审计功能上线时往用户 data/audit 写了 89KB）。

覆盖：产物落盘、Markdown/Excel 生成、公式注入防护、redact 钩子接入。
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from app.security.redact import SENSITIVE_COLUMNS, redact_frame, redact_text
from app.services.export_service import (
    _excel_safe,
    _sheet_title,
    build_markdown,
    build_xlsx,
    collect_turns,
)
from app.storage.artifacts import (
    MAX_ARTIFACT_COLS,
    MAX_ARTIFACT_ROWS,
    artifacts_dir,
    list_artifacts,
    load_artifact,
    save_artifact,
)


@pytest.fixture
def session_file(tmp_path):
    """一个「会话文件路径」。不真正创建 xlsx —— 产物逻辑只关心所在目录。"""
    folder = tmp_path / "sess123"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "数据.xlsx"
    target.write_bytes(b"stub")
    return target


@pytest.fixture
def sample_turns():
    return [{
        "question": "各区域销售额如何？",
        "answer": "华东最高。\n\n| 区域 | 销售额 |\n| --- | --- |\n| 华东 | 100 |",
        "steps": [{"step": 1, "tool": "load_sheet", "arguments": {"sheet": "销售明细"},
                   "ok": True, "result": "已切换", "error": None}],
    }]


# ============================================================ 产物落盘
def test_dataframe_is_saved_as_csv(session_file) -> None:
    frame = pd.DataFrame({"区域": ["华东", "华北"], "销售额": [100.0, 200.0]})
    ref = save_artifact(session_file, "run_pandas", frame, step=3)

    assert ref is not None and ref.saved
    assert ref.step == 3
    assert ref.rows == 2 and ref.cols == 2
    directory = artifacts_dir(session_file)
    assert (directory / ref.filename).exists()
    assert (directory / "manifest.json").exists()

    back = load_artifact(session_file, ref)
    assert back is not None
    assert list(back.columns) == ["区域", "销售额"]
    assert back["销售额"].tolist() == [100.0, 200.0]


def test_series_is_saved(session_file) -> None:
    ref = save_artifact(session_file, "run_pandas", pd.Series([1, 2, 3], name="a"), step=1)
    assert ref is not None and ref.saved


def test_csv_uses_utf8_sig_so_excel_does_not_mojibake(session_file) -> None:
    """Excel 不认无BOM 的 UTF-8 CSV，中文会变乱码 —— 最常见的坑。"""
    ref = save_artifact(session_file, "run_pandas", pd.DataFrame({"门店": ["西安"]}), step=1)
    raw = (artifacts_dir(session_file) / ref.filename).read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf"), "缺少 UTF-8 BOM"
    assert "西安".encode("utf-8") in raw


def test_non_tabular_result_is_not_saved(session_file) -> None:
    """dict / 标量落盘成 CSV 没有意义，不应产生文件。"""
    assert save_artifact(session_file, "run_pandas", {"shape": (10, 2)}) is None
    assert save_artifact(session_file, "run_pandas", 42) is None
    assert save_artifact(session_file, "run_pandas", "一段文本") is None
    assert not artifacts_dir(session_file).exists()


def test_empty_frame_is_skipped(session_file) -> None:
    assert save_artifact(session_file, "run_pandas", pd.DataFrame(), step=1) is None


def test_oversized_result_is_recorded_but_not_written(session_file) -> None:
    """超限要**如实标注**，不能静默消失 —— 用户需要知道数据没导出去。"""
    big = pd.DataFrame({"a": range(MAX_ARTIFACT_ROWS + 1)})
    ref = save_artifact(session_file, "run_pandas", big, step=2)

    assert ref is not None
    assert ref.saved is False
    assert ref.status == "too_large"
    assert "未导出" in ref.reason
    assert ref.rows == MAX_ARTIFACT_ROWS + 1
    assert [r.status for r in list_artifacts(session_file)] == ["too_large"]


def test_too_many_columns_is_recorded(session_file) -> None:
    wide = pd.DataFrame({f"c{i}": [1] for i in range(MAX_ARTIFACT_COLS + 1)})
    ref = save_artifact(session_file, "run_pandas", wide, step=1)
    assert ref is not None and ref.status == "too_large"


def test_sequence_increments_across_calls(session_file) -> None:
    first = save_artifact(session_file, "run_pandas", pd.DataFrame({"a": [1]}), step=1)
    second = save_artifact(session_file, "run_pandas", pd.DataFrame({"a": [2]}), step=2)
    assert first.seq == 1 and second.seq == 2


def test_list_artifacts_recovers_when_manifest_lost(session_file) -> None:
    """清单是辅助文件，丢了不该连带丢数据。"""
    ref = save_artifact(session_file, "run_pandas", pd.DataFrame({"a": [1]}), step=1)
    (artifacts_dir(session_file) / "manifest.json").unlink()

    recovered = list_artifacts(session_file)
    assert len(recovered) == 1
    assert recovered[0].filename == ref.filename


def test_load_artifact_tolerates_missing_file(session_file) -> None:
    ref = save_artifact(session_file, "run_pandas", pd.DataFrame({"a": [1]}), step=1)
    (artifacts_dir(session_file) / ref.filename).unlink()
    assert load_artifact(session_file, ref) is None


def test_corrupt_manifest_is_tolerated(session_file) -> None:
    save_artifact(session_file, "run_pandas", pd.DataFrame({"a": [1]}), step=1)
    (artifacts_dir(session_file) / "manifest.json").write_text("{ 坏 json", encoding="utf-8")
    assert len(list_artifacts(session_file)) == 1


# ============================================================ redact钩子
def test_redact_is_identity_by_default(session_file) -> None:
    """W2-4 还没做，当前必须是恒等 —— 但链路必须已经接上。

    注意：整列都是数字时 pandas 会读回 int64（CSV 不带类型信息），
    所以断言用 str() 比较而不是类型比较。
    """
    frame = pd.DataFrame({"手机号": ["13800138000"]})
    ref = save_artifact(session_file, "run_pandas", frame, step=1)
    back = load_artifact(session_file, ref)
    assert [str(v) for v in back["手机号"]] == ["13800138000"]


def test_redact_hook_is_applied_on_save(session_file, monkeypatch) -> None:
    """加规则后，落盘产物必须已脱敏 —— 否则库里躺着明文。"""
    import app.storage.artifacts as artifacts_mod

    monkeypatch.setattr(
        artifacts_mod, "redact_frame", lambda f: f.assign(手机号="***")
    )
    frame = pd.DataFrame({"手机号": ["13800138000"], "城市": ["西安"]})
    ref = save_artifact(session_file, "run_pandas", frame, step=1)
    back = load_artifact(session_file, ref)
    assert back["手机号"].tolist() == ["***"]
    assert back["城市"].tolist() == ["西安"]


def test_redact_frame_masks_by_column_name() -> None:
    SENSITIVE_COLUMNS.add("身份证")
    try:
        masked = redact_frame(pd.DataFrame({"身份证": ["1101011990..."], "城市": ["西安"]}))
        assert masked["身份证"].tolist() == ["***"]
        assert masked["城市"].tolist() == ["西安"]
    finally:
        SENSITIVE_COLUMNS.discard("身份证")


def test_redact_text_applies_patterns() -> None:
    import re

    import app.security.redact as redact_mod

    redact_mod.PATTERNS.append((re.compile(r"1\d{10}"), "1**********"))
    try:
        assert redact_text("联系方式 13800138000") == "联系方式 1**********"
    finally:
        redact_mod.PATTERNS.clear()


# ============================================================ 轮次拆分
def _msg(role, content, steps=None):
    message = {"role": role, "content": content}
    if steps is not None:
        message["steps"] = steps
    return message


def test_collect_turns_pairs_questions_with_answers() -> None:
    turns = collect_turns([
        _msg("user", "问题一"),
        _msg("assistant", "答案一", [{"step": 1, "tool": "list_sheets"}]),
        _msg("user", "问题二"),
        _msg("assistant", "答案二"),
    ])
    assert len(turns) == 2
    assert turns[0]["question"] == "问题一"
    assert turns[0]["answer"] == "答案一"
    assert turns[1]["question"] == "问题二"


def test_tool_only_assistant_turn_is_merged_into_next() -> None:
    """只调工具没文字的中间轮次不应单独成轮。"""
    turns = collect_turns([
        _msg("user", "问题"),
        _msg("assistant", "", [{"step": 1}]),
        _msg("assistant", "最终答案"),
    ])
    assert len(turns) == 1
    assert turns[0]["answer"] == "最终答案"


def test_question_without_answer_still_produces_turn() -> None:
    """撞上 max_steps 时模型可能没给文字结论，仍要保留提问。"""
    turns = collect_turns([_msg("user", "问题")])
    assert len(turns) == 1
    assert turns[0]["question"] == "问题"
    assert turns[0]["answer"] == ""


def test_empty_history_yields_no_turns() -> None:
    assert collect_turns([]) == []


# ============================================================ Markdown
def test_markdown_contains_conclusion_and_question(sample_turns) -> None:
    bundle = build_markdown(
        filename="销售数据.xlsx", session_id="abc123", turns=sample_turns,
        artifacts=[], session_path=None, generated_at="2026-10-08 12:00",
    )
    text = bundle.content.decode("utf-8")
    assert "分析报告 · 销售数据.xlsx" in text
    assert "各区域销售额如何？" in text
    assert "华东最高" in text
    assert "abc123" in text
    assert bundle.filename.endswith(".md")


def test_markdown_only_last_turn(sample_turns) -> None:
    turns = sample_turns + [{"question": "第二个问题", "answer": "第二个答案", "steps": []}]
    bundle = build_markdown(
        filename="a.xlsx", session_id="s", turns=turns, artifacts=[],
        session_path=None, generated_at="now", only_last_turn=True,
    )
    text = bundle.content.decode("utf-8")
    assert "第二个答案" in text
    assert "各区域销售额如何" not in text


def test_markdown_handles_empty_history() -> None:
    bundle = build_markdown(
        filename="a.xlsx", session_id="s", turns=[], artifacts=[],
        session_path=None, generated_at="now",
    )
    assert "还没有分析结论" in bundle.content.decode("utf-8")


# ============================================================ Excel
def test_xlsx_has_expected_sheets(sample_turns) -> None:
    from openpyxl import load_workbook

    bundle = build_xlsx(
        filename="a.xlsx", session_id="s", turns=sample_turns, artifacts=[],
        session_path=None, generated_at="now",
    )
    book = load_workbook(io.BytesIO(bundle.content))
    assert book.sheetnames == ["结论", "执行轨迹"]
    assert bundle.filename.endswith(".xlsx")


def test_xlsx_includes_artifact_sheets(session_file, sample_turns) -> None:
    from openpyxl import load_workbook

    frame = pd.DataFrame({"区域": ["华东", "华北"], "销售额": [100.0, 200.0]})
    ref = save_artifact(session_file, "run_pandas", frame, step=3)
    assert ref is not None

    bundle = build_xlsx(
        filename="a.xlsx", session_id="s", turns=sample_turns,
        artifacts=list_artifacts(session_file), session_path=session_file,
        generated_at="now",
    )
    book = load_workbook(io.BytesIO(bundle.content))
    data_sheet = [n for n in book.sheetnames if n.startswith("数据-")]
    assert len(data_sheet) == 1
    rows = list(book[data_sheet[0]].iter_rows(values_only=True))
    assert rows[0] == ("区域", "销售额")
    assert rows[1] == ("华东", 100.0)


def test_xlsx_sheets_explain_oversized_artifacts(session_file, sample_turns) -> None:
    """超限产物要有「未导出说明」sheet，而不是让人以为漏了。"""
    from openpyxl import load_workbook

    big = pd.DataFrame({"a": range(MAX_ARTIFACT_ROWS + 1)})
    save_artifact(session_file, "run_pandas", big, step=1)

    bundle = build_xlsx(
        filename="a.xlsx", session_id="s", turns=sample_turns,
        artifacts=list_artifacts(session_file), session_path=session_file,
        generated_at="now",
    )
    book = load_workbook(io.BytesIO(bundle.content))
    assert "未导出说明" in book.sheetnames


def test_xlsx_trace_sheet_lists_steps(sample_turns) -> None:
    from openpyxl import load_workbook

    bundle = build_xlsx(
        filename="a.xlsx", session_id="s", turns=sample_turns, artifacts=[],
        session_path=None, generated_at="now",
    )
    book = load_workbook(io.BytesIO(bundle.content))
    rows = list(book["执行轨迹"].iter_rows(values_only=True))
    assert rows[0][:3] == ("轮次", "步骤", "工具")
    assert rows[1][2] == "load_sheet"
    assert rows[1][4] == "成功"


# ============================================ Excel 安全与兼容性
@pytest.mark.parametrize(
    "value", ["=cmd|' /C calc'!A1", "+1+1", "-1+1", "@SUM(A1)"]
)
def test_formula_injection_is_neutralized(value: str) -> None:
    """以 = + - @ 开头的单元格会被 Excel 当公式执行，可读取本地文件。"""
    assert _excel_safe(value).startswith("'")


def test_normal_values_are_untouched() -> None:
    assert _excel_safe("华东") == "华东"
    assert _excel_safe(100) == 100
    assert _excel_safe(None) is None


def test_sheet_title_respects_excel_limits() -> None:
    assert len(_sheet_title("数据-" + "x" * 100)) <= 31
    for char in ':\\/?*[]':
        assert char not in _sheet_title(f"a{char}b")


# ============================================================ HTTP 层
def test_export_markdown_endpoint(client, session) -> None:
    client.post("/api/chat", json={"session_id": session.session_id, "message": "整体情况"})
    response = client.get(f"/api/sessions/{session.session_id}/export?format=md")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "attachment" in response.headers["content-disposition"]
    # 中文文件名要按 RFC 5987 编码，否则部分浏览器乱码
    assert "filename*=UTF-8''" in response.headers["content-disposition"]
    assert "分析报告" in response.content.decode("utf-8")


def test_export_xlsx_endpoint(client, session) -> None:
    client.post("/api/chat", json={"session_id": session.session_id, "message": "整体情况"})
    response = client.get(f"/api/sessions/{session.session_id}/export?format=xlsx")

    assert response.status_code == 200
    assert "spreadsheetml" in response.headers["content-type"]
    assert response.content[:2] == b"PK"  # xlsx 是 zip


def test_export_last_turn_flag(client, session) -> None:
    client.post("/api/chat", json={"session_id": session.session_id, "message": "第一轮"})
    client.post("/api/chat", json={"session_id": session.session_id, "message": "第二轮"})

    whole = client.get(f"/api/sessions/{session.session_id}/export?format=md")
    assert "第 1 轮" in whole.content.decode("utf-8")
    assert "第 2 轮" in whole.content.decode("utf-8")

    last = client.get(
        f"/api/sessions/{session.session_id}/export?format=md&last_turn=true"
    )
    text = last.content.decode("utf-8")
    assert "第 1 轮" not in text
    assert "结论" in text


def test_export_rejects_unknown_format(client, session) -> None:
    response = client.get(f"/api/sessions/{session.session_id}/export?format=pdf")
    assert response.status_code == 400


def test_export_unknown_session_is_404(client) -> None:
    assert client.get("/api/sessions/nope/export").status_code == 404


def test_export_requires_login_when_auth_enabled(auth_client) -> None:
    response = auth_client.get("/api/sessions/abc/export?format=md")
    assert response.status_code == 401


def test_cannot_export_other_users_session(auth_client, auth_store, auth_service) -> None:
    """归属隔离：导出是唯一会把原始数据整表带出去的链路，必须校验。"""
    admin = auth_service.repo.get_user_by_username("admin")
    his = auth_store.create("管理员的数据.xlsx", _xlsx_bytes(), owner_id=admin.user_id)

    auth_client.post(
        "/api/auth/login", json={"username": "admin", "password": "admin-pass-123"}
    )
    assert auth_client.get(f"/api/sessions/{his.session_id}/export").status_code == 200

    auth_client.cookies.clear()
    auth_service.create_user("bob", "bob-pass-1234")
    auth_client.post(
        "/api/auth/login", json={"username": "bob", "password": "bob-pass-1234"}
    )
    assert auth_client.get(f"/api/sessions/{his.session_id}/export").status_code == 404


def test_session_deletion_removes_artifacts_dir(client, session, store) -> None:
    """产物目录是子目录，unlink 对它无效 —— 曾经会导致删除残留。"""
    from app.storage.artifacts import artifacts_dir

    folder = session.path.parent
    (folder / "artifacts").mkdir(parents=True, exist_ok=True)
    (folder / "artifacts" / "001-run_pandas.csv").write_text("a", encoding="utf-8")
    assert artifacts_dir(session.path).exists()

    client.delete(f"/api/files/{session.session_id}")
    assert not folder.exists(), "会话目录应被完整清理（含 artifacts 子目录）"


def _xlsx_bytes() -> bytes:
    import pandas as pd

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame({"a": [1, 2]}).to_excel(writer, sheet_name="S", index=False)
    return buffer.getvalue()
