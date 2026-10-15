"""RAG-02 分块器测试。

覆盖要求里的每一条：正常分块、标题层级、空文档、超长段落、重叠、元数据、
JSONL 序列化、来源追溯，以及对复制语料的来源核对。

核心不变量（最容易出错、也最重要）：
**片段的 text 必须是源文件 [line_start, line_end] 的连续切片** ——
唯一例外是"单行超长被字符硬切"，此时 text 是该行的子串，且同行的切片拼起来能还原原文。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from rag_lab.chunker import (
    DEFAULT_CORPUS_DIR,
    DEFAULT_OUT_FILE,
    LAB_ROOT,
    NO_SECTION,
    ChunkConfig,
    build_report,
    iter_corpus_files,
    main,
    split_corpus,
    split_document,
    write_jsonl,
)

REPO_ROOT = LAB_ROOT.parent
CORPUS_DIR = LAB_ROOT / "corpus"


# ============================================================ 正常分块 / 标题
def test_splits_by_heading_and_keeps_lines_contiguous() -> None:
    doc = "# 标题一\n\n第一段内容。\n\n## 子标题\n\n第二段内容。\n"
    chunks = split_document(doc, doc_id="d", source_path="corpus/d.md")

    assert len(chunks) == 2, "标题是硬边界：两个标题 = 两个片段"
    assert chunks[0].text == "# 标题一\n\n第一段内容。"
    assert (chunks[0].line_start, chunks[0].line_end) == (1, 3)
    assert chunks[1].text == "## 子标题\n\n第二段内容。"
    assert (chunks[1].line_start, chunks[1].line_end) == (5, 7)


def test_section_tracks_heading_hierarchy() -> None:
    doc = "# A\n\n内容A\n\n## B\n\n内容B\n\n### C\n\n内容C\n\n## D\n\n内容D\n"
    sections = [c.section for c in split_document(doc, doc_id="d", source_path="d.md")]
    assert sections == ["A", "A > B", "A > B > C", "A > D"], "回到 H2 必须丢掉 H3"


def test_content_before_first_heading_is_marked() -> None:
    doc = "开头没有标题的一段话。\n\n# 标题\n\n正文\n"
    chunks = split_document(doc, doc_id="d", source_path="d.md")
    assert chunks[0].section == NO_SECTION
    assert chunks[1].section == "标题"


def test_empty_heading_is_merged_into_the_next_section() -> None:
    """标题紧跟标题时不单独成片 —— 否则会产出十几字符的碎片。"""
    doc = "# 顶层\n\n## 空标题\n\n### 有内容\n\n正文内容\n"
    chunks = split_document(doc, doc_id="d", source_path="d.md")

    assert len(chunks) == 1, "整篇只该有一个片段"
    assert chunks[0].text == "# 顶层\n\n## 空标题\n\n### 有内容\n\n正文内容"
    assert chunks[0].section == "顶层 > 空标题 > 有内容", "归属最深的那个标题"


def test_heading_with_empty_section_is_not_split_from_its_content() -> None:
    """标题后紧跟超长代码块时，标题不能被挤成孤片。"""
    code = "```python\n" + "\n".join(f"print({i})" for i in range(200)) + "\n```"
    doc = f"# T\n\n## 2. 大段代码\n\n{code}\n"
    chunks = split_document(
        doc, doc_id="d", source_path="d.md", config=ChunkConfig(max_chars=300, overlap_chars=40)
    )

    assert len(chunks) > 1
    assert "## 2. 大段代码" in chunks[0].text, "标题必须跟着它的内容走"
    assert chunks[0].text.count("```") == 1, "首个片段应当已经进入代码块"
    assert all(len(c.text) > 40 for c in chunks), "不该出现标题孤片"


def test_trailing_empty_heading_is_kept_as_a_chunk() -> None:
    """文件末尾的空标题没有下一节可并入，只能单独成片（内容不能丢）。"""
    doc = "# T\n\n正文\n\n## 尾标题\n"
    chunks = split_document(doc, doc_id="d", source_path="d.md")
    assert chunks[-1].text == "## 尾标题"
    assert chunks[-1].section == "T > 尾标题"


def test_blank_document_yields_nothing() -> None:
    assert split_document("", doc_id="d", source_path="d.md") == []
    assert split_document("\n\n   \n\t\n", doc_id="d", source_path="d.md") == []


# ============================================================ 超长段落
def test_overlong_single_line_is_char_split_and_losslessly_restorable() -> None:
    line = "超" * 3000
    doc = f"# T\n\n{line}\n"
    config = ChunkConfig(max_chars=100, overlap_chars=20)
    chunks = split_document(doc, doc_id="d", source_path="d.md", config=config)

    body = [c for c in chunks if c.text.strip("超") == ""]  # 纯片段（不含标题）
    assert len(body) > 1, "超长单行必须被切开"
    assert all(len(c.text) <= 100 for c in body), "每一片都不能超过上限"
    assert all(c.line_start == c.line_end == 3 for c in body), "字符切片仍指向原行"

    # 无损性：把所有片段里的连续文本按顺序取出来拼接，必须还原原文（不丢不重）
    # 注意首个片段里还带着标题，所以不能直接拼 chunk.text
    restored = "".join(re.findall(r"超+", "\n".join(c.text for c in chunks)))
    assert restored == line


def test_overlong_paragraph_is_split_by_lines_without_losing_content() -> None:
    body = "\n".join(f"- 第{i}行" + "x" * 60 for i in range(1, 41))
    doc = f"# T\n\n{body}\n"
    config = ChunkConfig(max_chars=300, overlap_chars=0)
    chunks = split_document(doc, doc_id="d", source_path="d.md", config=config)

    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)
    merged = "\n".join(c.text for c in chunks)
    for i in range(1, 41):
        assert merged.count(f"- 第{i}行") == 1, "内容既不能丢，也不能重复（overlap=0）"


# ============================================================ 重叠
def _multi_line_doc() -> str:
    body = "\n".join(f"- 第{i}行" + "y" * 40 for i in range(1, 31))
    return f"# T\n\n{body}\n"


def test_overlap_repeats_tail_of_previous_chunk() -> None:
    config = ChunkConfig(max_chars=300, overlap_chars=100)
    chunks = split_document(_multi_line_doc(), doc_id="d", source_path="d.md", config=config)

    assert len(chunks) > 1
    tail = "\n".join(chunks[0].text.splitlines()[-2:])
    assert chunks[0].text.endswith(tail)
    assert chunks[1].text.startswith(tail), "相邻片段之间必须有重叠"
    # 重叠意味着行区间可以交叠，但不能完全重复
    assert chunks[1].line_start <= chunks[0].line_end
    assert chunks[1].line_end > chunks[0].line_end


def test_max_chars_is_a_hard_limit_even_with_large_overlap() -> None:
    """重叠是从预留额度里挤出来的，不允许把片段顶破上限（踩过一次）。"""
    config = ChunkConfig(max_chars=400, overlap_chars=300)
    chunks = split_corpus(CORPUS_DIR, config=config)

    assert chunks
    assert max(len(c.text) for c in chunks) <= 400, "片段长度必须 ≤ max_chars"
    assert any(
        later.line_start <= earlier.line_end for earlier, later in zip(chunks, chunks[1:])
    ), "重叠应当真的生效，而不是被静默关掉"


def test_zero_overlap_produces_disjoint_ranges() -> None:
    config = ChunkConfig(max_chars=300, overlap_chars=0)
    chunks = split_document(_multi_line_doc(), doc_id="d", source_path="d.md", config=config)
    for earlier, later in zip(chunks, chunks[1:]):
        assert later.line_start > earlier.line_end, "关闭重叠后行区间不得相交"


def test_no_overlap_across_headings() -> None:
    """跨标题不重叠：把上一节的尾巴带进新标题，section 标签就错了。"""
    first = "\n".join(f"- 甲{i}" + "z" * 40 for i in range(1, 12))
    second = "\n".join(f"- 乙{i}" + "w" * 40 for i in range(1, 12))
    doc = f"# 甲节\n\n{first}\n\n# 乙节\n\n{second}\n"
    chunks = split_document(
        doc, doc_id="d", source_path="d.md", config=ChunkConfig(max_chars=300, overlap_chars=100)
    )

    starts = [c for c in chunks if c.text.startswith("# 乙节")]
    assert len(starts) == 1
    assert starts[0].section == "乙节", "新节的第一个片段必须归属新节"
    assert "z" * 40 not in starts[0].text, "新节片段里不该出现上一节的内容（重叠）"


# ============================================================ 表格 / 代码块
def test_table_rows_and_code_fence_are_not_torn() -> None:
    rows = ["| a | b |", "| --- | --- |", *[f"| {i} | {i * 2} |" for i in range(40)]]
    code = "```python\n" + "\n".join(f"print({i})" for i in range(20)) + "\n```"
    doc = f"# T\n\n" + "\n".join(rows) + "\n\n" + code + "\n"
    # 小上限强制切开：表格按行切、代码块按行切，但都不允许出现"半行"
    chunks = split_document(
        doc, doc_id="d", source_path="d.md", config=ChunkConfig(max_chars=200, overlap_chars=0)
    )

    merged = "\n".join(c.text for c in chunks)
    # 数据行不得丢失或重复（表头行会被表格续片重复，属预期行为，见下一条用例）
    for row in rows[2:]:
        assert merged.count(row) == 1, f"数据行不应丢失或重复: {row}"
    for chunk in chunks:
        if "| a | b |" in chunk.text:
            # 表头只允许出现在片段开头，或紧跟在本节标题之后
            prefix = chunk.text.split("| a | b |")[0]
            assert not prefix.strip() or prefix.lstrip().startswith("#")
    assert merged.count("```python") == 1 and merged.count("```") == 2, "代码围栏必须成对且完整"
    for chunk in chunks:
        for line in chunk.text.splitlines():
            assert line == line.strip() or line.strip().startswith("|"), "不允许出现被截断的行"


def test_table_continuation_repeats_the_header() -> None:
    """表格被按行切开时，续片必须补回表头 —— 否则"有身体没脑袋"。"""
    rows = ["| 客户 | 金额 |", "| --- | --- |", *[f"| C{i:03d} | {i * 7} |" for i in range(1, 60)]]
    doc = "# T\n\n" + "\n".join(rows) + "\n"
    config = ChunkConfig(max_chars=200, overlap_chars=40)
    chunks = split_document(doc, doc_id="d", source_path="d.md", config=config)

    header = "| 客户 | 金额 |\n| --- | --- |"
    table_chunks = [c for c in chunks if c.text.startswith("|")]
    assert len(table_chunks) > 1, "表格必须被切开，才谈得上续片"

    for chunk in table_chunks:
        assert chunk.text.startswith(header), f"{chunk.chunk_id} 是缺表头的表格片段"
        assert all(len(c.text) <= 200 for c in chunks)

    # 重复的表头被记录在字段里（可机器校验，而不是"看起来像"）
    assert sum(1 for c in chunks if c.repeated_header == header) >= 1
    assert all(c.repeated_header == "" for c in chunks if c.text.startswith("# T"))

    # 数据行既不能丢，也不能被重叠复制（表头重复不算数据行）
    merged = "\n".join(c.text for c in chunks)
    for row in rows[2:]:
        assert merged.count(row) == 1, f"数据行被丢失或复制: {row}"


def test_carry_starting_mid_table_brings_the_header() -> None:
    """重叠恰好从表格中段开始时，也要把表头带过来（另一条路径）。"""
    rows = ["| 甲 | 乙 |", "| --- | --- |"] + [
        f"| 行{i} | {i} |" + "x" * 28 for i in range(1, 6)
    ]
    doc = "# T\n\n" + "\n".join(rows) + "\n\n" + "表后说明：" + "补" * 160 + "\n"
    chunks = split_document(
        doc, doc_id="d", source_path="d.md", config=ChunkConfig(max_chars=400, overlap_chars=150)
    )

    carried = [c for c in chunks if c.repeated_header]
    assert carried, "这条用例要覆盖的是「重叠跨到表格中段」那条路径"
    assert carried[0].text.startswith("| 甲 | 乙 |\n| --- | --- |")
    assert carried[0].line_start > 3, "重叠取的是表格中段，而不是整张表"


def test_table_without_separator_row_still_repeats_first_line() -> None:
    rows = ["| 甲 | 乙 |", *[f"| {i} | {i} |" for i in range(40)]]
    doc = "# T\n\n" + "\n".join(rows) + "\n"
    chunks = split_document(
        doc, doc_id="d", source_path="d.md", config=ChunkConfig(max_chars=160, overlap_chars=20)
    )
    continuation = [c for c in chunks if c.repeated_header]
    assert continuation, "没有分隔行时也要把首行当作表头重复"
    assert all(c.text.startswith("| 甲 | 乙 |") for c in continuation)


# ============================================================ 元数据 / 序列化
def test_metadata_fields_are_complete_and_consistent() -> None:
    doc = "# 一\n\n内容一\n\n# 二\n\n内容二\n"
    chunks = split_document(doc, doc_id="指标定义", source_path="corpus/指标定义.md")

    assert len({c.chunk_id for c in chunks}) == len(chunks)
    for index, chunk in enumerate(chunks, start=1):
        assert chunk.chunk_id == f"指标定义#{index:04d}"
        assert chunk.doc_id == "指标定义"
        assert chunk.source_path == "corpus/指标定义.md"
        assert chunk.section
        assert chunk.line_start <= chunk.line_end
        assert chunk.content_hash == hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()[:16]


def test_jsonl_roundtrip_keeps_chinese_and_fields(tmp_path: Path) -> None:
    doc = "# 标题\n\n含中文、全角标点，以及 ✅ 符号。\n"
    chunks = split_document(doc, doc_id="d", source_path="corpus/d.md")
    out = write_jsonl(chunks, tmp_path / "chunks.jsonl")

    raw = out.read_text(encoding="utf-8")
    assert "✅" in raw and "\\u" not in raw, "中文/符号不得被转义"
    assert raw.endswith("\n"), "每行一个 JSON 对象，末尾保留换行"

    objects = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assert len(objects) == len(chunks)
    assert set(objects[0]) == {
        "chunk_id",
        "doc_id",
        "source_path",
        "section",
        "text",
        "line_start",
        "line_end",
        "content_hash",
        "repeated_header",
    }
    assert objects[0]["text"] == chunks[0].text


# ============================================================ 配置
@pytest.mark.parametrize(
    "overrides",
    [
        {"max_chars": 10},
        {"max_chars": 200, "overlap_chars": -1},
        {"max_chars": 200, "overlap_chars": 200},
    ],
)
def test_invalid_config_is_rejected(overrides: dict) -> None:
    with pytest.raises(ValueError):
        ChunkConfig(**overrides)


def test_default_paths_point_into_the_lab() -> None:
    """命令行默认读写 rag_lab 内部，不受当前工作目录影响。"""
    assert DEFAULT_CORPUS_DIR == LAB_ROOT / "corpus"
    assert DEFAULT_OUT_FILE == LAB_ROOT / "chunks.jsonl"


# ============================================================ 语料与来源追溯
def test_corpus_contains_expected_documents() -> None:
    names = {path.name for path in iter_corpus_files(CORPUS_DIR)}
    assert names == {
        "README.md",
        "架构说明.md",
        "指标定义.md",
        "测试指南.md",
        "DA-017-配额与并发保护.md",
        "DA-015-敏感字段打码.md",
    }
    assert not any("codebuddy" in name for name in names), "排除 IDE 计划缓存"


def test_copied_documents_are_byte_identical_to_their_sources() -> None:
    """来源核对：复制必须无损（逐字节比对）。"""
    pairs = {
        "测试指南.md": REPO_ROOT / "data" / "samples" / "测试指南.md",
        "DA-017-配额与并发保护.md": REPO_ROOT / "docs" / "DA-017-配额与并发保护.md",
        "DA-015-敏感字段打码.md": REPO_ROOT / "docs" / "DA-015-敏感字段打码.md",
    }
    for name, source in pairs.items():
        assert source.exists(), f"源文件不存在: {source}"
        assert (CORPUS_DIR / name).read_bytes() == source.read_bytes(), f"复制不一致: {name}"


def test_new_documents_are_substantive() -> None:
    for name in ("README.md", "架构说明.md", "指标定义.md"):
        text = (CORPUS_DIR / name).read_text(encoding="utf-8")
        assert len(text) > 1500, f"{name} 内容过少，疑似占位"
        assert "待确认" in text or name == "README.md", "无法确认的内容必须标注待确认"


def test_every_chunk_is_traceable_to_its_source_lines() -> None:
    """最重要的一条：片段文本必须能在源文件里按行号定位回来。"""
    chunks = split_corpus(CORPUS_DIR, config=ChunkConfig(max_chars=300, overlap_chars=40))
    assert chunks, "语料非空，必须切出片段"

    for chunk in chunks:
        source_lines = (LAB_ROOT / chunk.source_path).read_text(encoding="utf-8").splitlines()
        span_text = "\n".join(source_lines[chunk.line_start - 1 : chunk.line_end])

        char_split = (
            chunk.line_start == chunk.line_end
            and len(chunk.text) < len(source_lines[chunk.line_start - 1])
        )
        if char_split:
            assert chunk.text in span_text, f"{chunk.chunk_id} 无法在源文件行区间中定位"
            assert chunk.repeated_header == "", "字符硬切的片段不该带重复表头"
            continue

        # 一般情况：精确的连续行切片；表格续片：重复表头 + 主区间切片
        expected = (
            f"{chunk.repeated_header}\n{span_text}" if chunk.repeated_header else span_text
        )
        assert chunk.text == expected, f"{chunk.chunk_id} 与源文件行区间对不上"

        # 重复表头必须是本文件里真实存在的表头行（不是拼出来的）
        if chunk.repeated_header:
            assert chunk.repeated_header in "\n".join(source_lines)


def test_report_mentions_counts_and_anomalies() -> None:
    chunks = split_corpus(CORPUS_DIR, config=ChunkConfig(max_chars=300, overlap_chars=40))
    report = build_report(chunks, config=ChunkConfig(max_chars=300, overlap_chars=40))
    assert "片段总数" in report and "长度:" in report and "可疑切分" in report
    for doc_id in {chunk.doc_id for chunk in chunks}:
        assert doc_id in report


# ============================================================ CLI
def test_cli_writes_jsonl_and_reports(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("# A\n\n内容 A\n", encoding="utf-8")
    (corpus / "b.md").write_text("# B\n\n内容 B\n", encoding="utf-8")
    out = tmp_path / "chunks.jsonl"

    code = main(["--corpus", str(corpus), "--out", str(out), "--max-chars", "200", "--overlap", "20"])

    assert code == 0
    assert [json.loads(line)["doc_id"] for line in out.read_text("utf-8").splitlines()] == ["a", "b"]
    captured = capsys.readouterr().out
    assert "片段总数: 2" in captured and str(out) in captured


def test_cli_returns_nonzero_on_empty_corpus(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["--corpus", str(empty), "--out", str(tmp_path / "c.jsonl")]) == 1
