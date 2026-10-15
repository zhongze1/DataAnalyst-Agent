"""RAG-04 评估测试（**全离线**）。

一个技巧：测试里的"提问"直接取片段原文，这样 FakeEmbedder 的哈希向量也能给出
**确定**的排名（相同文本 → 余弦 1.0 → 必排第一），三种方式的结果都可精确断言。
真实评估当然不能用这种提问方式 —— 那由 `eval/questions.jsonl` 负责。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_lab.chunker import Chunk, write_jsonl
from rag_lab.embedding import EmbeddingCache, EmbeddingConfig, build_embedder, embed_chunks
from rag_lab.evaluate import (
    DEFAULT_QUESTIONS,
    Question,
    Verdict,
    evaluate,
    judge,
    load_questions,
    main,
    render_report,
    score_mode,
)
from rag_lab.retrieve import Hit, Retriever

METRIC_TEXT = "客单价的分母是下单客户数，另一种口径是笔单价。"
ARCH_TEXT = "SSE 事件包括 start / tool_args / step / token / done / error。"
METRIC_SECTION = "3. 客单价"
ARCH_SECTION = "6. 流式响应"


def _chunk(text: str, *, doc_id: str, section: str, index: int) -> Chunk:
    import hashlib

    return Chunk(
        chunk_id=f"{doc_id}#{index:04d}",
        doc_id=doc_id,
        source_path=f"corpus/{doc_id}.md",
        section=section,
        text=text,
        line_start=1,
        line_end=1,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
    )


def _build_retriever(tmp_path: Path) -> Retriever:
    chunks = [
        _chunk(METRIC_TEXT, doc_id="指标定义", section=METRIC_SECTION, index=1),
        _chunk(ARCH_TEXT, doc_id="架构说明", section=ARCH_SECTION, index=2),
    ]
    config = EmbeddingConfig(provider="fake", cache_path=tmp_path / "emb.jsonl")
    cache = EmbeddingCache(config.cache_path)
    embed_chunks(chunks, build_embedder(config), cache, space=config.space, batch_size=10)
    return Retriever(chunks, config, cache=cache)


def _question(**overrides: object) -> Question:
    base: dict = {
        "qid": "Q99",
        "question": METRIC_TEXT,
        "expect_doc": "指标定义",
        "also_ok_docs": [],
        "expect_section": METRIC_SECTION,
        "answer_contains": "笔单价",
        "type": "口径",
        "scored": True,
        "note": "测试用",
    }
    base.update(overrides)
    return Question(**base)  # type: ignore[arg-type]


def _hit(chunk: Chunk, rank: int) -> Hit:
    return Hit(chunk=chunk, score=1.0, rank=rank, why="测试")


# ============================================================ 题目文件
def test_load_questions_parses_the_drafted_file() -> None:
    questions = load_questions()

    assert len(questions) == 20
    assert sum(1 for q in questions if q.scored) == 18
    assert sum(1 for q in questions if not q.scored) == 2
    assert len({q.qid for q in questions}) == len(questions)
    assert all(q.question and q.type for q in questions)
    scored = [q for q in questions if q.scored]
    assert all(q.answer_contains and q.expect_doc for q in scored)


def test_load_questions_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_questions(tmp_path / "nope.jsonl")


# ============================================================ 判定规则
def test_judge_requires_both_document_and_answer_string() -> None:
    chunk_ok = _chunk(METRIC_TEXT, doc_id="指标定义", section=METRIC_SECTION, index=1)
    chunk_same_doc_but_unrelated = _chunk("别的段落", doc_id="指标定义", section="1. 销售额", index=2)
    chunk_other_doc_with_answer = _chunk(METRIC_TEXT, doc_id="测试指南", section="x", index=3)
    question = _question()

    assert judge([_hit(chunk_ok, 2)], question) == Verdict(True, 2, "指标定义 / 3. 客单价")
    miss_wrong_content = judge([_hit(chunk_same_doc_but_unrelated, 1)], question)
    assert not miss_wrong_content.hit and "top1 落在" in miss_wrong_content.detail
    miss_wrong_doc = judge([_hit(chunk_other_doc_with_answer, 1)], question)
    assert not miss_wrong_doc.hit
    assert not judge([], question).hit


def test_judge_accepts_also_ok_docs() -> None:
    chunk = _chunk(METRIC_TEXT, doc_id="测试指南", section="x", index=1)
    question = _question(also_ok_docs=["测试指南"])
    assert judge([_hit(chunk, 1)], question).hit


def test_judge_treats_trap_questions_separately() -> None:
    question = _question(answer_contains=None, scored=False, qid="T01")
    verdict = judge([_hit(_chunk(METRIC_TEXT, doc_id="指标定义", section="s", index=1), 1)], question)
    assert not verdict.hit and verdict.detail == "陷阱题不计分"


# ============================================================ 指标
def test_score_mode_math() -> None:
    verdicts = {
        "Q1": Verdict(True, 1, ""),
        "Q2": Verdict(True, 2, ""),
        "Q3": Verdict(False, None, ""),
        "Q4": Verdict(False, None, ""),
    }
    score = score_mode("vector", verdicts)

    assert score.total == 4 and score.hits == 2
    assert score.recall == pytest.approx(0.5)
    assert score.mrr == pytest.approx((1.0 + 0.5) / 4), "未命中的题按 0 计入分母"


def test_score_mode_handles_empty_input() -> None:
    score = score_mode("keyword", {})
    assert (score.total, score.hits, score.recall, score.mrr) == (0, 0, 0.0, 0.0)


# ============================================================ 端到端
def test_evaluate_scores_all_three_modes_and_separates_traps(tmp_path: Path) -> None:
    retriever = _build_retriever(tmp_path)
    questions = [
        _question(qid="Q01"),
        _question(qid="Q02", question="语料里根本没有的内容xyz", answer_contains="不存在的串"),
        _question(
            qid="T01",
            expect_doc=None,
            answer_contains=None,
            scored=False,
            type="陷阱题",
        ),
    ]

    evaluation = evaluate(retriever, questions, k=3)

    assert len(evaluation.scores) == 3
    for score in evaluation.scores:
        assert score.total == 2, "陷阱题不进分母"
        assert score.hits == 1, "只有 Q01 能命中"
        assert score.recall == pytest.approx(0.5)
        assert score.mrr == pytest.approx(0.5), "命中在第 1 名 → 1/1 / 2 题"
    assert len(evaluation.per_question) == 2
    assert len(evaluation.traps) == 1


def test_evaluate_limit_restricts_question_count(tmp_path: Path) -> None:
    evaluation = evaluate(
        _build_retriever(tmp_path), [_question(qid="Q01"), _question(qid="Q02")], k=2, limit=1
    )
    assert len(evaluation.per_question) == 1


# ============================================================ 报告
def test_render_report_contains_sections_and_diagnostics(tmp_path: Path) -> None:
    questions = [
        _question(qid="Q01"),
        _question(qid="Q02", question="根本问不到的东西xyz", answer_contains="不存在的串"),
        _question(qid="T01", answer_contains=None, scored=False, type="陷阱题"),
    ]
    report = render_report(evaluate(_build_retriever(tmp_path), questions, k=3))

    assert "# RAG-04 检索评估报告" in report
    for section in ("## 一、三路对比", "## 二、逐题明细", "## 三、为什么召回",
                    "## 四、没被全部命中的题", "## 五、陷阱题"):
        assert section in report
    assert "| vector |" in report and "| keyword |" in report and "| hybrid |" in report
    assert "Q02" in report and "未命中" in report, "漏召回必须出现在诊断区"
    assert "T01" in report and "陷阱题" in report
    assert "为什么召回" in report and "RRF" in report


def test_report_marks_all_hit_case(tmp_path: Path) -> None:
    report = render_report(evaluate(_build_retriever(tmp_path), [_question()], k=2))
    assert "（无：三种方式都命中）" in report


# ============================================================ CLI
def _write_questions(path: Path) -> Path:
    rows = [
        {
            "qid": "Q01",
            "question": METRIC_TEXT,
            "expect_doc": "指标定义",
            "also_ok_docs": [],
            "expect_section": METRIC_SECTION,
            "answer_contains": "笔单价",
            "type": "口径",
            "scored": True,
            "note": "测试用",
        },
        {
            "qid": "T01",
            "question": "无解的问题",
            "expect_doc": None,
            "also_ok_docs": [],
            "expect_section": None,
            "answer_contains": None,
            "type": "陷阱题",
            "scored": False,
            "note": "测试用",
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    return path


def test_cli_writes_the_report(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    retriever = _build_retriever(tmp_path)
    chunks_file = write_jsonl(retriever.chunks, tmp_path / "chunks.jsonl")
    out = tmp_path / "report.md"

    code = main(
        [
            "--questions",
            str(_write_questions(tmp_path / "q.jsonl")),
            "--chunks",
            str(chunks_file),
            "--cache",
            str(tmp_path / "emb.jsonl"),
            "--provider",
            "fake",
            "--k",
            "2",
            "--out",
            str(out),
        ]
    )

    assert code == 0
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "Q01" in text and "陷阱题" in text
    printed = capsys.readouterr().out
    assert "recall@2" in printed and "报告已写入" in printed


def test_default_questions_file_exists() -> None:
    assert DEFAULT_QUESTIONS.exists()
