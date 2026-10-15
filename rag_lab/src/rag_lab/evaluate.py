"""RAG-04：在 ``eval/questions.jsonl`` 上评估三种检索方式。

为什么三种要分开算：**先证明检索器能找对内容，再谈 Agent 会不会用**。
把三路揉成一个数字，出了问题也不知道该修哪一路。

指标：

- **recall@k**：计分题里有多少题在 top-k 内命中；
- **MRR**：命中题的 ``1/首次命中排名`` 均值 —— 衡量"命中有多靠前"；
- 命中判定见 ``eval/README.md``：top-k 内存在一条，**来自可接受文档**且**文本含标准答案串**；
- **陷阱题**（``scored=false``）不计入指标，只记录最高分：
  语料里没有答案时，检索器越自信反而越危险。

报告里的每道题都会给出 top1 的 ``why`` —— 这是本阶段的硬要求：
"能解释为什么这条被召回"比"分数是多少"更重要。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

from rag_lab.chunker import LAB_ROOT
from rag_lab.embedding import (
    DEFAULT_CACHE_FILE,
    DEFAULT_CHUNKS_FILE,
    EmbeddingCache,
    EmbeddingError,
    config_from_env,
    load_chunks,
)
from rag_lab.retrieve import Hit, Retriever

MODES = ("vector", "keyword", "hybrid")

DEFAULT_QUESTIONS = LAB_ROOT / "eval" / "questions.jsonl"
DEFAULT_REPORT = LAB_ROOT / "eval" / "report.md"


@dataclass(frozen=True)
class Question:
    """一道评估题及其标准答案标注。"""

    qid: str
    question: str
    expect_doc: str | None
    also_ok_docs: list[str] = field(default_factory=list)
    expect_section: str | None = None
    answer_contains: str | None = None
    type: str = ""
    scored: bool = True
    note: str = ""

    @property
    def acceptable_docs(self) -> set[str]:
        return {self.expect_doc, *self.also_ok_docs} - {None}


@dataclass
class Verdict:
    """一道题在某种检索方式下的判定结果。"""

    hit: bool
    rank: int | None
    detail: str

    def cell(self) -> str:
        if not self.hit:
            return "✗"
        return f"✅ #{self.rank}"


def load_questions(path: Path | None = None) -> list[Question]:
    source = Path(path or DEFAULT_QUESTIONS)
    if not source.exists():
        raise FileNotFoundError(f"找不到题目文件: {source}")
    questions: list[Question] = []
    for raw in source.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        item = json.loads(line)
        questions.append(
            Question(
                qid=item["qid"],
                question=item["question"],
                expect_doc=item.get("expect_doc"),
                also_ok_docs=list(item.get("also_ok_docs") or []),
                expect_section=item.get("expect_section"),
                answer_contains=item.get("answer_contains"),
                type=item.get("type", ""),
                scored=bool(item.get("scored", True)),
                note=item.get("note", ""),
            )
        )
    return questions


def judge(hits: list[Hit], question: Question) -> Verdict:
    """判定一组召回是否"真正回答了这道题"。

    条件要同时满足：片段来自可接受文档 **且** 文本里含标准答案串。
    只看排名不看内容，会把"召回了同一文档但无关的片段"误判成命中。
    """
    if question.answer_contains is None:
        return Verdict(hit=False, rank=None, detail="陷阱题不计分")
    for hit in hits:
        if (
            hit.chunk.doc_id in question.acceptable_docs
            and question.answer_contains in hit.chunk.text
        ):
            return Verdict(
                hit=True,
                rank=hit.rank,
                detail=f"{hit.chunk.doc_id} / {hit.section}",
            )
    if hits:
        top = hits[0]
        return Verdict(
            hit=False,
            rank=None,
            detail=f"未命中；top1 落在 {top.chunk.doc_id} / {top.chunk.section}",
        )
    return Verdict(hit=False, rank=None, detail="无召回结果")


@dataclass
class ModeScore:
    mode: str
    recall: float
    mrr: float
    hits: int
    total: int


def score_mode(mode: str, verdicts: dict[str, Verdict]) -> ModeScore:
    """算一种方式的 recall@k 与 MRR（未命中的题按 0 计入分母）。"""
    total = len(verdicts)
    hits = sum(1 for verdict in verdicts.values() if verdict.hit)
    mrr = (
        sum(
            1.0 / verdict.rank
            for verdict in verdicts.values()
            if verdict.hit and verdict.rank
        )
        / total
        if total
        else 0.0
    )
    return ModeScore(
        mode=mode,
        recall=hits / total if total else 0.0,
        mrr=mrr,
        hits=hits,
        total=total,
    )


@dataclass
class Evaluation:
    """一次完整评估的原始结果，供渲染报告使用。"""

    k: int
    space: str
    chunks: int
    scores: list[ModeScore]
    per_question: list[tuple[Question, dict[str, Verdict], dict[str, str]]]
    traps: list[tuple[Question, dict[str, Verdict], dict[str, str]]]
    generated_at: str = ""


def evaluate(
    retriever: Retriever,
    questions: list[Question],
    *,
    k: int = 5,
    limit: int | None = None,
) -> Evaluation:
    """跑完所有题目。``limit`` 用于小规模试跑（省一次查询向量化的钱）。"""
    selected = questions[:limit] if limit else questions
    scores: list[ModeScore] = []
    per_question: list[tuple[Question, dict[str, Verdict], dict[str, str]]] = []
    traps: list[tuple[Question, dict[str, Verdict], dict[str, str]]] = []

    for question in selected:
        result = retriever.search(question.question, k=k, mode="all")
        verdicts = {mode: judge(result.mode(mode), question) for mode in MODES}
        why = {mode: (result.mode(mode)[0].why if result.mode(mode) else "无结果") for mode in MODES}
        if question.scored:
            per_question.append((question, verdicts, why))
        else:
            traps.append((question, verdicts, why))

    for mode in MODES:
        scores.append(score_mode(mode, {q.qid: verdicts[mode] for q, verdicts, _ in per_question}))

    return Evaluation(
        k=k,
        space=retriever.space,
        chunks=len(retriever.chunks),
        scores=scores,
        per_question=per_question,
        traps=traps,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


# --------------------------------------------------------------------- 报告
def _truncate(text: str, limit: int = 34) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_report(evaluation: Evaluation) -> str:
    lines = [
        "# RAG-04 检索评估报告",
        "",
        f"> 生成时间（本地）：{evaluation.generated_at}　|　k = {evaluation.k}　|　"
        f"片段数 = {evaluation.chunks}　|　向量空间 = `{evaluation.space}`",
        "> 命中判定：top-k 内存在一条，来自可接受文档 **且** 文本含标准答案串"
        "（标准答案与可接受文档见 `questions.jsonl`）。",
        "",
        "## 一、三路对比",
        "",
        "| 方式 | recall@{k} | MRR | 命中/计分题 |".format(k=evaluation.k),
        "| --- | --- | --- | --- |",
    ]
    for score in evaluation.scores:
        lines.append(
            f"| {score.mode} | **{score.recall:.3f}** | {score.mrr:.3f} | "
            f"{score.hits}/{score.total} |"
        )

    lines += [
        "",
        "> 三种方式各自独立计算，没有做任何融合调参 —— 先看清哪一路强、哪一路弱。",
        "",
        "## 二、逐题明细",
        "",
        "| 题号 | 类型 | 问题 | 期望文档 | 向量 | 关键词 | 混合 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for question, verdicts, _ in evaluation.per_question:
        expect = question.expect_doc or "-"
        if question.also_ok_docs:
            expect += f"（+{'/'.join(question.also_ok_docs)}）"
        cells = " | ".join(verdicts[mode].cell() for mode in MODES)
        lines.append(
            f"| {question.qid} | {question.type} | {_truncate(question.question)} | "
            f"{_truncate(expect, 26)} | {cells} |"
        )

    lines += [
        "",
        "## 三、为什么召回（每题 top1 的理由）",
        "",
        "| 题号 | 向量 | 关键词 | 混合 |",
        "| --- | --- | --- | --- |",
    ]
    for question, _, why in evaluation.per_question:
        lines.append(
            f"| {question.qid} | {_truncate(why['vector'], 40)} | "
            f"{_truncate(why['keyword'], 44)} | {_truncate(why['hybrid'], 48)} |"
        )

    missed = [
        (question, verdicts, why)
        for question, verdicts, why in evaluation.per_question
        if not all(verdicts[mode].hit for mode in MODES)
    ]
    lines += ["", "## 四、没被全部命中的题（诊断入口）", ""]
    if not missed:
        lines.append("（无：三种方式都命中）")
    for question, verdicts, why in missed:
        lines.append(f"### {question.qid}　{question.question}")
        lines.append("")
        lines.append(f"- 期望：`{question.expect_doc}`"
                     f"（章节 {question.expect_section or '-'}）｜标准串 `{question.answer_contains}`")
        lines.append(f"- 备注：{question.note}")
        for mode in MODES:
            verdict = verdicts[mode]
            mark = "✅ 命中" if verdict.hit else "✗ 未命中"
            lines.append(f"- **{mode}**：{mark} —— {verdict.detail}")
            if not verdict.hit:
                lines.append(f"  - top1 为什么召回：{why[mode]}")
        lines.append("")

    lines += ["## 五、陷阱题（不计分，只看自信度）", ""]
    if not evaluation.traps:
        lines.append("（无）")
    for question, _, why in evaluation.traps:
        lines.append(f"### {question.qid}　{question.question}")
        lines.append("")
        lines.append(f"- 语料里的正确立场：{question.note}")
        for mode in MODES:
            lines.append(f"- **{mode}** top1：{why[mode]}")
        lines.append("")
    lines.append(
        "> 分数越高越危险：这类问题语料里根本没有答案，"
        "高分意味着检索器在「无答案」时也很自信。"
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """命令行入口：``uv run python -m rag_lab.evaluate``。"""
    parser = argparse.ArgumentParser(
        prog="python -m rag_lab.evaluate",
        description="在 eval/questions.jsonl 上评估三种检索方式并写 eval/report.md",
    )
    parser.add_argument("--k", type=int, default=5, help="每种方式取前 K 条")
    parser.add_argument(
        "--provider",
        choices=["fake", "api"],
        help="覆盖 .env 里的 provider；fake 只能验证流程能不能跑通（它不懂语义）",
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_FILE)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_FILE)
    parser.add_argument("--limit", type=int, help="只跑前 N 题（小规模试跑，省查询向量钱）")
    args = parser.parse_args(argv)

    try:
        config = config_from_env()
        if args.provider:
            config = replace(config, provider=args.provider)
        chunks = load_chunks(args.chunks)
        retriever = Retriever(chunks, config, cache=EmbeddingCache(args.cache))
        questions = load_questions(args.questions)
        evaluation = evaluate(retriever, questions, k=args.k, limit=args.limit)
    except EmbeddingError as exc:
        print(f"[{exc.code}] {exc}", flush=True)
        return 3
    except (ValueError, FileNotFoundError) as exc:
        print(f"[EVAL_NOT_READY] {exc}", flush=True)
        return 4

    report = render_report(evaluation)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")

    print(f"片段数: {evaluation.chunks} | 计分题: {evaluation.scores[0].total if evaluation.scores else 0}"
          f" | k={evaluation.k}")
    for score in evaluation.scores:
        print(f"  {score.mode:8s} recall@{evaluation.k}={score.recall:.3f}  MRR={score.mrr:.3f}"
              f"  ({score.hits}/{score.total})")
    print(f"报告已写入: {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - 由 CLI 测试覆盖 main()
    sys.exit(main())
