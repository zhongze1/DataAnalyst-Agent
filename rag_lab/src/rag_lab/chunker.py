"""Markdown 分块器（RAG-02）。

目标：把语料切成"边界尽量自然"的片段，并保留**可追溯的来源信息**。
本阶段只做纯文本处理 —— 不调用 Embedding、不引入向量库、不读写索引。

分块策略（四步）：

1. **解析**：按行扫描，识别标题 / 代码块 / 表格 / 段落，得到带行号的"原子块"，
   同时维护标题栈，算出每个块所属的 ``section``（标题路径，如 ``3. Agent Loop > 关键点``）。
   标题、代码块、表格都是**原子单位**，不会被从中间切开。
2. **标题优先，但标题不孤立**：标题是硬边界（遇到标题一定开新片段），
   但"只有标题、紧跟另一个标题"的**空标题**不单独成片 —— 它会被并入下一节的第一片，
   否则会产出一堆十几字符的碎片（踩过：``## 关联代码/文件`` 紧跟 ``### 修改``）。
3. **装箱**：同一 section 内顺序装箱，新内容累计超过 ``max_chars − overlap_chars − 1`` 就切段，
   并在相邻片段之间按**整行**回退保留 ``overlap_chars`` 的重叠（不切碎行）。
   重叠占的是预留额度，因此**片段长度永不突破 max_chars**（见 :class:`_Budget`）。
4. **表格续片补表头**：表格超过上限时按行切开，续片会在开头**重复原始表头**，
   并把这份重复内容记在 ``repeated_header`` 字段里（不再"有身体没脑袋"）。

**可追溯性契约**（被 ``tests/test_chunker.py`` 锁定）：

- ``line_start`` / ``line_end`` 是源文件中的**行号闭区间**（指该片段的**新内容**）；
- 一般情况下 ``text == repeated_header + "\\n" + "\\n".join(源文件行[line_start-1:line_end])``
  （``repeated_header`` 为空时即精确等于连续行切片；区间内部的空行原样保留）；
- 唯一例外是"单行超长被字符硬切"：此时 ``line_start == line_end``，
  ``text`` 是该行的**子串**；把同一行的所有切片按顺序拼起来能还原原文。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

# rag_lab/ 根目录：本文件位于 rag_lab/src/rag_lab/chunker.py
LAB_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_DIR = LAB_ROOT / "corpus"
DEFAULT_OUT_FILE = LAB_ROOT / "chunks.jsonl"

DEFAULT_MAX_CHARS = 1200
DEFAULT_OVERLAP_CHARS = 150

# 片段过短阈值（只用于质量报告，不影响切分结果）
SHORT_CHUNK_CHARS = 40

NO_SECTION = "(无标题)"

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")


@dataclass(frozen=True)
class ChunkConfig:
    """分块参数。"""

    max_chars: int = DEFAULT_MAX_CHARS
    overlap_chars: int = DEFAULT_OVERLAP_CHARS

    def __post_init__(self) -> None:
        if self.max_chars < 50:
            raise ValueError(f"max_chars 过小（{self.max_chars}），至少 50")
        if self.overlap_chars < 0:
            raise ValueError(f"overlap_chars 不能为负（{self.overlap_chars}）")
        if self.overlap_chars >= self.max_chars:
            raise ValueError(
                f"overlap_chars({self.overlap_chars}) 必须小于 max_chars({self.max_chars})"
                "，否则切片无法收敛"
            )


@dataclass(frozen=True)
class _Budget:
    """切片预算 —— 让"长度上限"成为**硬约束**。

    片段 = 重叠(尾部若干行) + 换行 + 新内容，所以只要
    ``新内容 ≤ max_chars − overlap − 1``，就有：

        片段长度 ≤ overlap + 1 + (max_chars − overlap − 1) == max_chars

    否则"重叠"会把片段顶出上限（踩过：小上限下出现 445 > 400 的片段）。
    """

    body: int  # 单个片段的**新内容**上限
    overlap: int  # 实际使用的重叠上限（可被压缩，但不会突破 max_chars）

    @classmethod
    def of(cls, config: ChunkConfig) -> "_Budget":
        min_body = 50  # 新内容至少要装得下一点，否则切片会退化成一堆碎片
        overlap = max(0, min(config.overlap_chars, config.max_chars - min_body - 1))
        return cls(body=max(min_body, config.max_chars - overlap - 1), overlap=overlap)


@dataclass(frozen=True)
class Chunk:
    """一个文本片段及其来源信息。"""

    chunk_id: str
    doc_id: str
    source_path: str
    section: str
    text: str
    line_start: int
    line_end: int
    content_hash: str
    # 表格被切开时，续片开头重复的**原始表头**（为空表示没有重复内容）
    repeated_header: str = ""

    def to_dict(self) -> dict[str, object]:
        """字段顺序固定，便于 chunks.jsonl 逐行阅读与 diff。"""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "section": self.section,
            "text": self.text,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "content_hash": self.content_hash,
            "repeated_header": self.repeated_header,
        }


@dataclass(frozen=True)
class _Block:
    """解析出来的原子块（只记类型、行区间与归属 section）。"""

    kind: str  # heading | code | table | paragraph
    line_start: int  # 1-based，闭区间
    line_end: int
    section: str


@dataclass(frozen=True)
class _Span:
    """片段内的一段来源区间。

    ``char_from`` / ``char_to`` 只在"单行超长被字符硬切"时使用（此时是**行内**偏移）；
    ``repeat_header`` 只在表格续片上使用（指向该表格的表头行区间）。
    """

    line_start: int
    line_end: int
    char_from: int | None = None
    char_to: int | None = None
    repeat_header: tuple[int, int] | None = None

    @property
    def char_split(self) -> bool:
        return self.char_from is not None


def content_hash(text: str) -> str:
    """片段内容指纹，取 sha256 前 16 位（够用且短）。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------- 解析
def _parse_blocks(lines: list[str]) -> list[_Block]:
    """把源文件切成原子块（跳过纯空行：它们是分隔符，不该单独成片）。"""
    blocks: list[_Block] = []
    stack: list[str] = []  # 标题栈，索引 = 级别 - 1
    index = 0
    total = len(lines)

    def section_of() -> str:
        return " > ".join(stack) if stack else NO_SECTION

    while index < total:
        line = lines[index]

        heading = _HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            del stack[level - 1 :]
            stack.append(heading.group(2))
            blocks.append(_Block("heading", index + 1, index + 1, section_of()))
            index += 1
            continue

        if _FENCE_RE.match(line):
            end = index + 1
            while end < total and not _FENCE_RE.match(lines[end]):
                end += 1
            last = min(end, total - 1)  # 围栏未闭合时吃到文件末尾
            blocks.append(_Block("code", index + 1, last + 1, section_of()))
            index = last + 1
            continue

        if line.lstrip().startswith("|"):
            end = index
            while end < total and lines[end].lstrip().startswith("|"):
                end += 1
            blocks.append(_Block("table", index + 1, end, section_of()))
            index = end
            continue

        if not line.strip():
            index += 1
            continue

        # 段落：连续非空行，直到空行 / 标题 / 围栏 / 表格行
        end = index
        while end < total:
            candidate = lines[end]
            if (
                not candidate.strip()
                or _HEADING_RE.match(candidate)
                or _FENCE_RE.match(candidate)
                or candidate.lstrip().startswith("|")
            ):
                break
            end += 1
        blocks.append(_Block("paragraph", index + 1, end, section_of()))
        index = end

    return blocks


def _table_header_range(block: _Block, lines: list[str]) -> tuple[int, int] | None:
    """表格的表头行区间：首行；若第二行是 ``| --- |`` 分隔行，则一起算进去。"""
    if block.line_start > block.line_end:
        return None
    header_end = block.line_start
    if block.line_end >= block.line_start + 1:
        cells = [
            cell.strip()
            for cell in lines[block.line_start].strip().strip("|").split("|")
        ]
        if cells and all(_SEPARATOR_CELL_RE.match(cell or "") for cell in cells):
            header_end = block.line_start + 1
    return (block.line_start, header_end)


# --------------------------------------------------------------------- 文本组装
def _region_text(lines: list[str], start: int, end: int) -> str:
    """源文件行区间的原文（含区间内空行）。"""
    return "\n".join(lines[start - 1 : end])


def assemble(lines: list[str], spans: list[_Span]) -> str:
    """把若干区间拼成片段文本。

    - 首个区间若带 ``repeat_header``，先把表头原文放在最前面（表格续片的"补脑袋"）；
    - 非字符切分的区间**直接从源文件取连续行**（含区间之间的空行），因此是源文件的连续切片；
    - 字符切分的区间取行内子串。
    """
    parts: list[str] = []
    first = spans[0]
    if first.repeat_header is not None:
        parts.append(_region_text(lines, *first.repeat_header))

    previous_end: int | None = None
    for span in spans:
        if span.char_split:
            parts.append(lines[span.line_start - 1][span.char_from : span.char_to])
        elif previous_end is None:
            parts.append(_region_text(lines, span.line_start, span.line_end))
        else:
            # 从上一区间的**下一行**继续取，中间的空白行会被原样带上
            parts.append("\n".join(lines[previous_end : span.line_end]))
        previous_end = span.line_end
    return "\n".join(parts)


def _joined_len(lines: list[str], spans: list[_Span]) -> int:
    return len(assemble(lines, spans))


def span_of(block: _Block) -> _Span:
    return _Span(block.line_start, block.line_end)


def _split_span_by_lines(block: _Block, lines: list[str], max_chars: int) -> list[_Span]:
    """把超大原子块按**行**切开；单行仍超长时按字符切。

    表格会记录表头区间，续片（非首块）带上它 —— 这是"表格续片补表头"的实现点。
    代码块与段落按同一条路径切，因此不会出现"半行"。
    """
    header = _table_header_range(block, lines) if block.kind == "table" else None
    spans: list[_Span] = []
    start = block.line_start
    while start <= block.line_end:
        line_text = lines[start - 1]
        if len(line_text) > max_chars:
            for position in range(0, len(line_text), max_chars):
                spans.append(
                    _Span(start, start, position, min(position + max_chars, len(line_text)))
                )
            start += 1
            continue
        end = start
        while end < block.line_end and _joined_len(
            lines, [_Span(start, end + 1)]
        ) <= max_chars:
            end += 1
        spans.append(_Span(start, end))
        start = end + 1

    if header is None or len(spans) <= 1:
        return spans
    # 续片补表头；但表头行本身不能又是续片（避免极短表头被无限重复）
    return [
        span if index == 0 or span.char_split else replace(span, repeat_header=header)
        for index, span in enumerate(spans)
    ]


def _mid_table_header(lines: list[str], line_no: int) -> tuple[int, int] | None:
    """若 ``line_no`` 落在某张表格的**中间**，返回该表格的表头行区间。

    用于"重叠恰好从表格中段开始"的情况：带上表头，续片才不是"有身体没脑袋"。
    """
    if line_no < 1 or line_no > len(lines) or not lines[line_no - 1].lstrip().startswith("|"):
        return None
    if line_no == 1 or not lines[line_no - 2].lstrip().startswith("|"):
        return None  # 已经是表格首行，不需要补
    top = line_no
    while top > 1 and lines[top - 2].lstrip().startswith("|"):
        top -= 1
    if top + 1 <= len(lines) and lines[top].lstrip().startswith("|"):
        cells = [cell.strip() for cell in lines[top].strip().strip("|").split("|")]
        if cells and all(_SEPARATOR_CELL_RE.match(cell or "") for cell in cells):
            return (top, top + 1)
    return (top, top)


def _overlap_span(lines: list[str], spans: list[_Span], overlap_chars: int) -> _Span | None:
    """从片段尾部按整行回退，得到下一片段要继承的重叠区间。

    四条边界（都踩过坑）：

    - 含字符切分的片段不回退（半个行做重叠没意义）；
    - 至少两行才回退（只回退一行等于整段重复）；
    - **只在最后一个原子块内部回退**，不跨块 —— 否则会把上一块的尾巴
      （例如表格的最后一行）拖进下一片段；
    - 若回退的起点落在表格中段，**补上该表格的表头**（记在 ``repeat_header`` 里）。
    """
    if overlap_chars <= 0 or not spans or any(span.char_split for span in spans):
        return None
    end = spans[-1].line_end
    floor = spans[-1].line_start  # 不跨块
    chosen = end
    while chosen > floor:
        candidate = _Span(chosen - 1, end)
        if _joined_len(lines, [candidate]) > overlap_chars:
            break
        chosen -= 1
    if end - chosen < 1:
        return None
    return _Span(chosen, end, repeat_header=_mid_table_header(lines, chosen))


# --------------------------------------------------------------------- 分块
def split_document(
    text: str,
    *,
    doc_id: str,
    source_path: str,
    config: ChunkConfig | None = None,
) -> list[Chunk]:
    """把一份 Markdown 文本切成片段。"""
    config = config or ChunkConfig()
    lines = text.splitlines()
    if not any(line.strip() for line in lines):
        return []

    budget = _Budget.of(config)
    blocks = _parse_blocks(lines)
    chunks: list[Chunk] = []
    spans: list[_Span] = []
    sections: list[str] = []
    kinds: list[str] = []  # 与 spans 对齐：heading | content
    carried = 0  # 开头的 carried 个 span 是"从上一片段借来的重叠"，不算真实内容
    deferred: list[tuple[_Span, str, str]] = []  # 只有标题、还没等到内容的片段

    def body_all_headings() -> bool:
        return bool(spans) and all(kind == "heading" for kind in kinds[carried:])

    def deferred_len() -> int:
        return _joined_len(lines, [span for span, _, _ in deferred]) if deferred else 0

    def consume_deferred() -> None:
        """把挂起的空标题接到当前片段开头（它们属于下一节的开头）。"""
        nonlocal carried
        for span, section, kind in deferred:
            spans.append(span)
            sections.append(section)
            kinds.append(kind)
        deferred.clear()

    def body_len() -> int:
        """当前片段里**新内容**的字符数（不含开头的重叠）。"""
        if len(spans) <= carried:
            return 0
        return _joined_len(lines, spans[carried:])

    def flush(
        keep_overlap: bool = True,
        carry_room: int | None = None,
        allow_defer: bool = True,
    ) -> None:
        """产出当前片段；``keep_overlap`` 决定是否把重叠留给下一片段。

        ``carry_room`` 是"下一片段还能放下多少重叠"的剩余额度：
        传入它就等于承诺 ``重叠 + 换行 + 下一片新内容 ≤ max_chars``。
        ``allow_defer=False`` 用于收尾：此时不能再把空标题"留给下一节"，
        否则文件末尾的标题会被静默丢掉。
        """
        nonlocal spans, sections, kinds, carried
        if not spans:
            return

        if allow_defer and len(spans) > carried and body_all_headings():
            # 只有标题、没有内容：不单独成片，留给下一节（除非标题比重叠额度还长，
            # 那时为了保住硬上限只能让它单独成片）
            if deferred_len() + body_len() <= budget.overlap:
                deferred = [
                    (span, sections[index], kinds[index])
                    for index, span in enumerate(spans)
                    if index >= carried
                ]
                spans, sections, kinds, carried = [], [], [], 0
                return

        chunk_text = assemble(lines, spans)
        if chunk_text.strip():
            header_span = spans[0].repeat_header
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{len(chunks) + 1:04d}",
                    doc_id=doc_id,
                    source_path=source_path,
                    section=sections[-1] if sections else NO_SECTION,
                    text=chunk_text,
                    line_start=spans[0].line_start,
                    line_end=spans[-1].line_end,
                    content_hash=content_hash(chunk_text),
                    repeated_header=(
                        _region_text(lines, *header_span) if header_span else ""
                    ),
                )
            )

        if not keep_overlap:
            spans, sections, kinds, carried = [], [], [], 0
            return
        limit = budget.overlap
        if carry_room is not None:
            limit = min(limit, max(0, carry_room))
        carry = _overlap_span(lines, spans, limit)
        last_section = sections[-1] if sections else NO_SECTION
        spans, sections, kinds = ([carry], [last_section], ["content"]) if carry else ([], [], [])
        carried = 1 if carry else 0

    for position, block in enumerate(blocks):
        following = blocks[position + 1] if position + 1 < len(blocks) else None

        if block.kind == "heading":
            # 只有"紧跟另一个标题"的标题才是空标题；文件末尾的标题算真正的分节标题
            # （否则它会粘到上一节的内容尾巴上，把上一节的 section 标签带偏）
            empty_heading = following is not None and following.kind == "heading"
            if not empty_heading:
                # 真正的分节标题：硬边界，且**不继承**上一节的重叠
                if spans:
                    flush(keep_overlap=False)
                consume_deferred()
            deferred.append((span_of(block), block.section, "heading"))
            continue

        # ---- 普通内容块 ----
        prefix = deferred_len()
        consume_deferred()
        # 首片要为挂过来的标题留位置，否则会顶破上限
        cap = budget.body if prefix == 0 else max(50, budget.body - prefix)
        block_span = span_of(block)
        pieces = (
            _split_span_by_lines(block, lines, cap)
            if _joined_len(lines, [block_span]) > cap
            else [block_span]
        )
        for piece in pieces:
            piece_len = _joined_len(lines, [piece])
            if (
                spans
                and not body_all_headings()
                and body_len() + 1 + piece_len > budget.body
            ):
                # 表格续片自带表头，不需要再叠重叠；其余情况按剩余空间给重叠
                if piece.repeat_header is not None:
                    flush(keep_overlap=False)
                else:
                    flush(carry_room=config.max_chars - piece_len - 1)
            spans.append(piece)
            sections.append(block.section)
            kinds.append("content")

    # 文件末尾挂着的标题：单独成一片（保住内容，哪怕它很短）
    consume_deferred()
    flush(allow_defer=False)
    return chunks


# --------------------------------------------------------------------- 语料
def iter_corpus_files(corpus_dir: Path) -> list[Path]:
    """语料文件（只取 Markdown，按文件名排序保证可复现）。"""
    return sorted(path for path in corpus_dir.glob("*.md") if path.is_file())


def split_corpus(
    corpus_dir: Path | None = None,
    *,
    config: ChunkConfig | None = None,
    lab_root: Path | None = None,
) -> list[Chunk]:
    """切分整个语料目录；``source_path`` 记相对 rag_lab 的路径，便于跨机器复现。"""
    corpus_dir = Path(corpus_dir or DEFAULT_CORPUS_DIR)
    root = Path(lab_root or LAB_ROOT)
    chunks: list[Chunk] = []
    for path in iter_corpus_files(corpus_dir):
        try:
            source_path = path.relative_to(root).as_posix()
        except ValueError:  # 语料目录不在 rag_lab 内时退回文件名
            source_path = path.name
        chunks.extend(
            split_document(
                path.read_text(encoding="utf-8"),
                doc_id=path.stem,
                source_path=source_path,
                config=config,
            )
        )
    return chunks


def write_jsonl(chunks: list[Chunk], out_file: Path | None = None) -> Path:
    """写 chunks.jsonl：一行一个 JSON 对象，中文不转义。"""
    out = Path(out_file or DEFAULT_OUT_FILE)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(
        json.dumps(chunk.to_dict(), ensure_ascii=False) for chunk in chunks
    )
    out.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
    return out


# --------------------------------------------------------------------- 报告
def _percentile(values: list[int], ratio: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(ratio * (len(ordered) - 1))))
    return ordered[index]


def _looks_like_headerless_table(chunk: Chunk) -> bool:
    """片段以表格行开头、但第二行不是分隔行 —— 很可能是一张没有表头的续片。"""
    if not chunk.text.lstrip().startswith("|"):
        return False
    if chunk.repeated_header:
        return False
    rows = [line for line in chunk.text.splitlines() if line.strip()]
    if len(rows) < 2:
        return True
    cells = [cell.strip() for cell in rows[1].strip().strip("|").split("|")]
    return not all(_SEPARATOR_CELL_RE.match(cell or "") for cell in cells)


def build_report(chunks: list[Chunk], *, config: ChunkConfig | None = None) -> str:
    """分块质量报告：数量、长度分布、可疑切分。供 CLI 打印与人工审核。"""
    config = config or ChunkConfig()
    lengths = [len(chunk.text) for chunk in chunks]
    lines = [
        "== 分块质量报告 ==",
        f"片段总数: {len(chunks)}",
        f"文档数: {len({chunk.doc_id for chunk in chunks})}",
        f"参数: max_chars={config.max_chars} overlap_chars={config.overlap_chars}",
    ]

    if not lengths:
        lines.append("（没有切出任何片段）")
        return "\n".join(lines)

    lines += [
        "长度: min={} p50={} p90={} max={}".format(
            min(lengths), _percentile(lengths, 0.5), _percentile(lengths, 0.9), max(lengths)
        ),
        f"表格续片（已补表头）: {sum(1 for c in chunks if c.repeated_header)}",
        "",
        "每份文档的片段数:",
    ]
    per_doc = Counter(chunk.doc_id for chunk in chunks)
    for doc_id, count in sorted(per_doc.items()):
        doc_lengths = [len(c.text) for c in chunks if c.doc_id == doc_id]
        lines.append(
            f"  - {doc_id}: {count} 片（长度 min={min(doc_lengths)} max={max(doc_lengths)}）"
        )

    anomalies: list[str] = []
    over_limit = [c for c in chunks if len(c.text) > config.max_chars]
    if over_limit:
        anomalies.append(
            f"超过长度上限的片段 {len(over_limit)} 个"
            f"（首个: {over_limit[0].chunk_id}，{len(over_limit[0].text)} 字符）"
        )
    headerless = [c for c in chunks if _looks_like_headerless_table(c)]
    if headerless:
        anomalies.append(
            f"疑似缺表头的表格片段 {len(headerless)} 个"
            f"（例如 {', '.join(c.chunk_id for c in headerless[:3])}）"
        )
    short = [c for c in chunks if len(c.text) < SHORT_CHUNK_CHARS]
    if short:
        anomalies.append(
            f"过短片段（<{SHORT_CHUNK_CHARS} 字符）{len(short)} 个"
            f"（例如 {', '.join(c.chunk_id for c in short[:3])}）"
        )
    no_section = [c for c in chunks if c.section == NO_SECTION]
    if no_section:
        anomalies.append(f"无标题归属的片段 {len(no_section)} 个（出现在首个标题之前）")

    lines += ["", "可疑切分:" if anomalies else "可疑切分: 无"]
    lines += [f"  ! {item}" for item in anomalies]
    return "\n".join(lines)


# --------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    """命令行入口：``uv run python -m rag_lab.chunker``。"""
    parser = argparse.ArgumentParser(
        prog="python -m rag_lab.chunker",
        description="把 rag_lab/corpus 下的 Markdown 切成 chunks.jsonl（纯文本，不涉及向量）",
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_DIR, help="语料目录")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE, help="输出 JSONL 路径")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help="片段长度上限")
    parser.add_argument(
        "--overlap", type=int, default=DEFAULT_OVERLAP_CHARS, help="相邻片段重叠字符数"
    )
    args = parser.parse_args(argv)

    config = ChunkConfig(max_chars=args.max_chars, overlap_chars=args.overlap)
    chunks = split_corpus(args.corpus, config=config)
    out = write_jsonl(chunks, args.out)

    print(f"语料目录: {args.corpus}")
    print(f"输出文件: {out}")
    print(build_report(chunks, config=config))
    return 0 if chunks else 1


if __name__ == "__main__":  # pragma: no cover - 由 CLI 测试覆盖 main()
    sys.exit(main())
