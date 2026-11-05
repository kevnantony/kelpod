"""Structural chunker (P1.1.7): section -> paragraph -> sentence fallback.

* Headings never end up in chunk text; they form ``heading_path`` (a stack by level).
* A section boundary (any heading) always closes the current chunk.
* Paragraph/list blocks of one section are packed greedily up to ``target_tokens``.
* A block longer than ``max_tokens`` is split at sentence boundaries (pieces up to
  ``target_tokens``) with ``overlap_fraction * target_tokens`` of trailing sentences
  repeated at the start of the next piece — the only place where overlap is used.
  Sentences longer than the target are bisected at word (then character) boundaries.
* Tables are their own chunks (``is_table``); a table above ``max_tokens`` is split by rows,
  each piece repeating the Markdown header.
* Every emitted chunk satisfies ``n_tokens <= max_tokens`` for the given counter.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from zllf.stages.ingest.tokens import TokenCounter

_SENTENCE_END = re.compile(
    r"(?:(?<=[.!?])|(?<=[.!?][\"')\]]))\s+(?=[\"'(\[]?[A-ZÄÖÜ0-9])"
)


class BlockLike(Protocol):
    @property
    def type(self) -> str: ...
    @property
    def text(self) -> str: ...
    @property
    def page(self) -> int | None: ...
    @property
    def level(self) -> int | None: ...
    @property
    def table_markdown(self) -> str | None: ...


@dataclass
class ChunkDraft:
    text: str
    n_tokens: int
    heading_path: list[str] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    is_table: bool = False


def split_sentences(text: str) -> list[str]:
    """Rule-based sentence split (``.!?`` followed by whitespace and an upper-case/digit)."""
    parts = [p.strip() for p in _SENTENCE_END.split(text)]
    return [p for p in parts if p]


def bisect_to_limit(text: str, counter: TokenCounter, limit: int) -> list[str]:
    """Split ``text`` at word (else character) midpoints until every piece fits ``limit``."""
    if counter.count(text) <= limit or len(text) <= 1:
        return [text]
    words = text.split(" ")
    if len(words) > 1:
        mid = len(words) // 2
        left, right = " ".join(words[:mid]), " ".join(words[mid:])
    else:
        mid = len(text) // 2
        left, right = text[:mid], text[mid:]
    return bisect_to_limit(left, counter, limit) + bisect_to_limit(right, counter, limit)


def _pack(units: Sequence[str], counter: TokenCounter, target: int, max_tokens: int,
          overlap_tokens: int, sep: str = " ") -> list[str]:
    """Greedy packing of units into pieces <= target, with trailing-unit overlap."""
    pieces: list[str] = []
    cur: list[str] = []
    cur_n = 0
    for unit in units:
        n = counter.count(unit)
        if cur and cur_n + n > target:
            pieces.append(sep.join(cur))
            tail: list[str] = []
            t = 0
            for prev in reversed(cur):
                pn = counter.count(prev)
                if t + pn > overlap_tokens:
                    break
                tail.insert(0, prev)
                t += pn + 1
            if t + n > max_tokens or len(tail) == len(cur):
                tail, t = [], 0
            cur, cur_n = tail, t
        cur.append(unit)
        cur_n += n + 1
    if cur:
        pieces.append(sep.join(cur))
    return pieces


def split_long_text(text: str, counter: TokenCounter, target: int, max_tokens: int,
                    overlap_fraction: float) -> list[str]:
    """Sentence-fallback split of one oversized block."""
    units: list[str] = []
    for sentence in split_sentences(" ".join(text.split("\n"))):
        if counter.count(sentence) > target:
            units.extend(bisect_to_limit(sentence, counter, target))
        else:
            units.append(sentence)
    overlap = round(overlap_fraction * target)
    return _pack(units, counter, target, max_tokens, overlap)


def _table_pieces(body: str, is_markdown: bool, counter: TokenCounter, target: int,
                  max_tokens: int) -> list[str]:
    if counter.count(body) <= max_tokens:
        return [body]
    lines = body.split("\n")
    header = lines[:2] if is_markdown and len(lines) >= 2 else []
    rows = lines[len(header):]
    head = "\n".join(header)
    budget = max(1, target - counter.count(head))
    units: list[str] = []
    for row in rows:
        units.extend(bisect_to_limit(row, counter, budget) if counter.count(row) > budget
                     else [row])
    out = []
    for piece in _pack(units, counter, budget, max_tokens, 0, sep="\n"):
        out.append(f"{head}\n{piece}" if head else piece)
    return out


def _pages(pages: Iterable[int | None]) -> tuple[int | None, int | None]:
    known = [p for p in pages if p is not None]
    return (min(known), max(known)) if known else (None, None)


def _enforce_max(text: str, counter: TokenCounter, max_tokens: int) -> list[str]:
    """Guard against non-additive tokenisation: re-split anything still above the max."""
    if counter.count(text) <= max_tokens:
        return [text]
    return bisect_to_limit(text, counter, max_tokens)


def chunk_blocks(
    blocks: Sequence[BlockLike],
    counter: TokenCounter,
    target_tokens: int,
    max_tokens: int,
    overlap_fraction: float = 0.10,
) -> list[ChunkDraft]:
    """Chunk one document's blocks (in reading order)."""
    target = max(1, min(target_tokens, max_tokens))
    stack: list[tuple[int, str]] = []
    out: list[ChunkDraft] = []
    buf: list[tuple[str, int | None]] = []
    buf_n = 0

    def path() -> list[str]:
        return [t for _, t in stack]

    def emit(texts: Iterable[str], pages: Sequence[int | None], is_table: bool) -> None:
        start, end = _pages(pages)
        for text in texts:
            for piece in _enforce_max(text, counter, max_tokens):
                if piece.strip():
                    out.append(ChunkDraft(piece, counter.count(piece), path(), start, end,
                                          is_table))

    def flush() -> None:
        nonlocal buf_n
        if not buf:
            return
        joined = "\n\n".join(t for t, _ in buf)
        pages = [p for _, p in buf]
        if counter.count(joined) <= max_tokens:
            emit([joined], pages, False)
        else:  # tokenizer not additive across joins: fall back to one chunk per block
            for t, p in buf:
                emit([t], [p], False)
        buf.clear()
        buf_n = 0

    for block in blocks:
        text = block.text.strip()
        if block.type == "heading":
            flush()
            level = block.level if block.level is not None else 1
            while stack and stack[-1][0] >= level:
                stack.pop()
            if text:
                stack.append((level, text))
            continue
        if block.type == "table":
            flush()
            body = (block.table_markdown or text).strip()
            if body:
                emit(_table_pieces(body, block.table_markdown is not None, counter, target,
                                   max_tokens), [block.page], True)
            continue
        if not text:
            continue
        n = counter.count(text)
        if n > max_tokens:
            flush()
            emit(split_long_text(text, counter, target, max_tokens, overlap_fraction),
                 [block.page], False)
            continue
        if buf and buf_n + n + 1 > target:
            flush()
        buf.append((text, block.page))
        buf_n += n + 1
    flush()
    return out
