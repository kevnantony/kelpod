"""Text normalisation (P1.1.6) and line-based structure recovery for plain-text sources.

All functions are pure. Order used by loaders for paginated text:
``strip_repeated_edges(pages)`` -> per page ``normalize_text`` -> ``blocks_from_text``.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence

from zllf.stages.ingest.model import ParsedBlock

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
# pdfium marks a hyphen at a line break as U+FFFE (or \x02): join the word parts
_PDFIUM_HYPHEN = re.compile(r"[\ufffe\x02][ \t]*\n?[ \t]*")
_HYPHEN_BREAK = re.compile(r"(\w)[-‐‑]\n[ \t]*([a-zß-ÿ])")
_INLINE_WS = re.compile(r"[ \t\f\v]+")
_MANY_NL = re.compile(r"\n{3,}")
_DIGITS = re.compile(r"\d+")
_PAGE_NO = re.compile(r"[-–\s]*(?:(?:page|seite|p\.|s\.)\s*)?#\s*(?:(?:of|von|/)\s*#)?[-–\s]*")

_BULLET = re.compile(r"^\s*(?:[-*+•▪●◦‣–]|\(?[0-9]{1,2}[.)]|\(?[a-z][.)])\s+")
_NUMBERED_HEADING = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,4})\.?\s+(\S.{0,100})$")
_TERMINAL = (".", "!", "?", ";", ",")


def normalize_text(text: str) -> str:
    """NFKC, drop zero-width/soft hyphens, de-hyphenate line breaks, clean whitespace.

    Line structure is kept (single ``\\n``); paragraph breaks are at most one blank line.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _PDFIUM_HYPHEN.sub("", text).translate(_ZERO_WIDTH)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    lines = [_INLINE_WS.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return _MANY_NL.sub("\n\n", text).strip()


def clean_inline(text: str) -> str:
    """Normalise and collapse to a single line (for paragraph/heading block text)."""
    return " ".join(normalize_text(text).split())


def _edge_key(line: str) -> str:
    return _DIGITS.sub("#", " ".join(line.lower().split()))


def strip_repeated_edges(
    pages: Sequence[str], edge_lines: int = 3, min_pages: int = 3, min_fraction: float = 0.5
) -> list[str]:
    """Remove running headers/footers: lines at the top/bottom ``edge_lines`` of a page that
    repeat (digits ignored, e.g. "Page 3 of 9") on at least ``min_fraction`` of the pages.
    Only applied to documents with ``>= min_pages`` pages.
    """
    if len(pages) < min_pages:
        return list(pages)
    split = [[ln for ln in p.replace("\r\n", "\n").split("\n")] for p in pages]
    counts: Counter[str] = Counter()
    raw_counts: Counter[str] = Counter()
    for lines in split:
        nonempty = [ln for ln in lines if ln.strip()]
        edge = nonempty[:edge_lines] + nonempty[-edge_lines:]
        counts.update({_edge_key(ln) for ln in edge})
        raw_counts.update({" ".join(ln.split()) for ln in edge})
    threshold = max(2, int(min_fraction * len(pages) + 0.999))
    # identical lines, or lines that only differ by a page number ("Page 3 of 9", "- 4 -")
    repeated = {_edge_key(r) for r, c in raw_counts.items() if c >= threshold and r}
    repeated |= {k for k, c in counts.items() if c >= threshold and _PAGE_NO.fullmatch(k)}
    if not repeated:
        return list(pages)
    out: list[str] = []
    for lines in split:
        idx = [i for i, ln in enumerate(lines) if ln.strip()]
        edge_idx = set(idx[:edge_lines] + idx[-edge_lines:])
        kept = [ln for i, ln in enumerate(lines)
                if not (i in edge_idx and _edge_key(ln) in repeated)]
        out.append("\n".join(kept))
    return out


def heading_level(line: str) -> int | None:
    """Heuristic heading detection for text without style information.

    ``"3.2 Fault codes"`` -> 2, ``"MAINTENANCE SCHEDULE"`` -> 1, else ``None``.
    """
    s = line.strip()
    if not s or len(s) > 100 or s.endswith(_TERMINAL) or s.endswith(":"):
        return None
    m = _NUMBERED_HEADING.match(s)
    if m:
        title = m.group(2)
        if title[0].isalpha() and title[0].isupper() and len(title.split()) <= 10:
            return m.group(1).count(".") + 1
        return None
    letters = [c for c in s if c.isalpha()]
    if len(letters) >= 4 and all(c.isupper() for c in letters) and len(s.split()) <= 8:
        return 1
    return None


def is_bullet(line: str) -> bool:
    return bool(_BULLET.match(line))


def split_paragraphs(text: str) -> list[str]:
    """Split on blank lines; drop empties."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def blocks_from_text(
    text: str,
    page: int | None = None,
    *,
    detect_headings: bool = True,
    wrapped_lines: bool = True,
) -> list[ParsedBlock]:
    """Recover headings, list blocks and paragraphs from normalised plain text.

    ``wrapped_lines``: lines inside a paragraph are soft-wrapped (PDF/OCR/TXT) and get
    joined; a short line ending with sentence punctuation also closes a paragraph.
    """
    blocks: list[ParsedBlock] = []
    buf: list[str] = []
    items: list[str] = []

    def flush_para() -> None:
        if buf:
            blocks.append(ParsedBlock("paragraph", " ".join(buf), page))
            buf.clear()

    def flush_list() -> None:
        if items:
            blocks.append(ParsedBlock("list", "\n".join(items), page))
            items.clear()

    for para in split_paragraphs(text):
        lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
        width = max(len(ln) for ln in lines)
        for line in lines:
            level = heading_level(line) if detect_headings else None
            if level is not None:
                flush_para()
                flush_list()
                blocks.append(ParsedBlock("heading", line, page, level=level))
                continue
            if is_bullet(line):
                flush_para()
                items.append(line)
                continue
            if items and wrapped_lines and not buf and not items[-1].endswith((".", "!", "?")):
                # continuation of a wrapped list item
                items[-1] = f"{items[-1]} {line}"
                continue
            flush_list()
            buf.append(line)
            closes = line.endswith((".", "!", "?", ":")) and len(line) < 0.7 * width
            if not wrapped_lines or closes:
                flush_para()
        flush_para()
        flush_list()
    return blocks
