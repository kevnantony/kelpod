"""Small pure text helpers for synthesis: whitespace normalisation, quote location,
robust JSON parsing of teacher output and a cheap language guess."""

from __future__ import annotations

import json
import re
from typing import Any

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"\w+", re.UNICODE)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
PLACEHOLDER = re.compile(r"\[[A-Z][A-Z0-9_]*_\d+\]")

_EN_STOP = frozenset((
    "the", "and", "of", "to", "is", "in", "that", "for", "are", "with", "as", "on",
    "this", "be", "by", "it", "from", "or", "which", "an", "at", "not",
))
_DE_STOP = frozenset((
    "der", "die", "das", "und", "ist", "nicht", "mit", "den", "von", "zu", "im", "für",
    "auf", "des", "ein", "eine", "dem", "sich", "bei", "werden", "wird", "oder", "auch",
    "nach",
))

ABSTENTIONS: dict[str, str] = {
    "en": "The documents do not contain this information.",
    "de": "Die Dokumente enthalten diese Information nicht.",
}


def normalize_ws(text: str) -> str:
    """Collapse all whitespace runs to one space and strip."""
    return _WS.sub(" ", text).strip()


def tokens(text: str) -> list[str]:
    """Lower-cased word tokens (unicode aware)."""
    return [t.lower() for t in _TOKEN.findall(text)]


def contains_normalized(haystack: str, needle: str) -> bool:
    """Exact substring check after whitespace normalisation (judge filter 1)."""
    n = normalize_ws(needle)
    return bool(n) and n in normalize_ws(haystack)


def find_span_relaxed(text: str, needle: str) -> tuple[int, int] | None:
    """Locate ``needle`` in ``text`` ignoring case, punctuation and whitespace.

    Returns the (start, end) character span of the matching word sequence in ``text`` so the
    caller can replace a slightly mangled teacher quote with the verbatim source span.
    """
    want = tokens(needle)
    if not want:
        return None
    matches = list(_TOKEN.finditer(text))
    words = [m.group(0).lower() for m in matches]
    k = len(want)
    for i in range(len(words) - k + 1):
        if words[i : i + k] == want:
            return matches[i].start(), matches[i + k - 1].end()
    return None


def contains_token_seq(haystack: str, needle: str) -> bool:
    """True if the word sequence of ``needle`` appears contiguously in ``haystack``."""
    return find_span_relaxed(haystack, needle) is not None


def window(text: str, start: int, end: int, chars: int) -> str:
    """``text[start:end]`` extended by up to ``chars`` on both sides, snapped to whitespace."""
    lo = max(0, start - chars)
    hi = min(len(text), end + chars)
    if lo > 0:
        sp = text.find(" ", lo, start)
        lo = sp + 1 if sp != -1 else lo
    if hi < len(text):
        sp = text.rfind(" ", end, hi)
        hi = sp if sp != -1 else hi
    return text[lo:hi]


def parse_json_object(raw: str) -> dict[str, Any] | None:
    """Parse a JSON object from teacher output; tolerate code fences and chatter.

    Returns None when no JSON object can be recovered (caller counts it as malformed).
    """
    s = _FENCE.sub("", raw.strip())
    for candidate in (s, _outer_braces(s)):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _outer_braces(s: str) -> str | None:
    lo, hi = s.find("{"), s.rfind("}")
    return s[lo : hi + 1] if 0 <= lo < hi else None


def str_field(obj: dict[str, Any], key: str, max_len: int = 4000) -> str | None:
    """A non-empty, whitespace-normalised string field or None."""
    v = obj.get(key)
    if not isinstance(v, str):
        return None
    v = normalize_ws(v)
    return v if v and len(v) <= max_len else None


def guess_language(text: str) -> str:
    """'de' or 'en' by stop-word counts (cheap; only used to pick the abstention text)."""
    toks = tokens(text)
    de = sum(1 for t in toks if t in _DE_STOP)
    en = sum(1 for t in toks if t in _EN_STOP)
    return "de" if de > en else "en"


def abstention(language: str) -> str:
    return ABSTENTIONS.get(language, ABSTENTIONS["en"])
