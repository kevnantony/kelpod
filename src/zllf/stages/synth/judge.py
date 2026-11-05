"""Judge loop: (1) verbatim quote check, (2) NLI entailment, (3) LLM judge, (4) dedup.

Every candidate either survives all filters or is dropped with a counted reason.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from zllf.llm.base import Embedder, NLIScorer
from zllf.stages.synth.common import EMBED_BATCH, NLI_BATCH, Candidate, SynthStats, Teacher
from zllf.stages.synth.prompts import teacher_chat
from zllf.stages.synth.textutil import (
    contains_normalized,
    find_span_relaxed,
    normalize_ws,
    parse_json_object,
    window,
)

MIN_QUOTE_CHARS = 12
MAX_QUOTE_CHARS = 1200
PREMISE_WINDOW_CHARS = 400  # quote ± window used as NLI premise for long chunks
LONG_CHUNK_CHARS = 1200

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answerable": {"enum": ["yes", "no"]},
        "complete": {"enum": ["yes", "no"]},
        "reason": {"type": "string"},
    },
    "required": ["answerable", "complete"],
}


def verify_quote(chunk_text: str, quote: str) -> tuple[str, tuple[int, int] | None, bool] | None:
    """Return (verbatim_quote, span, repaired) or None if the quote is not in the chunk.

    Exact substring after whitespace normalisation passes as-is. A quote that only differs
    in case/punctuation is *repaired*: replaced by the verbatim source span, so every stored
    quote is a true substring of its chunk.
    """
    q = normalize_ws(quote)
    if not q:
        return None
    if contains_normalized(chunk_text, q):
        idx = chunk_text.find(q)
        span = (idx, idx + len(q)) if idx >= 0 else find_span_relaxed(chunk_text, q)
        return q, span, False
    span = find_span_relaxed(chunk_text, q)
    if span is None:
        return None
    return normalize_ws(chunk_text[span[0] : span[1]]), span, True


def premise_for(chunk_text: str, span: tuple[int, int] | None) -> str | None:
    """Quote ± window for long chunks, else None (= whole chunk)."""
    if span is None or len(chunk_text) <= LONG_CHUNK_CHARS:
        return None
    return window(chunk_text, span[0], span[1], PREMISE_WINDOW_CHARS)


def quote_filter(cands: Sequence[Candidate], stats: SynthStats) -> list[Candidate]:
    """Filter 1: the supporting quote must occur verbatim in the source chunk."""
    kept: list[Candidate] = []
    for c in cands:
        if not c.needs_quote_check or c.chunk is None:
            kept.append(c)
            continue
        res = verify_quote(c.chunk.text, c.quote or "")
        ok = res is not None and MIN_QUOTE_CHARS <= len(res[0]) <= MAX_QUOTE_CHARS
        stats.check("substring", ok)
        if not ok or res is None:
            stats.drop(c.task, "quote_not_in_chunk" if res is None else "quote_length")
            continue
        c.quote = res[0]
        if res[2]:
            stats.quotes_repaired += 1
        if c.premise is None:
            c.premise = premise_for(c.chunk.text, res[1])
        kept.append(c)
    return kept


def nli_filter(
    cands: Sequence[Candidate], nli: NLIScorer, threshold: float, stats: SynthStats
) -> list[Candidate]:
    """Filter 2: premise (chunk or quote window) must entail the answer statement."""
    todo = [c for c in cands if c.needs_nli and c.chunk is not None]
    scores: list[float] = []
    for start in range(0, len(todo), NLI_BATCH):
        part = todo[start : start + NLI_BATCH]
        premises = [c.premise or c.chunk.text for c in part if c.chunk is not None]
        hyps = [c.hypothesis or c.answer for c in part]
        scores.extend(nli.entailment(premises, hyps))
    by_id = {id(c): s for c, s in zip(todo, scores, strict=True)}
    kept: list[Candidate] = []
    for c in cands:
        if id(c) not in by_id:  # verified elsewhere (extraction leaves) or not applicable
            kept.append(c)
            continue
        score = float(by_id[id(c)])
        c.entailment = round(score, 4)
        ok = score >= threshold
        stats.check("nli", ok)
        if not ok:
            stats.drop(c.task, "nli_below_threshold")
            continue
        kept.append(c)
    return kept


def parse_verdict(raw: str) -> bool | None:
    """True/False from judge output; None if malformed."""
    obj = parse_json_object(raw)
    if obj is not None:
        a, b = obj.get("answerable"), obj.get("complete")
        if isinstance(a, str) and isinstance(b, str):
            a, b = a.strip().lower(), b.strip().lower()
            if a in ("yes", "no") and b in ("yes", "no"):
                return a == "yes" and b == "yes"
        return None
    head = raw.strip().lower()[:8]
    if head.startswith("yes"):
        return True
    if head.startswith("no"):
        return False
    return None


def llm_judge_filter(
    cands: Sequence[Candidate], teacher: Teacher, stats: SynthStats
) -> list[Candidate]:
    """Filter 3: teacher (separate prompt) checks answerability + completeness."""
    todo = [c for c in cands if c.needs_judge and c.chunk is not None]
    chats = [
        teacher_chat(
            "judge",
            passage=c.premise or (c.chunk.text if c.chunk else ""),
            question=c.judge_question or c.user,
            answer=c.answer,
        )
        for c in todo
    ]
    outputs = teacher.run("judge", chats, json_schema=JUDGE_SCHEMA, max_tokens=128)
    verdicts = {id(c): parse_verdict(o) for c, o in zip(todo, outputs, strict=True)}
    kept: list[Candidate] = []
    for c in cands:
        if id(c) not in verdicts:
            kept.append(c)
            continue
        v = verdicts[id(c)]
        if v is None:
            stats.malformed["judge"] += 1
            stats.check("llm_judge", False)
            stats.drop(c.task, "judge_malformed")
            continue
        c.judge_ok = v
        stats.check("llm_judge", v)
        if not v:
            stats.drop(c.task, "judge_rejected")
            continue
        kept.append(c)
    return kept


def embed(embedder: Embedder, texts: Sequence[str]) -> np.ndarray:
    parts = [embedder.encode(list(texts[i : i + EMBED_BATCH]))
             for i in range(0, len(texts), EMBED_BATCH)]
    if not parts:
        return np.zeros((0, 1), dtype=np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32)


def greedy_dedup(emb: np.ndarray, threshold: float) -> list[int]:
    """Indices kept by greedy near-duplicate removal (first occurrence wins)."""
    kept: list[int] = []
    for i in range(emb.shape[0]):
        if kept and float(np.max(emb[kept] @ emb[i])) >= threshold:
            continue
        kept.append(i)
    return kept


def dedup_filter(
    cands: Sequence[Candidate], embedder: Embedder, threshold: float, stats: SynthStats
) -> list[Candidate]:
    """Filter 4: drop near-duplicate questions (cosine >= threshold) within each task."""
    keep_ids: set[int] = set()
    for task in sorted({c.task for c in cands}):
        group = [c for c in cands if c.task == task]
        emb = embed(embedder, [c.user for c in group])
        idx = set(greedy_dedup(emb, threshold))
        for i, c in enumerate(group):
            stats.check("dedup", i in idx)
            if i in idx:
                keep_ids.add(id(c))
            else:
                stats.drop(task, "near_duplicate_question")
    return [c for c in cands if id(c) in keep_ids]
