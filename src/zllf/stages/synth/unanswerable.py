"""Unanswerable in-domain questions with a calibrated abstention target.

The teacher proposes plausible questions plus the fact that would answer them
(``assumed_fact``). A question is kept only if it is verified absent from the corpus:
its embedding is not close to any chunk AND no top-k chunk entails the assumed fact.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from zllf.core.schemas import Chunk
from zllf.llm.base import Embedder, NLIScorer
from zllf.stages.synth.common import Candidate, SynthStats, Teacher
from zllf.stages.synth.judge import embed
from zllf.stages.synth.prompts import teacher_chat
from zllf.stages.synth.textutil import abstention, guess_language, parse_json_object, str_field

QUESTIONS_PER_PROMPT = 5
MAX_CHUNK_COSINE = 0.80  # question vs. any chunk: above -> probably answerable
MAX_ENTAILMENT = 0.50  # any top-k chunk entailing the assumed fact -> answerable
TOP_K = 5
OVERSAMPLE = 2.0
LANGUAGE_NAMES = {"en": "English", "de": "German"}


def unanswerable_schema(n: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "minItems": 1,
                "maxItems": n,
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "assumed_fact": {"type": "string"},
                    },
                    "required": ["question", "assumed_fact"],
                },
            }
        },
        "required": ["items"],
    }


def pick_seed_chunks(chunks: Sequence[Chunk], n: int) -> list[Chunk]:
    """Round-robin over documents (sorted) so every split gets unanswerables."""
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    order = sorted(by_doc)
    out: list[Chunk] = []
    i = 0
    while len(out) < n and by_doc:
        progressed = False
        for doc in order:
            lst = by_doc[doc]
            if i < len(lst):
                out.append(lst[i])
                progressed = True
                if len(out) == n:
                    break
        if not progressed:
            break
        i += 1
    return out


def generate_unanswerable(
    teacher: Teacher,
    seeds: Sequence[Chunk],
    terms: Sequence[str],
    stats: SynthStats,
) -> list[Candidate]:
    """Teacher proposals (not yet verified). Language follows the seed chunk."""
    langs = [guess_language(c.text) for c in seeds]
    chats = [
        teacher_chat(
            "unanswerable", passage=c.text, terms=list(terms), n_questions=QUESTIONS_PER_PROMPT,
            language_name=LANGUAGE_NAMES.get(lang, "English"),
        )
        for c, lang in zip(seeds, langs, strict=True)
    ]
    outputs = teacher.run(
        "unanswerable", chats, json_schema=unanswerable_schema(QUESTIONS_PER_PROMPT),
        max_tokens=768,
    )
    out: list[Candidate] = []
    for chunk, lang, raw in zip(seeds, langs, outputs, strict=True):
        obj = parse_json_object(raw)
        items = obj.get("items") if obj is not None else None
        if not isinstance(items, list):
            stats.malformed["unanswerable"] += 1
            continue
        for item in items[:QUESTIONS_PER_PROMPT]:
            stats.candidates["unanswerable"] += 1
            q = str_field(item, "question", 600) if isinstance(item, dict) else None
            fact = str_field(item, "assumed_fact", 1000) if isinstance(item, dict) else None
            if q is None or fact is None:
                stats.drop("unanswerable", "malformed_json")
                continue
            out.append(Candidate(
                task="unanswerable", user=q, answer=abstention(lang), doc_id=chunk.doc_id,
                hypothesis=fact, answerable=False,
                needs_quote_check=False, needs_nli=False, needs_judge=False,
            ))
    return out


def verify_absent(
    cands: Sequence[Candidate],
    all_chunks: Sequence[Chunk],
    chunk_emb: np.ndarray,
    embedder: Embedder,
    nli: NLIScorer,
    stats: SynthStats,
) -> list[Candidate]:
    """Keep questions with low similarity to every chunk and no entailing chunk."""
    if not cands:
        return []
    q_emb = embed(embedder, [c.user for c in cands])
    sims = q_emb @ chunk_emb.T if chunk_emb.size else np.zeros((len(cands), 0))
    survivors: list[tuple[Candidate, list[int]]] = []
    for i, c in enumerate(cands):
        row = sims[i]
        max_sim = float(row.max()) if row.size else 0.0
        ok = max_sim < MAX_CHUNK_COSINE
        stats.check("unanswerable_embedding", ok)
        if not ok:
            stats.drop("unanswerable", "similar_to_corpus")
            continue
        top = np.argsort(-row, kind="stable")[:TOP_K].tolist() if row.size else []
        survivors.append((c, top))
    premises = [all_chunks[j].text for _, top in survivors for j in top]
    hyps = [c.hypothesis or c.user for c, top in survivors for _ in top]
    scores = nli.entailment(premises, hyps) if premises else []
    kept: list[Candidate] = []
    pos = 0
    for c, top in survivors:
        part = scores[pos : pos + len(top)]
        pos += len(top)
        ok = (max(part) if part else 0.0) < MAX_ENTAILMENT
        stats.check("unanswerable_nli", ok)
        if not ok:
            stats.drop("unanswerable", "entailed_by_corpus")
            continue
        kept.append(c)
    return kept


def cap_round_robin(cands: Sequence[Candidate], target: int) -> list[Candidate]:
    """Keep at most ``target`` items, interleaving documents (deterministic)."""
    by_doc: dict[str, list[Candidate]] = {}
    for c in cands:
        by_doc.setdefault(c.doc_id or "", []).append(c)
    out: list[Candidate] = []
    i = 0
    while len(out) < target:
        added = False
        for doc in sorted(by_doc):
            if i < len(by_doc[doc]) and len(out) < target:
                out.append(by_doc[doc][i])
                added = True
        if not added:
            break
        i += 1
    return out
