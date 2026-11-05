"""Chunk-grounded task generators. Each returns judge-ready :class:`Candidate` objects and
counts every candidate it creates or drops in :class:`SynthStats`.

* quote-first QA (Mode A core): quotes -> one question/answer per verified quote
* summarisation, definition/acronym (from the taxonomy), drafting (Mode B back-translation)
* extraction (Mode C): schema-valid JSON whose every leaf is verbatim in / entailed by the chunk
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import Any

from jsonschema import validators

from zllf.core.schemas import Chunk
from zllf.llm.base import NLIScorer
from zllf.stages.synth.common import Candidate, SynthStats, Teacher
from zllf.stages.synth.judge import MAX_QUOTE_CHARS, MIN_QUOTE_CHARS, premise_for, verify_quote
from zllf.stages.synth.prompts import render, teacher_chat
from zllf.stages.synth.taxonomy import TermInfo
from zllf.stages.synth.textutil import contains_token_seq, parse_json_object, str_field

QA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "answer": {"type": "string"},
        "statement": {"type": "string"},
    },
    "required": ["question", "answer", "statement"],
}
SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "instruction": {"type": "string"},
        "summary": {"type": "string"},
        "quote": {"type": "string"},
    },
    "required": ["instruction", "summary", "quote"],
}
DEFINITION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "answer": {"type": "string"},
        "quote": {"type": "string"},
    },
    "required": ["question", "answer", "quote"],
}
DRAFTING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"passage_excerpt": {"type": "string"}, "notes": {"type": "string"}},
    "required": ["passage_excerpt", "notes"],
}
# Used in Mode C when the profile does not provide ``json_schema``.
DEFAULT_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "values": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {"label": {"type": "string"}, "value": {"type": "string"}},
                "required": ["label", "value"],
            },
        },
    },
    "required": ["values"],
}


def quotes_schema(lo: int, hi: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "quotes": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": max(1, lo),
                "maxItems": max(1, hi),
            }
        },
        "required": ["quotes"],
    }


def _heading(chunk: Chunk) -> str:
    return " > ".join(chunk.heading_path)


# ---------------------------------------------------------------- quote-first QA
def generate_qa(
    teacher: Teacher, chunks: Sequence[Chunk], stats: SynthStats, quotes_range: tuple[int, int]
) -> list[Candidate]:
    """Step 1: teacher copies key quotes (verified verbatim). Step 2: one QA pair per quote."""
    lo, hi = quotes_range
    chats = [
        teacher_chat("quotes", passage=c.text, heading=_heading(c), min_quotes=lo, max_quotes=hi)
        for c in chunks
    ]
    outputs = teacher.run("quotes", chats, json_schema=quotes_schema(lo, hi), max_tokens=768)
    pending: list[tuple[Chunk, str, tuple[int, int] | None]] = []
    for chunk, raw in zip(chunks, outputs, strict=True):
        obj = parse_json_object(raw)
        quotes = obj.get("quotes") if obj is not None else None
        if not isinstance(quotes, list):
            stats.malformed["quotes"] += 1
            continue
        seen: set[str] = set()
        for q in quotes[: max(1, hi)]:
            if not isinstance(q, str):
                continue
            stats.candidates["qa"] += 1
            res = verify_quote(chunk.text, q)
            ok = res is not None and MIN_QUOTE_CHARS <= len(res[0]) <= MAX_QUOTE_CHARS
            stats.check("substring", ok)
            if res is None:
                stats.drop("qa", "quote_not_in_chunk")
                continue
            if not ok:
                stats.drop("qa", "quote_length")
                continue
            if res[0] in seen:
                stats.drop("qa", "duplicate_quote")
                continue
            seen.add(res[0])
            stats.quotes_repaired += int(res[2])
            pending.append((chunk, res[0], res[1]))

    chats = [teacher_chat("qa", quote=q, passage=c.text) for c, q, _ in pending]
    outputs = teacher.run("qa", chats, json_schema=QA_SCHEMA, max_tokens=384)
    out: list[Candidate] = []
    for (chunk, quote, span), raw in zip(pending, outputs, strict=True):
        obj = parse_json_object(raw)
        question = str_field(obj, "question", 600) if obj else None
        answer = str_field(obj, "answer", 2000) if obj else None
        if question is None or answer is None:
            stats.malformed["qa"] += 1
            stats.drop("qa", "malformed_json")
            continue
        statement = (str_field(obj, "statement", 2000) if obj else None) or answer
        out.append(Candidate(
            task="qa", user=question, answer=answer, chunk=chunk, doc_id=chunk.doc_id,
            quote=quote, hypothesis=statement, premise=premise_for(chunk.text, span),
            needs_quote_check=False,  # verified above
        ))
    return out


# ---------------------------------------------------------------- summarisation
def generate_summaries(
    teacher: Teacher, chunks: Sequence[Chunk], stats: SynthStats
) -> list[Candidate]:
    chats = [teacher_chat("summarization", passage=c.text, heading=_heading(c)) for c in chunks]
    outputs = teacher.run("summarization", chats, json_schema=SUMMARY_SCHEMA, max_tokens=512)
    out: list[Candidate] = []
    for chunk, raw in zip(chunks, outputs, strict=True):
        stats.candidates["summarization"] += 1
        obj = parse_json_object(raw)
        fields = [str_field(obj, k) if obj else None for k in ("instruction", "summary", "quote")]
        instruction, summary, quote = fields
        if instruction is None or summary is None or quote is None:
            stats.malformed["summarization"] += 1
            stats.drop("summarization", "malformed_json")
            continue
        out.append(Candidate(
            task="summarization", user=instruction, answer=summary, chunk=chunk,
            doc_id=chunk.doc_id, quote=quote, hypothesis=summary, premise=chunk.text,
        ))
    return out


# ---------------------------------------------------------------- definitions
def generate_definitions(
    teacher: Teacher,
    terms: Sequence[TermInfo],
    chunk_by_id: dict[str, Chunk],
    stats: SynthStats,
) -> list[Candidate]:
    """One definition/acronym question per taxonomy term, grounded in its first source chunk."""
    pairs = [(t, chunk_by_id[t.source_chunk_ids[0]]) for t in terms
             if t.source_chunk_ids and t.source_chunk_ids[0] in chunk_by_id]
    chats = [teacher_chat("definition", passage=c.text, term=t.term) for t, c in pairs]
    outputs = teacher.run("definition", chats, json_schema=DEFINITION_SCHEMA, max_tokens=384)
    out: list[Candidate] = []
    for (_, chunk), raw in zip(pairs, outputs, strict=True):
        stats.candidates["definition"] += 1
        obj = parse_json_object(raw)
        question, answer, quote = (
            [str_field(obj, k) for k in ("question", "answer", "quote")] if obj else [None] * 3
        )
        if question is None or answer is None or quote is None:
            stats.malformed["definition"] += 1
            stats.drop("definition", "malformed_json")
            continue
        out.append(Candidate(
            task="definition", user=question, answer=answer, chunk=chunk, doc_id=chunk.doc_id,
            quote=quote, hypothesis=answer,
        ))
    return out


# ---------------------------------------------------------------- drafting (Mode B)
def generate_drafting(
    teacher: Teacher, chunks: Sequence[Chunk], stats: SynthStats
) -> list[Candidate]:
    """Back-translation: verbatim house-style excerpt is the target, teacher notes the input."""
    chats = [teacher_chat("drafting", passage=c.text, heading=_heading(c)) for c in chunks]
    outputs = teacher.run("drafting", chats, json_schema=DRAFTING_SCHEMA, max_tokens=768)
    out: list[Candidate] = []
    for chunk, raw in zip(chunks, outputs, strict=True):
        stats.candidates["drafting"] += 1
        obj = parse_json_object(raw)
        excerpt = str_field(obj, "passage_excerpt") if obj else None
        notes = str_field(obj, "notes") if obj else None
        if excerpt is None or notes is None:
            stats.malformed["drafting"] += 1
            stats.drop("drafting", "malformed_json")
            continue
        res = verify_quote(chunk.text, excerpt)
        ok = res is not None and len(res[0]) >= MIN_QUOTE_CHARS
        stats.check("substring", ok)
        if res is None or not ok:
            stats.drop("drafting", "quote_not_in_chunk" if res is None else "quote_length")
            continue
        stats.quotes_repaired += int(res[2])
        verbatim = res[0]
        out.append(Candidate(
            task="drafting", user=render("sample_drafting.user.j2", notes=notes),
            answer=verbatim, chunk=chunk, doc_id=chunk.doc_id, quote=verbatim,
            # notes must not add facts that the target text lacks
            premise=verbatim, hypothesis=notes, judge_question=notes,
            needs_quote_check=False,
        ))
    return out


# ---------------------------------------------------------------- extraction (Mode C)
def iter_leaves(obj: Any, path: str = "$") -> Iterator[tuple[str, Any]]:
    """(json-path, scalar) for every leaf of a JSON value."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from iter_leaves(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from iter_leaves(v, f"{path}[{i}]")
    else:
        yield path, obj


def _leaf_text(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    return text or None


def _leaf_label(path: str) -> str:
    last = path.rsplit(".", 1)[-1]
    return last.split("[", 1)[0].replace("_", " ")


def check_extraction(
    instance: Any,
    chunk_text: str,
    schema: dict[str, Any],
    nli: NLIScorer,
    threshold: float,
) -> tuple[str | None, float | None, str | None]:
    """Validate an extracted instance. Returns (drop_reason | None, min_leaf_score, quote).

    Every string/number leaf must appear verbatim (word sequence) in the chunk or be entailed
    by it (NLI ``"<label>: <value>"`` >= threshold). Booleans/nulls are schema-checked only.
    """
    validator_cls = validators.validator_for(schema)
    if not validator_cls(schema).is_valid(instance):
        return "extraction_schema_invalid", None, None
    leaves = [(p, t) for p, v in iter_leaves(instance) if (t := _leaf_text(v)) is not None]
    if not leaves:
        return "extraction_empty", None, None
    scores: dict[str, float] = {}
    to_nli: list[tuple[str, str]] = []
    for path, text in leaves:
        if contains_token_seq(chunk_text, text):
            scores[path] = 1.0
        else:
            to_nli.append((path, f"{_leaf_label(path)}: {text}"))
    if to_nli:
        ent = nli.entailment([chunk_text] * len(to_nli), [h for _, h in to_nli])
        scores.update({p: float(s) for (p, _), s in zip(to_nli, ent, strict=True)})
    min_score = min(scores.values())
    if min_score < threshold:
        return "extraction_leaf_unsupported", round(min_score, 4), None
    verbatim = [t for p, t in leaves if scores[p] == 1.0 and len(t) >= MIN_QUOTE_CHARS]
    quote = None
    for text in sorted(verbatim, key=len, reverse=True):
        res = verify_quote(chunk_text, text)
        if res is not None:
            quote = res[0]
            break
    return None, round(min_score, 4), quote


def generate_extraction(
    teacher: Teacher,
    chunks: Sequence[Chunk],
    stats: SynthStats,
    schema: dict[str, Any],
    nli: NLIScorer,
    threshold: float,
) -> list[Candidate]:
    schema_json = json.dumps(schema, ensure_ascii=False, sort_keys=True)
    chats = [teacher_chat("extraction", passage=c.text, schema_json=schema_json) for c in chunks]
    outputs = teacher.run("extraction", chats, json_schema=schema, max_tokens=1024)
    out: list[Candidate] = []
    for chunk, raw in zip(chunks, outputs, strict=True):
        stats.candidates["extraction"] += 1
        instance = parse_json_object(raw)
        if instance is None:
            stats.malformed["extraction"] += 1
            stats.drop("extraction", "malformed_json")
            continue
        reason, score, quote = check_extraction(instance, chunk.text, schema, nli, threshold)
        stats.check("extraction_leaves", reason is None)
        if reason is not None:
            stats.drop("extraction", reason)
            continue
        user = render("sample_extraction.user.j2", passage=chunk.text)
        out.append(Candidate(
            task="extraction", user=user,
            answer=json.dumps(instance, ensure_ascii=False),
            chunk=chunk, doc_id=chunk.doc_id, quote=quote, reference_json=instance,
            entailment=score, judge_question="Extract the structured data from the passage.",
            needs_quote_check=False, needs_nli=False,
        ))
    return out
