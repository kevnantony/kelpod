"""Taxonomy extraction (entities, acronyms + expansions, concepts) and cross-chunk merge."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from zllf.core.schemas import Chunk
from zllf.stages.synth.common import Teacher
from zllf.stages.synth.prompts import teacher_chat
from zllf.stages.synth.textutil import (
    PLACEHOLDER,
    contains_token_seq,
    normalize_ws,
    parse_json_object,
)

MAX_TERM_CHARS = 80
MAX_ACRONYM_CHARS = 12
MAX_ITEMS_PER_LIST = 15

TAXONOMY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "maxItems": MAX_ITEMS_PER_LIST,
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "type": {"type": "string"}},
                "required": ["name", "type"],
            },
        },
        "acronyms": {
            "type": "array",
            "maxItems": MAX_ITEMS_PER_LIST,
            "items": {
                "type": "object",
                "properties": {"acronym": {"type": "string"}, "expansion": {"type": "string"}},
                "required": ["acronym", "expansion"],
            },
        },
        "concepts": {
            "type": "array",
            "maxItems": MAX_ITEMS_PER_LIST,
            "items": {
                "type": "object",
                "properties": {"term": {"type": "string"}},
                "required": ["term"],
            },
        },
    },
    "required": ["entities", "acronyms", "concepts"],
}


@dataclass
class TermInfo:
    term: str
    kind: str  # entity | acronym | concept
    detail: str = ""  # entity type or acronym expansion (short)
    count: int = 0
    source_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class Taxonomy:
    terms: dict[tuple[str, str], TermInfo] = field(default_factory=dict)

    def add(self, kind: str, term: str, detail: str, chunk_id: str) -> None:
        key = (kind, term.casefold())
        info = self.terms.get(key)
        if info is None:
            info = self.terms[key] = TermInfo(term=term, kind=kind, detail=detail)
        if not info.detail and detail:
            info.detail = detail
        info.count += 1
        if chunk_id not in info.source_chunk_ids:
            info.source_chunk_ids.append(chunk_id)

    def of_kind(self, kind: str) -> list[TermInfo]:
        items = [t for t in self.terms.values() if t.kind == kind]
        return sorted(items, key=lambda t: (-t.count, t.term.casefold()))

    def names(self, limit: int = 40) -> list[str]:
        """Most frequent terms of all kinds (for unanswerable prompts)."""
        items = sorted(self.terms.values(), key=lambda t: (-t.count, t.kind, t.term.casefold()))
        return [t.term for t in items[:limit]]

    def to_json(self) -> dict[str, Any]:
        """taxonomy.json payload: terms and counts only, no long text."""
        return {
            "entities": [
                {"name": t.term, "type": t.detail, "count": t.count}
                for t in self.of_kind("entity")
            ],
            "acronyms": [
                {"acronym": t.term, "expansion": t.detail, "count": t.count}
                for t in self.of_kind("acronym")
            ],
            "concepts": [{"term": t.term, "count": t.count} for t in self.of_kind("concept")],
        }


def _clean_term(value: Any, max_chars: int) -> str | None:
    if not isinstance(value, str):
        return None
    v = normalize_ws(value).strip(" .,;:")
    if not v or len(v) > max_chars or PLACEHOLDER.search(v):
        return None
    return v


def parse_taxonomy(raw: str, chunk: Chunk) -> list[tuple[str, str, str]] | None:
    """(kind, term, detail) triples grounded in the chunk text; None if malformed."""
    obj = parse_json_object(raw)
    if obj is None:
        return None
    out: list[tuple[str, str, str]] = []
    lists = {k: obj.get(k, []) for k in ("entities", "acronyms", "concepts")}
    if not all(isinstance(v, list) for v in lists.values()):
        return None
    for item in lists["entities"][:MAX_ITEMS_PER_LIST]:
        if isinstance(item, dict):
            name = _clean_term(item.get("name"), MAX_TERM_CHARS)
            typ = _clean_term(item.get("type"), 40) or ""
            if name and contains_token_seq(chunk.text, name):
                out.append(("entity", name, typ))
    for item in lists["acronyms"][:MAX_ITEMS_PER_LIST]:
        if isinstance(item, dict):
            acr = _clean_term(item.get("acronym"), MAX_ACRONYM_CHARS)
            exp = _clean_term(item.get("expansion"), MAX_TERM_CHARS) or ""
            if acr and " " not in acr and acr in chunk.text:
                out.append(("acronym", acr, exp))
    for item in lists["concepts"][:MAX_ITEMS_PER_LIST]:
        term = _clean_term(item.get("term") if isinstance(item, dict) else item, MAX_TERM_CHARS)
        if term and contains_token_seq(chunk.text, term):
            out.append(("concept", term, ""))
    return out


def extract_taxonomy(teacher: Teacher, chunks: Sequence[Chunk]) -> Taxonomy:
    """Run the taxonomy prompt on every chunk and merge the (grounded) terms."""
    chats = [
        teacher_chat("taxonomy", passage=c.text, heading=" > ".join(c.heading_path))
        for c in chunks
    ]
    outputs = teacher.run("taxonomy", chats, json_schema=TAXONOMY_SCHEMA, max_tokens=512)
    tax = Taxonomy()
    for chunk, raw in zip(chunks, outputs, strict=True):
        triples = parse_taxonomy(raw, chunk)
        if triples is None:
            teacher.stats.malformed["taxonomy"] += 1
            continue
        for kind, term, detail in triples:
            tax.add(kind, term, detail, chunk.chunk_id)
    return tax
