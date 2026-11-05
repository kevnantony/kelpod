"""Helpers for synth tests: fake sanitised chunks and a scripted (well-behaved) teacher."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from zllf.core.schemas import Chunk
from zllf.llm.base import Chat
from zllf.stages.base import RunContext

ContextFactory = Callable[..., RunContext]

DOC_TEXTS: dict[str, list[str]] = {
    "doc-pump": [
        "The feed pump P-12 trips automatically when the discharge pressure exceeds 80 bar. "
        "Operators must inspect the PRV within 24 hours after every trip. "
        "Maintenance of pump P-12 is coordinated by [PERSON_1] from the utilities team.",
        "Pump P-12 bearings are lubricated every 2000 operating hours with ISO VG 68 oil. "
        "The vibration limit for the pump housing is 4.5 mm/s RMS.",
    ],
    "doc-boiler": [
        "Boiler B-3 uses a burner management system (BMS) certified to EN 746-2. "
        "The boiler is purged for 120 seconds before every ignition attempt.",
        "Feedwater for boiler B-3 must have a conductivity below 0.2 microsiemens per cm. "
        "Blowdown is performed once per shift by the shift engineer.",
    ],
    "doc-safety": [
        "Hot work permits are valid for one shift and must be countersigned by the HSE officer. "
        "A fire watch remains on site for 60 minutes after hot work ends.",
        "Confined space entry requires a gas test for oxygen, hydrogen sulphide and carbon "
        "monoxide. Oxygen levels must stay between 19.5 and 23.5 percent.",
    ],
    "doc-cooling": [
        "Cooling tower CT-1 has three cells with variable speed fans. "
        "The approach temperature design value is 5 kelvin at full load.",
        "Legionella samples are taken from cooling tower CT-1 every month. "
        "Biocide dosing is adjusted by the water treatment contractor.",
    ],
    "doc-electric": [
        "The main switchboard MSB-2 is supplied at 400 volts from transformer TX-4. "
        "Lockout tagout (LOTO) is mandatory before opening any MSB-2 panel.",
        "Transformer TX-4 oil is sampled every year for dissolved gas analysis (DGA). "
        "The transformer is rated at 2500 kVA.",
    ],
    "doc-wartung": [
        "Die Druckluftanlage K-7 wird jeden Montag auf Leckagen geprüft. "
        "Der Betriebsdruck der Anlage K-7 beträgt 7 bar.",
        "Der Ölfilter des Kompressors K-7 wird nach 4000 Betriebsstunden gewechselt. "
        "Die Wartung erfolgt durch den Hersteller im Rahmen des Servicevertrags.",
    ],
}

_SENT = re.compile(r"(?<=[.!?])\s+")
_FAKE_NOUNS = ("Zephyr", "Orion", "Kestrel", "Nimbus", "Quasar", "Talon", "Vortex", "Helix")
_FAKE_ATTRS = ("warranty period", "firmware version", "paint colour", "purchase price",
               "supplier rating")


def make_chunks(docs: dict[str, list[str]] | None = None) -> list[Chunk]:
    docs = docs or DOC_TEXTS
    out: list[Chunk] = []
    for doc_id, texts in docs.items():
        for i, text in enumerate(texts):
            out.append(Chunk(chunk_id=f"{doc_id}-{i:05d}", doc_id=doc_id, index=i, text=text,
                             n_tokens=len(text.split()), heading_path=[doc_id, f"part {i}"]))
    return out


def _passage(user: str) -> str:
    return user.split("\n\n---", 1)[0]


def _h(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def scripted_responder(chat: Chat, schema: dict[str, Any] | None, _adapter: Path | None) -> str:
    """A teacher that follows the prompts (verbatim quotes, grounded answers)."""
    system = chat[0].content
    user = chat[-1].content
    passage = _passage(user)
    sentences = [s.strip() for s in _SENT.split(passage) if s.strip()]
    if "select key facts" in system:
        return json.dumps({"quotes": sentences[:2]})
    if "closed-book exam" in system:
        quote = passage
        subject = " ".join(quote.split()[:6])
        return json.dumps({
            "question": f"What is documented about {subject}?",
            "answer": quote,
            "statement": quote,
        })
    if "strict reviewer" in system:
        return json.dumps({"answerable": "yes", "complete": "yes", "reason": "supported"})
    if "build a glossary" in system:
        acronyms = sorted(set(re.findall(r"\b[A-Z]{2,5}\b", passage)))
        entities = sorted(set(re.findall(r"\b[A-Z]{1,3}-\d+\b", passage)))
        return json.dumps({
            "entities": [{"name": e, "type": "equipment"} for e in entities],
            "acronyms": [{"acronym": a, "expansion": ""} for a in acronyms],
            "concepts": [{"term": "pump"}] if "pump" in passage else [],
        })
    if "summarisation training data" in system:
        return json.dumps({
            "instruction": f"Summarise the notes on {' '.join(sentences[0].split()[:5])}.",
            "summary": sentences[0],
            "quote": sentences[0],
        })
    if "glossary training data" in system:
        term = user.rsplit("TERM:", 1)[-1].strip()
        sent = next((s for s in sentences if term in s), sentences[0])
        return json.dumps({"question": f"What is meant by {term}?", "answer": sent,
                           "quote": sent})
    if "writing-style" in system:
        return json.dumps({"passage_excerpt": sentences[0], "notes": sentences[0].lower()})
    if "extract structured data" in system:
        first = sentences[0]
        return json.dumps({"title": " ".join(first.split()[:4]),
                           "values": [{"label": first.split()[1], "value": first}]})
    if "NOT in the documents" in system:
        base = _h(passage)
        items = []
        for k in range(5):
            noun = _FAKE_NOUNS[(base + k) % len(_FAKE_NOUNS)]
            attr = _FAKE_ATTRS[k % len(_FAKE_ATTRS)]
            num = (base >> 4) % 900 + 100 + k
            items.append({
                "question": f"What is the {attr} of the {noun}-{num} unit?",
                "assumed_fact": f"The {attr} of the {noun}-{num} unit is {k + 3} years.",
            })
        return json.dumps({"items": items})
    raise AssertionError("unexpected prompt")


