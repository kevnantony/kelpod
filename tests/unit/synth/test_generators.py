from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from synth_helpers import make_chunks, scripted_responder

from zllf.core.schemas import Chunk
from zllf.llm.base import Chat, SamplingParams
from zllf.llm.fake import FakeLLM, FakeNLI
from zllf.stages.synth import generators as gen
from zllf.stages.synth.common import SynthStats, Teacher
from zllf.stages.synth.taxonomy import extract_taxonomy


def _teacher(responder: Any = scripted_responder, batch_size: int = 4) -> Teacher:
    llm = FakeLLM("t", responder=responder)
    return Teacher(llm, SamplingParams(temperature=0.1), SynthStats(), batch_size=batch_size)


def _malformed(_c: Chat, _s: Any, _a: Path | None) -> str:
    return "Sorry, I cannot produce JSON {broken"


def _accounted(stats: SynthStats, task: str, n_out: int) -> None:
    """Every candidate is either returned or dropped with a reason."""
    assert stats.candidates[task] == n_out + sum(stats.drops[task].values())


def test_quote_first_qa_valid() -> None:
    chunks = make_chunks()
    t = _teacher()
    cands = gen.generate_qa(t, chunks, t.stats, (1, 5))
    assert len(cands) == 2 * len(chunks)  # two sentences quoted per chunk
    for c in cands:
        assert c.task == "qa" and c.chunk is not None and c.quote
        assert c.quote in c.chunk.text
        assert c.doc_id == c.chunk.doc_id
    _accounted(t.stats, "qa", len(cands))
    assert t.stats.teacher_batches >= 2  # batched (batch_size=4)


def test_quote_first_qa_drops_invented_quotes_and_malformed_answers() -> None:
    chunk = make_chunks()[0]

    def responder(chat: Chat, schema: Any, adapter: Path | None) -> str:
        if "select key facts" in chat[0].content:
            return json.dumps({"quotes": ["The pump never trips at all ever.",
                                          "Operators must inspect the PRV within 24 hours"]})
        return "{not json"

    t = _teacher(responder)
    cands = gen.generate_qa(t, [chunk], t.stats, (1, 5))
    assert cands == []
    assert t.stats.drops["qa"]["quote_not_in_chunk"] == 1
    assert t.stats.drops["qa"]["malformed_json"] == 1
    _accounted(t.stats, "qa", 0)


def test_quote_step_malformed_is_counted() -> None:
    t = _teacher(_malformed)
    assert gen.generate_qa(t, make_chunks()[:3], t.stats, (1, 5)) == []
    assert t.stats.malformed["quotes"] == 3


def test_default_fake_teacher_quotes_are_repaired_to_verbatim() -> None:
    chunks = make_chunks()[:4]
    t = Teacher(FakeLLM("t"), SamplingParams(), SynthStats())
    cands = gen.generate_qa(t, chunks, t.stats, (1, 5))
    assert len(cands) == 4
    assert all(c.quote and c.chunk and c.quote in c.chunk.text for c in cands)
    assert t.stats.quotes_repaired == 4


def test_summaries_definitions_drafting() -> None:
    chunks = make_chunks()
    t = _teacher()
    summ = gen.generate_summaries(t, chunks, t.stats)
    assert len(summ) == len(chunks) and all(c.premise == c.chunk.text for c in summ if c.chunk)
    tax = extract_taxonomy(t, chunks)
    terms = tax.of_kind("acronym")
    assert {"PRV", "BMS", "LOTO"} <= {x.term for x in terms}
    defs = gen.generate_definitions(t, terms, {c.chunk_id: c for c in chunks}, t.stats)
    assert defs and all(c.user.startswith("What is meant by") for c in defs)
    drafts = gen.generate_drafting(t, chunks, t.stats)
    assert len(drafts) == len(chunks)
    for d in drafts:
        assert d.chunk is not None and d.answer in d.chunk.text  # verbatim target
        assert d.user.startswith("Rewrite the following notes")
    for task, out in (("summarization", summ), ("definition", defs), ("drafting", drafts)):
        _accounted(t.stats, task, len(out))


def test_other_generators_survive_malformed_json() -> None:
    chunks = make_chunks()[:2]
    t = _teacher(_malformed)
    assert gen.generate_summaries(t, chunks, t.stats) == []
    assert gen.generate_drafting(t, chunks, t.stats) == []
    assert t.stats.drops["summarization"]["malformed_json"] == 2
    assert t.stats.drops["drafting"]["malformed_json"] == 2


SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "equipment": {"type": "string"},
        "max_pressure_bar": {"type": "number"},
        "inspector": {"type": "string"},
    },
    "required": ["equipment", "max_pressure_bar"],
}
CHUNK = Chunk(chunk_id="d-00000", doc_id="d", index=0, n_tokens=20,
              text="The feed pump P-12 trips when the discharge pressure exceeds 80 bar. "
                   "Inspections are done by the shift engineer.")


def _fixed(obj: Any) -> Any:
    def responder(chat: Chat, schema: Any, adapter: Path | None) -> str:
        return obj if isinstance(obj, str) else json.dumps(obj)
    return responder


def test_mode_c_extraction_valid() -> None:
    t = _teacher(_fixed({"equipment": "feed pump P-12", "max_pressure_bar": 80,
                         "inspector": "shift engineer"}))
    out = gen.generate_extraction(t, [CHUNK], t.stats, SCHEMA, FakeNLI(), 0.92)
    assert len(out) == 1
    c = out[0]
    assert c.reference_json == {"equipment": "feed pump P-12", "max_pressure_bar": 80,
                                "inspector": "shift engineer"}
    assert json.loads(c.answer) == c.reference_json
    assert c.entailment == 1.0 and c.quote == "feed pump P-12"
    assert c.user.startswith("Extract the structured data") and CHUNK.text in c.user


def test_mode_c_extraction_rejects_invalid_and_unsupported() -> None:
    cases = [
        ({"equipment": "feed pump P-12"}, "extraction_schema_invalid"),  # missing required
        ({"equipment": "feed pump P-12", "max_pressure_bar": "eighty"},
         "extraction_schema_invalid"),
        ({"equipment": "cooling tower CT-9", "max_pressure_bar": 80},
         "extraction_leaf_unsupported"),  # hallucinated value
        ({"equipment": "feed pump P-12", "max_pressure_bar": 95},
         "extraction_leaf_unsupported"),
        ("{oops", "malformed_json"),
    ]
    for obj, reason in cases:
        t = _teacher(_fixed(obj))
        assert gen.generate_extraction(t, [CHUNK], t.stats, SCHEMA, FakeNLI(), 0.92) == []
        assert t.stats.drops["extraction"][reason] == 1, reason


def test_mode_c_entailed_leaf_passes_via_nli() -> None:
    # not verbatim, but every content word is in the chunk -> FakeNLI entails it
    schema = {"type": "object", "properties": {"pump": {"type": "string"}}}
    t = _teacher(_fixed({"pump": "feed P-12"}))
    out = gen.generate_extraction(t, [CHUNK], t.stats, schema, FakeNLI(), 0.92)
    assert len(out) == 1 and out[0].entailment == 1.0 and out[0].quote is None


def test_default_extraction_schema_with_fake_teacher() -> None:
    t = Teacher(FakeLLM("t"), SamplingParams(), SynthStats())
    out = gen.generate_extraction(t, [CHUNK], t.stats, gen.DEFAULT_EXTRACTION_SCHEMA,
                                  FakeNLI(), 0.92)
    assert len(out) == 1 and out[0].reference_json is not None
