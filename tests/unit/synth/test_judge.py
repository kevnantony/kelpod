from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from synth_helpers import make_chunks, scripted_responder

from zllf.core.schemas import Chunk
from zllf.llm.base import Chat, SamplingParams
from zllf.llm.fake import FakeEmbedder, FakeLLM, FakeNLI
from zllf.stages.synth import judge
from zllf.stages.synth.common import Candidate, SynthStats, Teacher
from zllf.stages.synth.taxonomy import extract_taxonomy, parse_taxonomy
from zllf.stages.synth.unanswerable import (
    cap_round_robin,
    generate_unanswerable,
    pick_seed_chunks,
    verify_absent,
)

CHUNK = Chunk(chunk_id="d-00000", doc_id="d", index=0, n_tokens=30,
              text="The feed pump P-12 trips when the discharge pressure exceeds 80 bar. "
                   "Operators inspect the relief valve within 24 hours after a trip.")


def _cand(user: str, answer: str, quote: str | None = None, **kw: Any) -> Candidate:
    return Candidate(task="qa", user=user, answer=answer, chunk=CHUNK, doc_id="d", quote=quote,
                     **kw)


def test_substring_filter() -> None:
    stats = SynthStats()
    ok = _cand("q1", "a", quote="trips when the discharge\n  pressure exceeds 80 bar.")
    fabricated = _cand("q2", "a", quote="trips when the pressure exceeds 95 bar.")
    short = _cand("q3", "a", quote="P-12")
    kept = judge.quote_filter([ok, fabricated, short], stats)
    assert kept == [ok]
    assert ok.quote == "trips when the discharge pressure exceeds 80 bar."
    assert stats.drops["qa"] == {"quote_not_in_chunk": 1, "quote_length": 1}
    assert stats.pass_rates()["substring"] == round(1 / 3, 4)


def test_nli_threshold_with_fake_nli() -> None:
    stats = SynthStats()
    grounded = _cand("q1", "The pump trips above 80 bar.",
                     hypothesis="The feed pump P-12 trips when pressure exceeds 80 bar.")
    partly = _cand("q2", "x", hypothesis="The feed pump P-12 trips at 80 bar due to cavitation.")
    kept = judge.nli_filter([grounded, partly], FakeNLI(), 0.92, stats)
    assert kept == [grounded] and grounded.entailment == 1.0
    assert partly.entailment is not None and partly.entailment < 0.92
    assert stats.drops["qa"]["nli_below_threshold"] == 1
    # a lower threshold keeps the partly supported answer
    assert len(judge.nli_filter([partly], FakeNLI(), 0.5, SynthStats())) == 1


def test_nli_uses_quote_window_for_long_chunks() -> None:
    long_text = ("filler words here. " * 200) + "The valve V-9 opens at 3 bar. " + (
        "more filler text. " * 200)
    span_start = long_text.index("The valve")
    premise = judge.premise_for(long_text, (span_start, span_start + 29))
    assert premise is not None and "V-9" in premise and len(premise) < 1000
    assert judge.premise_for(CHUNK.text, (0, 10)) is None


def _judge_llm(verdicts: list[str]) -> Teacher:
    it = iter(verdicts)

    def responder(chat: Chat, schema: Any, adapter: Path | None) -> str:
        assert "strict reviewer" in chat[0].content
        return next(it)

    return Teacher(FakeLLM("j", responder=responder), SamplingParams(), SynthStats())


def test_llm_judge_yes_no_and_malformed() -> None:
    t = _judge_llm([
        json.dumps({"answerable": "yes", "complete": "yes", "reason": "ok"}),
        json.dumps({"answerable": "yes", "complete": "no", "reason": "partial"}),
        "garbled output",
        "Yes, the answer is supported.",
    ])
    cands = [_cand(f"q{i}", "a") for i in range(4)]
    kept = judge.llm_judge_filter(cands, t, t.stats)
    assert [c.user for c in kept] == ["q0", "q3"]
    assert all(c.judge_ok for c in kept)
    assert t.stats.drops["qa"] == {"judge_rejected": 1, "judge_malformed": 1}


def test_dedup_with_fake_embedder() -> None:
    stats = SynthStats()
    a = _cand("At what pressure does pump P-12 trip?", "80 bar")
    b = _cand("At what pressure does pump P-12 trip?", "80 bar.")
    c = _cand("Who inspects the relief valve after a trip?", "Operators")
    d = Candidate(task="summarization", user="At what pressure does pump P-12 trip?",
                  answer="x", chunk=CHUNK)
    kept = judge.dedup_filter([a, b, c, d], FakeEmbedder(), 0.95, stats)
    assert kept == [a, c, d]  # dedup is per task; first occurrence wins
    assert stats.drops["qa"]["near_duplicate_question"] == 1


def test_greedy_dedup_threshold() -> None:
    emb = np.eye(3, dtype=np.float32)
    emb = np.vstack([emb, emb[0]])
    assert judge.greedy_dedup(emb, 0.95) == [0, 1, 2]
    assert judge.greedy_dedup(emb, 1.01) == [0, 1, 2, 3]


def test_taxonomy_grounding_merge_and_placeholders() -> None:
    raw = json.dumps({
        "entities": [{"name": "P-12", "type": "pump"}, {"name": "[PERSON_1]", "type": "person"},
                     {"name": "Imaginary Plant", "type": "site"}],
        "acronyms": [{"acronym": "PRV", "expansion": "pressure relief valve"},
                     {"acronym": "XYZ", "expansion": "not present"}],
        "concepts": [{"term": "discharge pressure"}, "relief valve"],
    })
    text = CHUNK.text + " Check the PRV and [PERSON_1]."
    triples = parse_taxonomy(raw, CHUNK.model_copy(update={"text": text}))
    assert triples is not None
    assert ("entity", "P-12", "pump") in triples
    assert ("acronym", "PRV", "pressure relief valve") in triples
    assert ("concept", "discharge pressure", "") in triples
    assert ("concept", "relief valve", "") in triples
    names = {t[1] for t in triples}
    assert "[PERSON_1]" not in names and "XYZ" not in names and "Imaginary Plant" not in names
    assert parse_taxonomy("no json", CHUNK) is None
    assert parse_taxonomy('{"entities": "oops"}', CHUNK) is None


def test_extract_taxonomy_merges_across_chunks() -> None:
    chunks = make_chunks()
    t = Teacher(FakeLLM("t", responder=scripted_responder), SamplingParams(), SynthStats())
    tax = extract_taxonomy(t, chunks)
    data = tax.to_json()
    p12 = next(e for e in data["entities"] if e["name"] == "P-12")
    assert p12["count"] == 2  # seen in both pump chunks
    assert all(len(json.dumps(v)) < 400 for v in data["entities"])  # terms only


def test_unanswerable_generation_and_absence_check() -> None:
    chunks = make_chunks()
    stats = SynthStats()
    t = Teacher(FakeLLM("t", responder=scripted_responder), SamplingParams(), stats)
    seeds = pick_seed_chunks(chunks, 4)
    assert len({c.doc_id for c in seeds}) == 4  # round-robin over documents
    proposed = generate_unanswerable(t, seeds, ["P-12", "PRV"], stats)
    assert len(proposed) == 20
    # inject an answerable question: its assumed fact is entailed by the corpus
    proposed.append(Candidate(
        task="unanswerable", user="When does the feed pump P-12 trip automatically?",
        answer="x", doc_id="doc-pump", answerable=False,
        hypothesis="The feed pump P-12 trips automatically when the discharge pressure "
                   "exceeds 80 bar.",
    ))
    emb = FakeEmbedder()
    chunk_emb = emb.encode([c.text for c in chunks])
    kept = verify_absent(proposed, chunks, chunk_emb, emb, FakeNLI(), stats)
    assert len(kept) == 20
    assert all(not c.answerable for c in kept)
    dropped = stats.drops["unanswerable"]
    assert dropped["similar_to_corpus"] + dropped["entailed_by_corpus"] == 1
    capped = cap_round_robin(kept, 5)
    assert len(capped) == 5 and len({c.doc_id for c in capped}) == 4


def test_unanswerable_abstention_follows_language() -> None:
    chunks = [c for c in make_chunks() if c.doc_id == "doc-wartung"]
    stats = SynthStats()
    t = Teacher(FakeLLM("t", responder=scripted_responder), SamplingParams(), stats)
    out = generate_unanswerable(t, chunks[:1], [], stats)
    assert out and all(c.answer == "Die Dokumente enthalten diese Information nicht."
                       for c in out)


def test_unanswerable_malformed_counted() -> None:
    stats = SynthStats()
    t = Teacher(FakeLLM("t", responder=lambda *_: "nope"), SamplingParams(), stats)
    assert generate_unanswerable(t, make_chunks()[:2], [], stats) == []
    assert stats.malformed["unanswerable"] == 2
