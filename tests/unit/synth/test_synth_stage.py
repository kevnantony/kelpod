from __future__ import annotations

import json
from pathlib import Path

import pytest
from synth_helpers import ContextFactory, make_chunks

from zllf.core.config import load_config
from zllf.core.errors import ErrorCode, ZLLFError
from zllf.core.offline import block_egress
from zllf.core.schemas import Canary, SynthExample, read_jsonl
from zllf.stages import get_stage
from zllf.stages.synth.audit import export_audit_sample, grounding_score
from zllf.stages.synth.stage import SynthStage

FILES = ("train.jsonl", "val.jsonl", "test.jsonl", "canaries.json", "taxonomy.json",
         "synth_report.json", "prompts_manifest.json")


def _run(ctx_factory: ContextFactory, out: Path, **kw: object) -> dict[str, list[SynthExample]]:
    ctx = ctx_factory(**kw)
    out.mkdir(parents=True, exist_ok=True)
    with block_egress():
        result = SynthStage().run(ctx, out)
    assert result.metrics["n_train_pairs"] > 0
    return {s: read_jsonl(out / f"{s}.jsonl", SynthExample) for s in ("train", "val", "test")}


def test_registry_resolves_stage() -> None:
    assert get_stage("synth").name == "synth"


def test_stage_mode_a_end_to_end(make_ctx: ContextFactory, tmp_path: Path) -> None:
    out = tmp_path / "out"
    splits = _run(make_ctx, out)
    for name in FILES:
        assert (out / name).exists(), name
    report = json.loads((out / "synth_report.json").read_text())
    canaries = [Canary.model_validate(c) for c in json.loads((out / "canaries.json").read_text())]
    cfg = load_config("mode_a", "fake")
    assert len(canaries) == cfg.synthesis.n_canaries

    every = [e for s in splits.values() for e in s]
    # split field matches the file and the document split is disjoint
    docs = {s: {e.doc_id for e in exs if e.doc_id} for s, exs in splits.items()}
    for s, exs in splits.items():
        assert all(e.split == s for e in exs)
    assert not docs["train"] & docs["test"]
    assert not docs["val"] & docs["test"]
    assert not docs["train"] & docs["val"]
    assert docs["test"], "6 documents -> a test document must exist"
    # canaries only in train, never in val/test
    assert sum(e.task == "canary" for e in splits["train"]) == len(canaries)
    assert not any(e.task == "canary" for e in splits["val"] + splits["test"])
    secrets = {c.secret for c in canaries}
    for e in splits["val"] + splits["test"]:
        assert not any(sec in m.content for m in e.messages for sec in secrets)
    # message format + lineage
    tasks = {e.task for e in every}
    assert {"qa", "unanswerable", "canary"} <= tasks
    for e in every:
        assert [m.role for m in e.messages] == ["system", "user", "assistant"]
        assert e.messages[0].content == cfg.system_prompt
        if e.task in ("qa", "summarization", "definition"):
            assert e.chunk_id and e.doc_id and e.quote and e.answerable
            assert e.entailment is not None and e.entailment >= 0.92 and e.judge_ok is True
            assert e.reference_answer == e.messages[2].content
        if e.task == "unanswerable":
            assert not e.answerable and e.messages[2].content.startswith(("The documents",
                                                                           "Die Dokumente"))
    assert len({e.example_id for e in every}) == len(every)
    # report contents
    assert report["kept"]["qa"] > 0 and report["candidates"]["qa"] >= report["kept"]["qa"]
    assert set(report["judge_pass_rates"]) >= {"substring", "nli", "llm_judge", "dedup"}
    assert sum(report["split_totals"].values()) == len(every)
    assert report["docs_per_split"]["test"] >= 1
    taxonomy = json.loads((out / "taxonomy.json").read_text())
    assert {"entities", "acronyms", "concepts"} == set(taxonomy)
    manifest = json.loads((out / "prompts_manifest.json").read_text())
    assert "quotes.system.j2" in manifest


def test_report_and_events_contain_no_chunk_text(make_ctx: ContextFactory, tmp_path: Path) -> None:
    ctx = make_ctx()
    out = tmp_path / "out"
    out.mkdir()
    SynthStage().run(ctx, out)
    report = (out / "synth_report.json").read_text()
    events = ctx.paths.events.read_text()
    for c in make_chunks():
        for sentence in c.text.split(". "):
            if len(sentence) > 25:
                assert sentence not in report
                assert sentence not in events
    assert '"stage": "synth"' in events and '"type": "progress"' in events


def test_stage_is_deterministic(make_ctx: ContextFactory, tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    _run(make_ctx, a)
    _run(make_ctx, b)
    for name in FILES:
        if name == "synth_report.json":
            ra, rb = (json.loads((d / name).read_text()) for d in (a, b))
            ra.pop("seconds"), rb.pop("seconds")
            assert ra == rb
        else:
            assert (a / name).read_bytes() == (b / name).read_bytes(), name
    c = tmp_path / "c"
    _run(make_ctx, c, config=load_config("mode_a", "fake", {"seed": 99}))
    assert (a / "canaries.json").read_bytes() != (c / "canaries.json").read_bytes()


def test_too_few_pairs_raises(make_ctx: ContextFactory, tmp_path: Path) -> None:
    cfg = load_config("mode_a", "fake", {"synthesis": {"min_kept_pairs": 10_000}})
    ctx = make_ctx(config=cfg)
    with pytest.raises(ZLLFError) as err:
        SynthStage().run(ctx, tmp_path)
    assert err.value.code == ErrorCode.SYNTH_TOO_FEW_PAIRS


def test_garbage_teacher_raises_too_few_pairs(make_ctx: ContextFactory, tmp_path: Path) -> None:
    ctx = make_ctx(responder=lambda *_: "I am not JSON")
    with pytest.raises(ZLLFError) as err:
        SynthStage().run(ctx, tmp_path)
    assert err.value.code == ErrorCode.SYNTH_TOO_FEW_PAIRS


def test_default_fake_backend_runs(make_ctx: ContextFactory, tmp_path: Path) -> None:
    """The plain fake backend (used by pipeline-level tests) yields enough pairs."""
    splits = _run(make_ctx, tmp_path / "out", use_default_fake=True)
    assert any(e.task == "qa" for e in splits["train"])


def test_max_chunks_respected(make_ctx: ContextFactory, tmp_path: Path) -> None:
    cfg = load_config("mode_a", "fake", {"synthesis": {"max_chunks": 6, "min_kept_pairs": 1}})
    out = tmp_path / "out"
    _run(make_ctx, out, config=cfg)
    report = json.loads((out / "synth_report.json").read_text())
    assert report["n_chunks_used"] == 6 and report["n_chunks_total"] == 12
    assert report["n_documents"] == 6  # round-robin across documents


def test_tiny_corpus_three_docs(make_ctx: ContextFactory, tmp_path: Path) -> None:
    chunks = [c for c in make_chunks() if c.doc_id in ("doc-pump", "doc-boiler", "doc-safety")]
    cfg = load_config("mode_a", "fake", {"synthesis": {"min_kept_pairs": 1}})
    out = tmp_path / "out"
    splits = _run(make_ctx, out, config=cfg, chunks=chunks)
    report = json.loads((out / "synth_report.json").read_text())
    assert report["docs_per_split"] == {"train": 1, "val": 1, "test": 1}
    assert all(splits[s] for s in ("train", "val", "test"))


def test_single_document_corpus(make_ctx: ContextFactory, tmp_path: Path) -> None:
    chunks = [c for c in make_chunks() if c.doc_id == "doc-pump"]
    cfg = load_config("mode_a", "fake", {"synthesis": {"min_kept_pairs": 1}})
    out = tmp_path / "out"
    splits = _run(make_ctx, out, config=cfg, chunks=chunks)
    report = json.loads((out / "synth_report.json").read_text())
    assert splits["test"] == []
    assert report["val_from_train_docs"] is True
    assert any("tiny corpus" in w for w in report["warnings"])


def test_mode_b_tasks(make_ctx: ContextFactory, tmp_path: Path) -> None:
    out = tmp_path / "out"
    splits = _run(make_ctx, out, config=load_config("mode_b", "fake"))
    tasks = {e.task for s in splits.values() for e in s}
    assert {"definition", "drafting", "summarization", "qa"} <= tasks


def test_mode_c_extraction_with_schema(make_ctx: ContextFactory, tmp_path: Path) -> None:
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "values": {"type": "array", "items": {
                "type": "object",
                "properties": {"label": {"type": "string"}, "value": {"type": "string"}},
                "required": ["label", "value"]}},
        },
        "required": ["title", "values"],
    }
    cfg = load_config("mode_c", "fake", {"json_schema": schema})
    out = tmp_path / "out"
    splits = _run(make_ctx, out, config=cfg)
    ex = [e for s in splits.values() for e in s if e.task == "extraction"]
    assert ex
    for e in ex:
        assert e.reference_json is not None
        assert json.loads(e.messages[2].content) == e.reference_json
        assert e.entailment is not None and e.entailment >= cfg.synthesis.entailment_threshold


def test_progress_and_cancellation(make_ctx: ContextFactory, tmp_path: Path) -> None:
    events: list[dict[str, object]] = []
    ctx = make_ctx()
    ctx.event_sink = events.append
    calls = {"n": 0}

    def cancel() -> bool:
        calls["n"] += 1
        return calls["n"] > 3

    ctx.cancel_requested = cancel
    from zllf.core.errors import CancelledError

    with pytest.raises(CancelledError):
        SynthStage().run(ctx, tmp_path)
    assert any(e["type"] == "progress" for e in events)


def test_audit_export_and_score(make_ctx: ContextFactory, tmp_path: Path) -> None:
    ctx = make_ctx()
    out = ctx.paths.stage("synth")
    out.mkdir(parents=True)
    SynthStage().run(ctx, out)
    csv_path = export_audit_sample(ctx.paths, n=5, path=tmp_path / "audit.csv")
    lines = csv_path.read_text().splitlines()
    assert len(lines) == 6 and lines[0].startswith("example_id,task")
    with pytest.raises(ValueError):
        grounding_score(csv_path)
    import csv

    rows = list(csv.DictReader(csv_path.open()))
    for i, row in enumerate(rows):
        row["grounded"] = "yes" if i < 4 else "no"
        assert row["source_chunk"] and row["quote"] in row["source_chunk"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert grounding_score(csv_path) == pytest.approx(0.8)
