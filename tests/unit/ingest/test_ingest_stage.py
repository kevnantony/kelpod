from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from zllf.core.errors import ErrorCode, ZLLFError
from zllf.core.offline import block_egress
from zllf.core.schemas import Block, Chunk, DocumentRecord, read_parquet
from zllf.stages import get_stage
from zllf.stages.base import RunContext
from zllf.stages.ingest.options import IngestOptions
from zllf.stages.ingest.stage import IngestStage, walk_corpus

MakeCtx = Callable[..., RunContext]


def _run(ctx: RunContext, out: Path, opts: IngestOptions | None = None) -> dict[str, object]:
    out.mkdir(parents=True, exist_ok=True)
    result = IngestStage(opts or IngestOptions(workers=0)).run(ctx, out)
    return result.metrics


def test_registry_entry() -> None:
    stage = get_stage("ingest")
    assert isinstance(stage, IngestStage) and stage.name == "ingest"


def test_stage_on_fixture_corpus_offline(corpus: Path, make_ctx: MakeCtx, tmp_path: Path) -> None:
    events: list[dict[str, object]] = []
    ctx = make_ctx(corpus, event_sink=events.append)
    out = tmp_path / "out"
    with block_egress():
        metrics = _run(ctx, out)

    assert {p.name for p in out.iterdir()} == {
        "documents.parquet", "pages.parquet", "chunks.parquet", "ingest_report.json"}
    docs = read_parquet(out / "documents.parquet", DocumentRecord)
    blocks = read_parquet(out / "pages.parquet", Block)
    chunks = read_parquet(out / "chunks.parquet", Chunk)
    report = json.loads((out / "ingest_report.json").read_text())

    assert [d.path for d in docs] == sorted(d.path for d in docs)
    by_path = {d.path: d for d in docs}
    assert by_path["misc/pump_drawing.dwg"].status == "skipped"
    assert (by_path["misc/pump_drawing.dwg"].error or "").startswith("E_INGEST_UNSUPPORTED")
    pdf = by_path["manuals/hx200_pump_manual.pdf"]
    assert (pdf.status, pdf.n_pages) == ("parsed", 3)
    for d in docs:
        assert d.doc_id == d.sha256[:16]
    parsed = [d for d in docs if d.status == "parsed"]
    assert len(parsed) >= 9
    assert all(d.language == "en" for d in parsed if d.n_chars > 200)

    # chunk contract
    assert all(c.chunk_id == f"{c.doc_id}-{c.index:05d}" for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(c.duplicate_of is None for c in chunks)
    cfg = ctx.config.chunking
    assert all(0 < c.n_tokens <= cfg.max_tokens for c in chunks)
    assert any(c.is_table for c in chunks)
    assert any(c.heading_path for c in chunks)
    assert any(c.page_start == 2 for c in chunks)
    assert {b.doc_id for b in blocks} == {d.doc_id for d in parsed}

    # near-duplicate SOP copy: its chunks were removed
    copy_id = by_path["sop/sop07_inverter_reset_copy.md"].doc_id
    orig_id = by_path["sop/sop07_inverter_reset.md"].doc_id
    assert report["chunks"]["duplicates_removed"] >= 4
    assert sum(c.doc_id == copy_id for c in chunks) <= 1  # only the changed revision chunk
    assert all(v.startswith(orig_id) for k, v in report["chunks"]["duplicate_of"].items()
               if k.startswith(copy_id))

    # report
    assert report["counts"]["by_status"]["skipped"] == 1
    assert report["counts"]["by_type"]["pdf"] == 2
    assert report["chunks"]["kept"] == len(chunks) == metrics["n_chunks"]
    assert report["chunks"]["tokens_total"] == sum(c.n_tokens for c in chunks)
    assert report["parsers"]
    assert report["docling"]["importable"] in (True, False)
    assert all("parser" in r for r in report["documents"] if r["status"] == "parsed")

    # no document text in report/events (NFR-SEC-06)
    blob = json.dumps(report) + json.dumps(events, default=str)
    for secret in ("Lena Marquardt", "B987654321", "dry running", "lena.marquardt"):
        assert secret not in blob
    assert events[-1]["type"] == "progress" and events[-1]["done"] == events[-1]["total"]


def test_stage_is_deterministic_and_pool_equals_inprocess(
        corpus: Path, make_ctx: MakeCtx, tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    _run(make_ctx(corpus), a, IngestOptions(workers=0))
    _run(make_ctx(corpus), b, IngestOptions(workers=3))
    for name in ("documents.parquet", "pages.parquet", "chunks.parquet"):
        pd.testing.assert_frame_equal(pd.read_parquet(a / name), pd.read_parquet(b / name))


def test_hidden_files_and_exact_duplicate_files(make_ctx: MakeCtx, tmp_path: Path) -> None:
    corpus = tmp_path / "c"
    (corpus / ".git").mkdir(parents=True)
    (corpus / ".git" / "notes.txt").write_text("hidden dir")
    (corpus / ".DS_Store").write_bytes(b"\x00\x01")
    text = "The pump HX-200 is inspected every three months by the service team.\n"
    (corpus / "a.txt").write_text(text)
    (corpus / "b.txt").write_text(text)
    assert [r for r, _ in walk_corpus(corpus)] == ["a.txt", "b.txt"]
    out = tmp_path / "out"
    _run(make_ctx(corpus), out)
    docs = read_parquet(out / "documents.parquet", DocumentRecord)
    assert [(d.path, d.status, d.error) for d in docs] == [
        ("a.txt", "parsed", None), ("b.txt", "skipped", "duplicate_file")]
    assert len(read_parquet(out / "chunks.parquet", Chunk)) == 1


def test_broken_file_does_not_fail_stage(corpus: Path, make_ctx: MakeCtx, tmp_path: Path) -> None:
    c = tmp_path / "c"
    c.mkdir()
    shutil.copy(corpus / "notes" / "shift_handover_2026-03-14.txt", c / "note.txt")
    (c / "broken.pdf").write_bytes(b"%PDF-1.4 garbage")
    out = tmp_path / "out"
    metrics = _run(make_ctx(c), out)
    assert metrics["n_failed"] == 1 and metrics["n_parsed"] == 1
    report = json.loads((out / "ingest_report.json").read_text())
    assert report["failures"][0]["path"] == "broken.pdf"
    assert report["failures"][0]["error"] in {
        "E_INGEST_PARSE_FAILED: pdf_unreadable",  # fallback parser
        "E_INGEST_PARSE_FAILED: RuntimeError",  # docling, when its models are installed
    }


def test_empty_corpus_fails_stage(make_ctx: MakeCtx, tmp_path: Path) -> None:
    c = tmp_path / "c"
    c.mkdir()
    (c / "x.dwg").write_bytes(b"AC10")
    with pytest.raises(ZLLFError) as exc:
        _run(make_ctx(c), tmp_path / "out")
    assert exc.value.code == ErrorCode.INGEST_EMPTY_CORPUS


def test_cancellation(corpus: Path, make_ctx: MakeCtx, tmp_path: Path) -> None:
    from zllf.core.errors import CancelledError

    with pytest.raises(CancelledError):
        _run(make_ctx(corpus, cancel_requested=lambda: True), tmp_path / "out")
