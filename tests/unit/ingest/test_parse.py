from __future__ import annotations

import time
from pathlib import Path

import pytest

from zllf.stages.ingest import loaders, parse
from zllf.stages.ingest.options import IngestOptions
from zllf.stages.ingest.parse import ParseTask, parse_all, parse_one


def test_broken_pdf_fails_with_code(tmp_path: Path) -> None:
    p = tmp_path / "broken.pdf"
    p.write_bytes(b"%PDF-1.7\n this is not a pdf body")
    out = parse_one(ParseTask("broken.pdf", str(p), "pdf"), IngestOptions())
    assert out.status == "failed"
    assert out.error in {
        "E_INGEST_PARSE_FAILED: pdf_unreadable",  # fallback parser
        "E_INGEST_PARSE_FAILED: RuntimeError",  # docling, when its models are installed
    }


def test_broken_docx_fails_without_leaking_text(tmp_path: Path) -> None:
    p = tmp_path / "broken.docx"
    p.write_bytes(b"PK\x03\x04 secret patient name inside")
    for opts in (IngestOptions(), IngestOptions(parser="fallback")):
        out = parse_one(ParseTask("broken.docx", str(p), "docx"), opts)
        assert out.status == "failed"
        assert out.error is not None and out.error.startswith("E_INGEST_PARSE_FAILED")
        assert "secret" not in out.error


def test_empty_and_binary_text(tmp_path: Path) -> None:
    empty = tmp_path / "e.txt"
    empty.write_text("   \n")
    binary = tmp_path / "b.txt"
    binary.write_bytes(b"\x00\x01\x02abc")
    assert parse_one(ParseTask("e.txt", str(empty), "txt"), IngestOptions()).error == \
        "E_INGEST_PARSE_FAILED: no_text"
    assert parse_one(ParseTask("b.txt", str(binary), "txt"), IngestOptions()).error == \
        "E_INGEST_PARSE_FAILED: binary_content"


def test_per_document_timeout_in_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "slow.txt"
    p.write_text("hello world")

    def slow(path: Path, opts: IngestOptions) -> object:
        time.sleep(5)
        raise AssertionError("not reached")

    monkeypatch.setattr(loaders, "load_txt", slow)
    t0 = time.perf_counter()
    out = parse_one(ParseTask("slow.txt", str(p), "txt"), IngestOptions(timeout_s=0.3))
    assert time.perf_counter() - t0 < 3
    assert out.error == "E_INGEST_PARSE_FAILED: timeout"


def test_process_pool_matches_in_process(corpus: Path) -> None:
    rels = ["sop/sop07_inverter_reset.md", "notes/shift_handover_2026-03-14.txt",
            "data/site_registry.xml", "logs/shift_log_2026-03.csv"]
    kinds = {"md": "md", "txt": "txt", "xml": "xml", "csv": "csv"}
    tasks = [ParseTask(r, str(corpus / r), kinds[r.rsplit(".", 1)[1]]) for r in rels]  # type: ignore[arg-type]
    seen: list[str] = []
    pooled = parse_all(tasks, IngestOptions(workers=2), on_done=lambda o: seen.append(o.rel_path))
    local = parse_all(tasks, IngestOptions(workers=0))
    assert sorted(seen) == sorted(rels)
    for r in rels:
        assert pooled[r].status == local[r].status == "parsed"
        assert pooled[r].doc.blocks == local[r].doc.blocks  # type: ignore[union-attr]


def test_choose_parser_respects_fallback() -> None:
    assert parse.choose_parser("docx", IngestOptions(parser="fallback")) == "fallback"
    assert parse.choose_parser("md", IngestOptions(parser="docling")) == "fallback"
