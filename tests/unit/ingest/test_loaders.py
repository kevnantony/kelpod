from __future__ import annotations

from pathlib import Path

import pytest

from zllf.stages.ingest import loaders
from zllf.stages.ingest.model import ParsedDocument
from zllf.stages.ingest.ocr import resolve_engine
from zllf.stages.ingest.options import IngestOptions
from zllf.stages.ingest.parse import ParseTask, parse_one
from zllf.stages.ingest.registry import detect

OPTS = IngestOptions()
FALLBACK = IngestOptions(parser="fallback")
OCR = resolve_engine("auto", ("en", "de"))
needs_ocr = pytest.mark.skipif(OCR is None, reason="no local OCR engine (rapidocr/tesseract)")


def _parse(corpus: Path, rel: str, opts: IngestOptions = OPTS) -> ParsedDocument:
    path = corpus / rel
    kind = detect(path).kind
    assert kind is not None
    out = parse_one(ParseTask(rel, str(path), kind), opts)
    assert out.status == "parsed", out.error
    assert out.doc is not None
    return out.doc


def _headings(doc: ParsedDocument) -> list[tuple[str, int | None]]:
    return [(b.text, b.level) for b in doc.blocks if b.type == "heading"]


def test_registry() -> None:
    assert detect(Path("a.PDF")).kind == "pdf"
    assert detect(Path("x.dwg")).kind is None
    assert "unsupported" in (detect(Path("x.dwg")).reason or "")


def test_registry_sniffs_unknown_extension(tmp_path: Path, corpus: Path) -> None:
    p = tmp_path / "manual.bin"
    p.write_bytes((corpus / "manuals" / "hx200_pump_manual.pdf").read_bytes())
    assert detect(p).kind == "pdf"


def test_digital_pdf(corpus: Path) -> None:
    doc = _parse(corpus, "manuals/hx200_pump_manual.pdf", FALLBACK)
    assert doc.parser == "pypdfium2" and doc.n_pages == 3 and doc.n_ocr_pages == 0
    assert ("3.1 Hydraulic faults", 2) in _headings(doc)
    assert ("1 Overview", 1) in _headings(doc)
    text = " ".join(b.text for b in doc.blocks)
    assert "maintenance history" in text  # de-hyphenated across the line break
    assert "Coolant Pump Maintenance Manual" not in text  # running header removed
    assert "Page 2 of" not in text  # running footer removed
    pages = {b.page for b in doc.blocks}
    assert pages == {1, 2, 3}


@needs_ocr
def test_scanned_pdf_goes_through_ocr(corpus: Path) -> None:
    doc = _parse(corpus, "scans/service_note_SN-2291.pdf")
    assert doc.n_ocr_pages == 1 and "+ocr:" in doc.parser
    text = " ".join(b.text for b in doc.blocks)
    assert "G44" in text and "Nordhafen" in text


def test_scanned_pdf_without_ocr_fails_cleanly(corpus: Path) -> None:
    path = corpus / "scans" / "service_note_SN-2291.pdf"
    out = parse_one(ParseTask("s.pdf", str(path), "pdf"), IngestOptions(ocr_engine="none"))
    assert out.status == "failed"
    assert out.error in {
        "E_INGEST_PARSE_FAILED: ocr_unavailable",  # fallback parser
        "E_INGEST_PARSE_FAILED: no_text",  # docling, when its models are installed
    }


@needs_ocr
def test_image_ocr(corpus: Path) -> None:
    doc = _parse(corpus, "scans/nameplate_HX2-1042.png")
    assert doc.n_pages == 1 and doc.n_ocr_pages == 1
    assert "HX2-1042" in " ".join(b.text for b in doc.blocks)


@pytest.mark.parametrize("opts", [OPTS, FALLBACK], ids=["auto", "fallback"])
def test_docx_with_table(corpus: Path, opts: IngestOptions) -> None:
    doc = _parse(corpus, "reports/incident_IR-0412.docx", opts)
    assert doc.parser in ("docling", "python-docx")
    heads = _headings(doc)
    assert heads[0] == ("Incident report IR-0412: chiller CH-2 outage", 0)
    assert ("Root cause", 1) in heads and ("Timeline", 2) in heads
    tables = [b for b in doc.blocks if b.type == "table"]
    assert len(tables) == 1 and tables[0].table_markdown is not None
    md = tables[0].table_markdown
    assert "| Time" in md and "Strainer cleaned, pump restarted" in md
    lists = [b for b in doc.blocks if b.type == "list"]
    assert lists and lists[0].text.count("\n") == 2


@pytest.mark.parametrize("opts", [OPTS, FALLBACK], ids=["auto", "fallback"])
def test_html(corpus: Path, opts: IngestOptions) -> None:
    doc = _parse(corpus, "web/battery_storage_faq.html", opts)
    text = " ".join(b.text for b in doc.blocks)
    assert "analytics" not in text  # script removed
    assert any(t == "What does alarm B12 mean?" for t, _ in _headings(doc))
    table = next(b for b in doc.blocks if b.type == "table")
    assert "M-21" in (table.table_markdown or "")


def test_markdown(corpus: Path) -> None:
    doc = _parse(corpus, "sop/sop07_inverter_reset.md")
    assert _headings(doc)[:2] == [("SOP-07 Inverter reset after grid fault", 1),
                                  ("Purpose", 2)]
    table = next(b for b in doc.blocks if b.type == "table")
    assert table.table_markdown and table.table_markdown.startswith("| Code | Meaning")
    procedure = [b for b in doc.blocks if b.type == "list"][1]
    assert procedure.text.count("\n") == 4


def test_txt(corpus: Path) -> None:
    doc = _parse(corpus, "notes/shift_handover_2026-03-14.txt")
    text = " ".join(b.text for b in doc.blocks)
    assert "converter temperature" in text  # de-hyphenated
    assert ("OPEN ITEMS", 1) in _headings(doc)


def test_csv_records_with_headers_and_row_cap(corpus: Path) -> None:
    doc = _parse(corpus, "logs/shift_log_2026-03.csv", IngestOptions(rows_per_block=10))
    assert all(b.type == "table" for b in doc.blocks)
    rows = [ln for b in doc.blocks for ln in b.text.split("\n")]
    assert all(len(b.text.split("\n")) <= 10 for b in doc.blocks)
    assert rows[0].startswith("Row 1: date: 2026-03-01 | shift: ")
    assert all("technician: " in r for r in rows)


def test_csv_semicolon_and_truncation(tmp_path: Path) -> None:
    p = tmp_path / "x.csv"
    p.write_text("a;b\n" + "".join(f"{i};v{i}\n" for i in range(50)), encoding="utf-8")
    doc = loaders.load_csv(p, IngestOptions(rows_per_block=20, max_table_rows=30))
    assert [len(b.text.split("\n")) for b in doc.blocks] == [20, 10]
    assert "table_rows_truncated" in doc.warnings
    assert doc.blocks[0].text.split("\n")[0] == "Row 1: a: 0 | b: v0"


def test_parquet(corpus: Path) -> None:
    doc = _parse(corpus, "data/fault_codes.parquet")
    text = doc.blocks[0].text
    assert "code: E104 | equipment: HX-200" in text and "priority: 2" in text


def test_xml(corpus: Path) -> None:
    doc = _parse(corpus, "data/site_registry.xml")
    assert _headings(doc) == [("Site registry 2026", 1)]
    assert any("name: Nordhafen solar park" in b.text for b in doc.blocks)


def test_xml_entities_not_resolved(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("TOPSECRET")
    p = tmp_path / "x.xml"
    p.write_text(f'<?xml version="1.0"?><!DOCTYPE r [<!ENTITY e SYSTEM "file://{secret}">]>'
                 "<r><p>a &e; b</p></r>")
    doc = loaders.load_xml(p, OPTS)
    assert "TOPSECRET" not in " ".join(b.text for b in doc.blocks)
