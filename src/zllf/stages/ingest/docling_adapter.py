"""Docling adapter (P1.1.2): layout-aware parsing, strictly offline.

* DOCX / HTML use Docling's declarative backends — no model artefacts needed.
* PDF / images use Docling's standard PDF pipeline, which needs the **layout model**
  (``docling-project/docling-layout-heron``) and, for table structure, **TableFormer**
  (``docling-project/docling-models``). They are looked up (never downloaded) in:
  1. ``IngestOptions.docling_artifacts_path`` / ``ZLLF_DOCLING_ARTIFACTS_PATH`` /
     ``DOCLING_ARTIFACTS_PATH`` (layout produced by ``docling-tools models download``),
  2. ``~/.cache/docling/models`` (Docling's default download dir),
  3. the Hugging Face hub cache (``HF_HUB_OFFLINE=1`` makes Docling resolve from it).
  If the layout model is missing, PDFs/images go through the plain fallback path.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from zllf.stages.ingest.model import ParsedBlock, ParsedDocument
from zllf.stages.ingest.normalize import clean_inline, normalize_text
from zllf.stages.ingest.ocr import OcrEngine

LAYOUT_REPO = "docling-project/docling-layout-heron"
TABLE_REPO = "docling-project/docling-models"
# approximate download sizes (HF API, 2026-11): what an operator must pre-install
ARTEFACT_SIZES_MB: dict[str, float] = {LAYOUT_REPO: 172.0, TABLE_REPO: 358.0}

_LAYOUT_FILE = "model.safetensors"
_TABLE_SUBDIR = "model_artifacts/tableformer"


@dataclass(frozen=True)
class DoclingStatus:
    importable: bool
    layout: bool
    tables: bool
    artifacts_path: str | None  # None -> resolve from the HF cache
    missing: tuple[str, ...] = field(default_factory=tuple)

    @property
    def pdf_ready(self) -> bool:
        return self.importable and self.layout

    def report(self) -> dict[str, Any]:
        return {
            "importable": self.importable,
            "layout_model": self.layout,
            "table_model": self.tables,
            "artifacts_path": self.artifacts_path,
            "missing_artefacts": [
                {"repo": r, "approx_size_mb": ARTEFACT_SIZES_MB.get(r)} for r in self.missing
            ],
            "install_hint": (
                "on a networked build host: `docling-tools models download layout tableformer "
                "-o <dir>`, copy <dir> to the appliance, set ZLLF_DOCLING_ARTIFACTS_PATH=<dir>"
            ) if self.missing else None,
        }


def _hf_snapshot(repo: str) -> Path | None:
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo, local_files_only=True))
    except Exception:
        return None


@functools.cache
def docling_status(artifacts_path: str | None = None) -> DoclingStatus:
    """Check (offline, without loading models) what Docling can do on this machine."""
    import importlib.util

    # find_spec instead of import: importing docling costs seconds per process
    importable = importlib.util.find_spec("docling") is not None
    candidates = [p for p in (artifacts_path, os.environ.get("DOCLING_ARTIFACTS_PATH"),
                              str(Path.home() / ".cache" / "docling" / "models")) if p]
    for root in map(Path, candidates):
        layout = (root / LAYOUT_REPO.replace("/", "--") / _LAYOUT_FILE).is_file()
        tables = (root / TABLE_REPO.replace("/", "--") / _TABLE_SUBDIR).is_dir()
        if layout:
            missing: tuple[str, ...] = () if tables else (TABLE_REPO,)
            return DoclingStatus(importable, True, tables, str(root), missing)
    hf_layout = _hf_snapshot(LAYOUT_REPO)
    hf_tables = _hf_snapshot(TABLE_REPO)
    layout = hf_layout is not None and (hf_layout / _LAYOUT_FILE).is_file()
    tables = hf_tables is not None and (hf_tables / _TABLE_SUBDIR).is_dir()
    missing = tuple(r for r, ok in ((LAYOUT_REPO, layout), (TABLE_REPO, tables)) if not ok)
    return DoclingStatus(importable, layout, tables, None, missing)


@functools.cache
def _converter(kind: str, artifacts_path: str | None, tables: bool,
               ocr_name: str | None, ocr_langs: tuple[str, ...]) -> Any:
    import logging

    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter

    # docling logs file names per conversion at INFO; keep our logs to counts/IDs
    for name in ("docling", "docling_core", "docling_parse", "docling_ibm_models"):
        logging.getLogger(name).setLevel(logging.WARNING)
    if kind in ("docx", "html"):
        fmt = InputFormat.DOCX if kind == "docx" else InputFormat.HTML
        return DocumentConverter(allowed_formats=[fmt])

    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TesseractCliOcrOptions,
    )
    from docling.document_converter import ImageFormatOption, PdfFormatOption

    opts = PdfPipelineOptions()
    if artifacts_path is not None:
        opts.artifacts_path = artifacts_path
    opts.do_table_structure = tables
    opts.do_ocr = ocr_name is not None
    if ocr_name == "tesseract":
        opts.ocr_options = TesseractCliOcrOptions(lang=list(ocr_langs))
    for flag in ("do_picture_classification", "do_picture_description", "do_code_enrichment",
                 "do_formula_enrichment", "generate_picture_images", "generate_page_images"):
        if hasattr(opts, flag):
            setattr(opts, flag, False)
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.IMAGE],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=opts),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts),
        },
    )


def _page_of(item: Any) -> int | None:
    prov = getattr(item, "prov", None)
    if prov:
        return int(prov[0].page_no)
    return None


def convert_docling_document(doc: Any) -> list[ParsedBlock]:
    """Map a ``DoclingDocument`` body to blocks (page furniture is excluded)."""
    from docling_core.types.doc.document import TableItem  # type: ignore[attr-defined]
    from docling_core.types.doc.labels import DocItemLabel

    skip = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER, DocItemLabel.PICTURE}
    blocks: list[ParsedBlock] = []
    items: list[str] = []
    list_page: int | None = None

    def flush() -> None:
        nonlocal list_page
        if items:
            blocks.append(ParsedBlock("list", "\n".join(items), list_page))
            items.clear()
            list_page = None

    for item, _depth in doc.iterate_items():
        label = getattr(item, "label", None)
        if label in skip:
            continue
        page = _page_of(item)
        if isinstance(item, TableItem):
            flush()
            md = item.export_to_markdown(doc=doc).strip()
            grid = item.data.grid if item.data is not None else []
            text = "\n".join("\t".join(clean_inline(c.text) for c in row) for row in grid)
            if md or text:
                blocks.append(ParsedBlock("table", text or md, page, table_markdown=md or None))
            continue
        raw = getattr(item, "text", None)
        if not raw:
            continue
        text = clean_inline(raw)
        if not text:
            continue
        if label == DocItemLabel.LIST_ITEM:
            if not items:
                list_page = page
            items.append(f"- {text}")
            continue
        flush()
        if label == DocItemLabel.TITLE:
            blocks.append(ParsedBlock("heading", text, page, level=0))
        elif label == DocItemLabel.SECTION_HEADER:
            blocks.append(ParsedBlock("heading", text, page, level=int(getattr(item, "level", 1))))
        elif label == DocItemLabel.CAPTION:
            blocks.append(ParsedBlock("caption", text, page))
        elif label in (DocItemLabel.CODE, DocItemLabel.FORMULA):
            blocks.append(ParsedBlock("other", normalize_text(raw), page))
        else:
            blocks.append(ParsedBlock("paragraph", text, page))
    flush()
    return blocks


def parse_with_docling(path: Path, kind: str, status: DoclingStatus,
                       engine: OcrEngine | None) -> ParsedDocument:
    """Convert one file with Docling. Raises on conversion failure."""
    from docling.datamodel.base_models import ConversionStatus

    conv = _converter(kind, status.artifacts_path, status.tables,
                      engine.name if engine else None, engine.languages if engine else ())
    result = conv.convert(str(path), raises_on_error=False)
    if result.status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
        raise RuntimeError(f"docling status {result.status.value}")
    doc = result.document
    warnings = ["docling_partial"] if result.status == ConversionStatus.PARTIAL_SUCCESS else []
    n_pages = len(doc.pages) if kind in ("pdf", "image") else 0
    parser = "docling" + (f"+ocr:{engine.name}" if engine and kind in ("pdf", "image") else "")
    return ParsedDocument(blocks=convert_docling_document(doc), n_pages=n_pages,
                          parser=parser, warnings=warnings)
