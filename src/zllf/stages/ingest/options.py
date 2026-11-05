"""Ingest-specific knobs.

``RunConfig`` has no ``ingest`` section yet (contract change requested in the report), so
these options default to sensible values and can be overridden per process through
``ZLLF_INGEST_*`` environment variables or by passing an :class:`IngestOptions` to
:class:`~zllf.stages.ingest.stage.IngestStage`.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, replace
from typing import Literal

ParserChoice = Literal["auto", "docling", "fallback"]
OcrChoice = Literal["auto", "rapidocr", "tesseract", "none"]

_PARSERS: tuple[str, ...] = ("auto", "docling", "fallback")
_OCR: tuple[str, ...] = ("auto", "rapidocr", "tesseract", "none")


@dataclass(frozen=True)
class IngestOptions:
    """Parser / OCR / parallelism settings for the ingest stage (picklable)."""

    # "auto": Docling where it works offline (DOCX/HTML always; PDF/images only if the layout
    # model is cached locally), plain parsers otherwise. "docling": prefer Docling everywhere
    # (still falls back, with a warning, when its artefacts are missing). "fallback": never
    # use Docling.
    parser: ParserChoice = "auto"
    ocr_engine: OcrChoice = "auto"
    ocr_languages: tuple[str, ...] = ("en", "de")
    ocr_dpi: int = 200
    # pages with fewer extractable characters than this are treated as scanned -> OCR
    min_text_chars_per_page: int = 32
    # 0 = parse in-process (no pool); None = auto (min(4, cpus - 1, n_docs))
    workers: int | None = None
    timeout_s: float = 300.0  # per document
    # tabular sources (CSV/Parquet): rows per table block, hard cap on rows read per file
    rows_per_block: int = 25
    max_table_rows: int = 100_000
    docling_artifacts_path: str | None = None
    tokenizer: str | None = None  # override the student model name for token counting

    @classmethod
    def from_env(cls, base: IngestOptions | None = None) -> IngestOptions:
        """Apply ``ZLLF_INGEST_*`` / ``ZLLF_DOCLING_ARTIFACTS_PATH`` overrides."""
        opts = base or cls()
        env = os.environ
        updates: dict[str, object] = {}
        parser = env.get("ZLLF_INGEST_PARSER")
        if parser in _PARSERS:
            updates["parser"] = parser
        ocr = env.get("ZLLF_INGEST_OCR")
        if ocr in _OCR:
            updates["ocr_engine"] = ocr
        if (workers := env.get("ZLLF_INGEST_WORKERS")) is not None and workers.isdigit():
            updates["workers"] = int(workers)
        if (timeout := env.get("ZLLF_INGEST_TIMEOUT_S")) is not None:
            with contextlib.suppress(ValueError):
                updates["timeout_s"] = float(timeout)
        artifacts = env.get("ZLLF_DOCLING_ARTIFACTS_PATH") or env.get("DOCLING_ARTIFACTS_PATH")
        if artifacts and opts.docling_artifacts_path is None:
            updates["docling_artifacts_path"] = artifacts
        return replace(opts, **updates) if updates else opts  # type: ignore[arg-type]
