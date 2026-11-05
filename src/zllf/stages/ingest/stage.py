"""Stage ① Ingest (FR-01, FR-02 basic; P1.1).

Outputs in ``out_dir``: ``documents.parquet`` (DocumentRecord), ``pages.parquet`` (Block),
``chunks.parquet`` (Chunk, duplicates removed) and ``ingest_report.json`` (counts, IDs and
error codes only — never document text).
"""

from __future__ import annotations

import dataclasses
import functools
import json
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

from zllf.core.errors import ErrorCode, ZLLFError
from zllf.core.hashing import sha256_file
from zllf.core.logging import get_logger
from zllf.core.schemas import Block, Chunk, DocumentRecord, to_parquet
from zllf.stages.base import RunContext, Stage, StageResult
from zllf.stages.ingest.chunker import chunk_blocks
from zllf.stages.ingest.dedup import find_duplicates
from zllf.stages.ingest.docling_adapter import docling_status
from zllf.stages.ingest.model import ParsedDocument, ParseOutcome
from zllf.stages.ingest.ocr import engine_status, resolve_engine
from zllf.stages.ingest.options import IngestOptions
from zllf.stages.ingest.parse import ParseTask, default_workers, parse_all
from zllf.stages.ingest.registry import detect
from zllf.stages.ingest.tokens import get_token_counter

STAGE = "ingest"
_MAX_REPORTED_DUPLICATES = 1000


@dataclasses.dataclass
class _FileEntry:
    rel: str
    path: Path
    sha256: str
    mime: str
    kind: str | None
    skip_reason: str | None = None


def walk_corpus(corpus_dir: Path) -> list[tuple[str, Path]]:
    """All regular files below ``corpus_dir`` sorted by relative POSIX path.

    Hidden files/directories (any path component starting with ``.``) are skipped, as are
    symlinks resolving outside the corpus.
    """
    root = corpus_dir.resolve()
    out: list[tuple[str, Path]] = []
    for path in corpus_dir.rglob("*"):
        rel = path.relative_to(corpus_dir)
        if any(part.startswith(".") for part in rel.parts) or not path.is_file():
            continue
        if path.is_symlink() and not path.resolve().is_relative_to(root):
            continue
        out.append((rel.as_posix(), path))
    return sorted(out)


@functools.cache
def _language_detector(languages: tuple[str, ...]) -> Any | None:
    try:
        from lingua import IsoCode639_1, Language, LanguageDetectorBuilder

        langs = [Language.from_iso_code_639_1(getattr(IsoCode639_1, code.upper()))
                 for code in languages]
        if len(langs) < 2:
            return None
        return LanguageDetectorBuilder.from_languages(*langs).build()
    except Exception:
        return None


def detect_language(doc: ParsedDocument, languages: Sequence[str]) -> str | None:
    sample = " ".join(b.text for b in doc.blocks if b.type != "table")[:3000]
    if len(sample) < 200:  # tabular sources: fall back to cell text
        sample = " ".join(b.text for b in doc.blocks)[:3000]
    if len(sample) < 20:
        return None
    if len(languages) == 1:
        return languages[0]
    detector = _language_detector(tuple(languages))
    if detector is None:
        return None
    lang = detector.detect_language_of(sample)
    return lang.iso_code_639_1.name.lower() if lang is not None else None


class IngestStage(Stage):
    """Corpus folder -> documents, structural blocks and de-duplicated chunks."""

    name: ClassVar[str] = STAGE

    def __init__(self, options: IngestOptions | None = None) -> None:
        self.options = IngestOptions.from_env(options)

    def run(self, ctx: RunContext, out_dir: Path) -> StageResult:
        t0 = time.perf_counter()
        opts = self.options
        cfg = ctx.config
        log = get_logger("zllf.ingest", run_id=ctx.run_id, stage=STAGE)

        files = walk_corpus(ctx.corpus_dir)
        total = len(files)
        ctx.progress(STAGE, 0, total, "scanning")
        entries: list[_FileEntry] = []
        first_by_sha: dict[str, str] = {}
        for rel, path in files:
            ctx.check_cancelled()
            sha = sha256_file(path)
            ftype = detect(path)
            entry = _FileEntry(rel, path, sha, ftype.mime, ftype.kind, ftype.reason)
            if entry.kind is not None and sha in first_by_sha:
                entry.skip_reason = "duplicate_file"
            first_by_sha.setdefault(sha, rel)
            entries.append(entry)

        tasks = [ParseTask(e.rel, str(e.path), e.kind) for e in entries  # type: ignore[arg-type]
                 if e.kind is not None and e.skip_reason is None]
        done = total - len(tasks)
        ctx.progress(STAGE, done, total, "parsing")

        def on_done(_: ParseOutcome) -> None:
            nonlocal done
            done += 1
            ctx.progress(STAGE, done, total, "parsing")

        t_parse = time.perf_counter()
        outcomes = parse_all(tasks, opts, on_done=on_done, check_cancelled=ctx.check_cancelled)
        parse_seconds = time.perf_counter() - t_parse

        counter = get_token_counter(opts.tokenizer or cfg.models.student)
        ch = cfg.chunking
        documents: list[DocumentRecord] = []
        blocks: list[Block] = []
        chunks: list[Chunk] = []
        doc_rows: list[dict[str, Any]] = []
        for e in entries:
            doc_id = e.sha256[:16]
            row: dict[str, Any] = {"doc_id": doc_id, "path": e.rel, "kind": e.kind or "unknown"}
            if e.skip_reason is not None:
                code = ErrorCode.INGEST_UNSUPPORTED.value
                error = f"{code}: {e.skip_reason}" if e.kind is None else e.skip_reason
                documents.append(DocumentRecord(doc_id=doc_id, sha256=e.sha256, path=e.rel,
                                                mime=e.mime, status="skipped", error=error))
                if e.skip_reason == "duplicate_file":
                    row["duplicate_of_path"] = first_by_sha[e.sha256]
                doc_rows.append({**row, "status": "skipped", "error": error})
                continue
            out = outcomes[e.rel]
            doc = out.doc
            row.update(status=out.status, parser=doc.parser if doc else None,
                       seconds=round(out.seconds, 4), warnings=doc.warnings if doc else [])
            if out.status == "failed" or doc is None:
                documents.append(DocumentRecord(
                    doc_id=doc_id, sha256=e.sha256, path=e.rel, mime=e.mime, status="failed",
                    error=out.error, n_pages=doc.n_pages if doc else 0,
                    n_ocr_pages=doc.n_ocr_pages if doc else 0))
                doc_rows.append({**row, "error": out.error})
                continue
            for i, b in enumerate(doc.blocks):
                blocks.append(Block(doc_id=doc_id, block_index=i, type=b.type, text=b.text,
                                    page=b.page, level=b.level, table_markdown=b.table_markdown))
            drafts = chunk_blocks(doc.blocks, counter, ch.target_tokens, ch.max_tokens,
                                  ch.overlap_fraction)
            for i, d in enumerate(drafts):
                chunks.append(Chunk(chunk_id=f"{doc_id}-{i:05d}", doc_id=doc_id, index=i,
                                    text=d.text, n_tokens=d.n_tokens,
                                    heading_path=d.heading_path, page_start=d.page_start,
                                    page_end=d.page_end, is_table=d.is_table))
            documents.append(DocumentRecord(
                doc_id=doc_id, sha256=e.sha256, path=e.rel, mime=e.mime, status="parsed",
                n_pages=doc.n_pages, n_ocr_pages=doc.n_ocr_pages, n_chars=doc.n_chars,
                language=detect_language(doc, cfg.sanitize.languages)))
            row.update(n_pages=doc.n_pages, n_ocr_pages=doc.n_ocr_pages,
                       n_blocks=len(doc.blocks), n_chunks=len(drafts))
            doc_rows.append(row)

        dedup = find_duplicates([c.text for c in chunks], ch.dedup_jaccard,
                                ch.minhash_num_perm, ch.shingle_size)
        kept = [c for i, c in enumerate(chunks) if i not in dedup.duplicate_of]
        duplicate_map = {chunks[i].chunk_id: chunks[j].chunk_id
                         for i, j in sorted(dedup.duplicate_of.items())}
        if not kept:
            raise ZLLFError(ErrorCode.INGEST_EMPTY_CORPUS,
                            f"{total} files, {len(tasks)} parseable, 0 chunks")

        to_parquet(documents, out_dir / "documents.parquet")
        to_parquet(blocks, out_dir / "pages.parquet")
        to_parquet(kept, out_dir / "chunks.parquet")

        seconds = time.perf_counter() - t0
        report = self._report(
            entries=entries, documents=documents, doc_rows=doc_rows, chunks=chunks, kept=kept,
            duplicate_map=duplicate_map, exact=dedup.exact, near=dedup.near,
            counter_name=counter.name, seconds=seconds, parse_seconds=parse_seconds,
            n_tasks=len(tasks), ctx=ctx,
        )
        (out_dir / "ingest_report.json").write_text(json.dumps(report, indent=2, sort_keys=True),
                                                    encoding="utf-8")
        metrics = {
            "n_files": total,
            "n_parsed": report["counts"]["by_status"].get("parsed", 0),
            "n_failed": report["counts"]["by_status"].get("failed", 0),
            "n_skipped": report["counts"]["by_status"].get("skipped", 0),
            "n_pages": report["pages"]["total"],
            "n_ocr_pages": report["pages"]["ocr"],
            "n_chunks": len(kept),
            "n_tokens": report["chunks"]["tokens_total"],
            "n_duplicates": dedup.exact + dedup.near,
            "seconds": round(seconds, 3),
        }
        ctx.progress(STAGE, total, total, "done")
        log.info("ingest_done", **metrics)
        return StageResult(metrics=metrics)

    # ------------------------------------------------------------------ report
    def _report(self, *, entries: list[_FileEntry], documents: list[DocumentRecord],
                doc_rows: list[dict[str, Any]], chunks: list[Chunk], kept: list[Chunk],
                duplicate_map: dict[str, str], exact: int, near: int, counter_name: str,
                seconds: float, parse_seconds: float, n_tasks: int,
                ctx: RunContext) -> dict[str, Any]:
        opts = self.options
        by_status = Counter(d.status for d in documents)
        by_type: Counter[str] = Counter()
        by_type_status: dict[str, Counter[str]] = {}
        for e, d in zip(entries, documents, strict=True):
            kind = e.kind or "unsupported"
            by_type[kind] += 1
            by_type_status.setdefault(kind, Counter())[d.status] += 1
        parsers = Counter(r["parser"] for r in doc_rows if r.get("parser"))
        n_pages = sum(d.n_pages for d in documents if d.status == "parsed")
        n_ocr = sum(d.n_ocr_pages for d in documents)
        tokens = [c.n_tokens for c in kept]
        dstatus = docling_status(opts.docling_artifacts_path)
        engine = resolve_engine(opts.ocr_engine, opts.ocr_languages)
        workers = opts.workers if opts.workers is not None else default_workers(n_tasks)
        return {
            "stage": STAGE,
            "run_id": ctx.run_id,
            "seconds": round(seconds, 3),
            "parse_seconds": round(parse_seconds, 3),
            "documents_per_second": round(len(documents) / seconds, 3) if seconds else None,
            "pages_per_second": round(n_pages / parse_seconds, 3) if parse_seconds else None,
            "workers": workers,
            "counts": {
                "files": len(documents),
                "by_status": dict(sorted(by_status.items())),
                "by_type": dict(sorted(by_type.items())),
                "by_type_status": {k: dict(sorted(v.items()))
                                   for k, v in sorted(by_type_status.items())},
            },
            "pages": {"total": n_pages, "ocr": n_ocr},
            "parsers": dict(sorted(parsers.items())),
            "failures": [{"path": d.path, "doc_id": d.doc_id, "error": d.error}
                         for d in documents if d.status == "failed"],
            "skipped": [{"path": d.path, "doc_id": d.doc_id, "reason": d.error}
                        for d in documents if d.status == "skipped"],
            "chunks": {
                "before_dedup": len(chunks),
                "kept": len(kept),
                "duplicates_removed": exact + near,
                "duplicates_exact": exact,
                "duplicates_near": near,
                "duplicate_of": dict(list(duplicate_map.items())[:_MAX_REPORTED_DUPLICATES]),
                "tables": sum(1 for c in kept if c.is_table),
                "tokens_total": sum(tokens),
                "tokens_mean": round(sum(tokens) / len(tokens), 1) if tokens else 0,
                "tokens_max": max(tokens, default=0),
            },
            "tokenizer": counter_name,
            "chunking": ctx.config.chunking.model_dump(),
            "options": dataclasses.asdict(opts),
            "docling": dstatus.report(),
            "ocr": {"engine": engine.name if engine else None,
                    "languages": list(engine.languages) if engine else [],
                    **engine_status(opts.ocr_languages)},
            "documents": doc_rows,
        }
