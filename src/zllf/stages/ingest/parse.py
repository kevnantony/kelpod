"""Per-document parse dispatch and the parallel runner (P1.1.9).

``parse_one`` never raises: every failure becomes a ``failed`` :class:`ParseOutcome` with an
error code and a text-free reason. ``parse_all`` runs documents in a process pool with a
per-document timeout (SIGALRM inside the worker) plus a parent-side stall backstop that
kills hung workers (e.g. stuck in native code) and isolates the culprit.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zllf.core.errors import ErrorCode
from zllf.stages.ingest import loaders
from zllf.stages.ingest.model import ParsedDocument, ParseOutcome
from zllf.stages.ingest.ocr import resolve_engine
from zllf.stages.ingest.options import IngestOptions
from zllf.stages.ingest.registry import Kind


@dataclass(frozen=True)
class ParseTask:
    rel_path: str
    abs_path: str
    kind: Kind


class DocumentTimeout(Exception):
    pass


@contextmanager
def _alarm(seconds: float) -> Iterator[None]:
    """Raise :class:`DocumentTimeout` after ``seconds`` (main thread only, else no-op)."""
    if seconds <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _handler(signum: int, frame: Any) -> None:
        raise DocumentTimeout()

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def choose_parser(kind: Kind, opts: IngestOptions) -> str:
    """'docling' or 'fallback' for this file kind (Docling only where it works offline)."""
    if opts.parser == "fallback" or kind not in ("pdf", "docx", "html", "image"):
        return "fallback"
    from zllf.stages.ingest.docling_adapter import docling_status

    status = docling_status(opts.docling_artifacts_path)
    if not status.importable:
        return "fallback"
    if kind in ("docx", "html"):
        return "docling"
    return "docling" if status.pdf_ready else "fallback"


def _parse(task: ParseTask, opts: IngestOptions) -> ParsedDocument:
    path = Path(task.abs_path)
    kind = task.kind
    engine = (resolve_engine(opts.ocr_engine, opts.ocr_languages)
              if kind in ("pdf", "image") else None)
    warnings: list[str] = []
    if kind in ("pdf", "docx", "html", "image"):
        if choose_parser(kind, opts) == "docling":
            from zllf.stages.ingest.docling_adapter import docling_status, parse_with_docling

            status = docling_status(opts.docling_artifacts_path)
            doc = parse_with_docling(path, kind, status, engine)
            if kind == "pdf":
                texts = loaders.pdf_page_texts(path)
                doc.n_ocr_pages = len(loaders.scanned_pages(
                    texts, opts.min_text_chars_per_page)) if engine else 0
            elif kind == "image":
                doc.n_ocr_pages = doc.n_pages if engine else 0
            return doc
        if opts.parser == "docling":
            warnings.append("docling_artifacts_missing")
    loader: dict[str, Callable[[Path, IngestOptions], ParsedDocument]] = {
        "txt": loaders.load_txt, "md": loaders.load_md, "csv": loaders.load_csv,
        "parquet": loaders.load_parquet, "xml": loaders.load_xml, "html": loaders.load_html,
        "docx": loaders.load_docx,
    }
    if kind == "pdf":
        doc = loaders.load_pdf(path, opts, engine)
    elif kind == "image":
        doc = loaders.load_image(path, opts, engine)
    else:
        doc = loader[kind](path, opts)
    doc.warnings = warnings + doc.warnings
    return doc


def parse_one(task: ParseTask, opts: IngestOptions) -> ParseOutcome:
    """Parse one file; never raises (except KeyboardInterrupt/SystemExit)."""
    t0 = time.perf_counter()
    try:
        with _alarm(opts.timeout_s):
            doc = _parse(task, opts)
    except DocumentTimeout:
        return ParseOutcome(task.rel_path, "failed",
                            error=f"{ErrorCode.INGEST_PARSE_FAILED.value}: timeout",
                            seconds=time.perf_counter() - t0)
    except loaders.ParseError as exc:
        return ParseOutcome(task.rel_path, "failed",
                            error=f"{ErrorCode.INGEST_PARSE_FAILED.value}: {exc}",
                            seconds=time.perf_counter() - t0)
    except Exception as exc:  # isolate any loader failure to this document
        # exception *type* only: messages of parser libraries may quote document content
        return ParseOutcome(task.rel_path, "failed",
                            error=f"{ErrorCode.INGEST_PARSE_FAILED.value}: "
                                  f"{type(exc).__name__}",
                            seconds=time.perf_counter() - t0)
    dt = time.perf_counter() - t0
    if not doc.blocks:
        reason = "ocr_unavailable" if "ocr_unavailable" in doc.warnings else "no_text"
        return ParseOutcome(task.rel_path, "failed", doc=doc,
                            error=f"{ErrorCode.INGEST_PARSE_FAILED.value}: {reason}", seconds=dt)
    return ParseOutcome(task.rel_path, "parsed", doc=doc, seconds=dt)


def _failed(task: ParseTask, reason: str) -> ParseOutcome:
    return ParseOutcome(task.rel_path, "failed",
                        error=f"{ErrorCode.INGEST_PARSE_FAILED.value}: {reason}")


def _worker_init() -> None:
    """Pool initializer: ingest never needs the network, so workers block all egress."""
    from zllf.core.offline import apply_offline_env, block_egress

    apply_offline_env()
    block_egress().__enter__()


def default_workers(n_tasks: int) -> int:
    return max(0, min(4, (os.cpu_count() or 2) - 1, n_tasks))


def _kill(executor: ProcessPoolExecutor) -> None:
    procs = list(getattr(executor, "_processes", {}).values())
    for p in procs:
        if p.is_alive():
            p.kill()
    executor.shutdown(wait=False, cancel_futures=True)


def parse_all(
    tasks: Sequence[ParseTask],
    opts: IngestOptions,
    on_done: Callable[[ParseOutcome], None] = lambda o: None,
    check_cancelled: Callable[[], None] = lambda: None,
) -> dict[str, ParseOutcome]:
    """Parse all tasks; returns outcomes keyed by relative path (order-independent)."""
    workers = opts.workers if opts.workers is not None else default_workers(len(tasks))
    results: dict[str, ParseOutcome] = {}
    if workers <= 0 or len(tasks) <= 1:
        for task in tasks:
            check_cancelled()
            out = parse_one(task, opts)
            results[task.rel_path] = out
            on_done(out)
        return results

    backstop = opts.timeout_s * 1.5 + 30.0
    pending = list(tasks)
    ctx = mp.get_context("spawn")
    while pending:
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                       initializer=_worker_init)
        futures: dict[Future[ParseOutcome], ParseTask] = {
            executor.submit(parse_one, t, opts): t for t in pending
        }
        order = {t.rel_path: i for i, t in enumerate(pending)}
        not_done = set(futures)
        last_progress = time.monotonic()
        broken = False
        try:
            while not_done:
                check_cancelled()
                done, not_done = wait(not_done, timeout=1.0, return_when=FIRST_COMPLETED)
                for fut in done:
                    task = futures[fut]
                    try:
                        out = fut.result()
                    except Exception:  # BrokenProcessPool: a worker died (segfault/OOM)
                        broken = True
                        not_done.add(fut)
                        continue
                    results[task.rel_path] = out
                    on_done(out)
                    last_progress = time.monotonic()
                if broken or time.monotonic() - last_progress > backstop:
                    break
        except BaseException:
            _kill(executor)
            raise
        if not not_done:
            executor.shutdown(wait=True)
            break
        _kill(executor)
        remaining = sorted((futures[f] for f in not_done), key=lambda t: order[t.rel_path])
        if workers == 1:
            # single worker runs tasks in submission order: the first unfinished one is it
            culprit = remaining[0]
            out = _failed(culprit, "worker_crashed" if broken else "timeout")
            results[culprit.rel_path] = out
            on_done(out)
            remaining = remaining[1:]
        workers = 1
        pending = remaining
    return results
