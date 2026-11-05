"""Worker loop (P1.8): claim → run_pipeline → map outcome to run state.

- One run at a time per machine (single accelerator): a worker holds an exclusive file lock
  (``<data>/worker.lock``) while it executes runs; a second worker waits as a standby.
- Claiming is an atomic ``UPDATE run SET state='running' … WHERE state='queued'`` with a
  lease (owner + expiry) that a heartbeat thread renews.
- On start-up, runs left ``running`` by a dead worker (lease expired, or owner process on
  this host gone) are re-queued; ``run_pipeline`` then resumes from the manifest.
- Cancel: the API sets ``run.cancel_requested``; the pipeline polls it via
  ``cancel_requested``. SIGTERM/stop() interrupts the current stage cooperatively and puts
  the run back to ``queued`` so it resumes on the next start.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import socket
import threading
import time
from pathlib import Path
from typing import IO, Any

from sqlalchemy import update
from sqlalchemy.engine import Engine
from sqlmodel import Session, col, select

from zllf.api.settings import Settings
from zllf.core.errors import USER_MESSAGES, ErrorCode
from zllf.core.logging import get_logger
from zllf.core.manifest import STAGE_ORDER, RunPaths, append_event, read_manifest
from zllf.db.engine import audit, session_scope
from zllf.db.models import Adapter, Document, Run, StageRun

log = get_logger("zllf.worker")
STAGE_EVENTS = frozenset({"stage_started", "stage_completed", "stage_failed",
                          "awaiting_approval", "cancelled"})


def _host() -> str:
    return socket.gethostname()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _owner_dead(owner: str | None) -> bool:
    """True when ``owner`` (``host:pid:token``) is a process on this host that is gone."""
    if not owner:
        return True
    parts = owner.split(":")
    if len(parts) < 2 or parts[0] != _host() or not parts[1].isdigit():
        return False
    return not _pid_alive(int(parts[1]))


class _AcceleratorLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh: IO[str] | None = None

    def acquire(self, stop: threading.Event, poll: float) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = self.path.open("a+")
        while not stop.is_set():
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._fh = fh
                return True
            except BlockingIOError:
                stop.wait(poll)
        fh.close()
        return False

    def release(self) -> None:
        if self._fh is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None


class Worker:
    """Executes queued runs one at a time."""

    def __init__(self, settings: Settings, engine: Engine, *, worker_id: str | None = None
                 ) -> None:
        self.settings = settings
        self.engine = engine
        self.worker_id = worker_id or f"{_host()}:{os.getpid()}:{threading.get_ident()}"
        self.stop_event = threading.Event()
        self._lock = _AcceleratorLock(settings.root / "worker.lock")
        self._has_lock = False

    # ------------------------------------------------------------------ lifecycle
    def stop(self) -> None:
        self.stop_event.set()

    def run_forever(self) -> None:
        log.info("worker_started", worker_id=self.worker_id)
        try:
            if not self._ensure_lock():
                return
            self.recover_stale()
            while not self.stop_event.is_set():
                if not self.run_once():
                    self.recover_stale()
                    self.stop_event.wait(self.settings.worker_poll_seconds)
        finally:
            self._release_lock()
            log.info("worker_stopped", worker_id=self.worker_id)

    def _ensure_lock(self) -> bool:
        if not self._has_lock:
            self._has_lock = self._lock.acquire(self.stop_event, self.settings.worker_poll_seconds)
        return self._has_lock

    def close(self) -> None:
        """Release the accelerator lock (after ``run_once`` use)."""
        self._release_lock()

    def _release_lock(self) -> None:
        if self._has_lock:
            self._lock.release()
            self._has_lock = False

    # ------------------------------------------------------------------ claiming
    def recover_stale(self) -> int:
        """Re-queue runs whose worker died (lease expired or owner pid gone)."""
        now = time.time()
        n = 0
        with session_scope(self.engine) as db:
            for run in db.exec(select(Run).where(Run.state == "running")).all():
                if run.lease_owner == self.worker_id:
                    continue
                expired = run.lease_expires_at is None or run.lease_expires_at < now
                if expired or _owner_dead(run.lease_owner):
                    run.state = "cancelled" if run.cancel_requested else "queued"
                    run.lease_owner, run.lease_expires_at = None, None
                    if run.state == "cancelled":
                        run.finished_at = now
                        run.error_code, run.error_message = "E_CANCELLED", _msg("E_CANCELLED")
                    db.add(run)
                    audit(db, "worker", "run.recovered", "run", run.id, new_state=run.state)
                    n += 1
        if n:
            log.info("stale_runs_recovered", count=n)
        return n

    def claim_next(self) -> str | None:
        """Atomically move the oldest queued run to ``running`` under our lease."""
        with session_scope(self.engine) as db:
            candidates = db.exec(select(Run.id).where(Run.state == "queued")
                                 .order_by(col(Run.created_at)).limit(5)).all()
            for run_id in candidates:
                now = time.time()
                result = db.exec(
                    update(Run)
                    .where(col(Run.id) == run_id, col(Run.state) == "queued")
                    .values(state="running", lease_owner=self.worker_id,
                            lease_expires_at=now + self.settings.lease_seconds,
                            started_at=now, finished_at=None, error_code=None,
                            error_message=None)
                )
                if result.rowcount == 1:
                    audit(db, "worker", "run.started", "run", run_id, worker=self.worker_id)
                    return str(run_id)
        return None

    def run_once(self) -> bool:
        """Process at most one queued run. Returns True if a run was processed."""
        if not self._ensure_lock():
            return False
        run_id = self.claim_next()
        if run_id is None:
            return False
        self.process(run_id)
        return True

    # ------------------------------------------------------------------ execution
    def process(self, run_id: str) -> str:
        from zllf.pipeline import run_pipeline

        with Session(self.engine) as db:
            run = db.get(Run, run_id)
            assert run is not None
            paths = RunPaths(Path(run.run_dir))
        log.info("run_started", run_id=run_id)
        heartbeat = threading.Thread(target=self._heartbeat, args=(run_id,), daemon=True)
        done = threading.Event()
        self._hb_done = done
        heartbeat.start()
        cancel = _CancelPoll(self.engine, run_id, self.stop_event)
        try:
            outcome: str = run_pipeline(
                paths,
                cancel_requested=cancel,
                event_sink=lambda event: self._on_event(run_id, paths, event),
            )
        except Exception as exc:  # pipeline bug / unreadable manifest
            log.error("run_crashed", run_id=run_id, error=type(exc).__name__)
            outcome = "failed"
        finally:
            done.set()
            heartbeat.join(timeout=5)
        return self._finish(run_id, paths, outcome, interrupted=self.stop_event.is_set()
                            and not cancel.user_cancelled)

    def _heartbeat(self, run_id: str) -> None:
        interval = max(0.5, self.settings.lease_seconds / 3)
        while not self._hb_done.wait(interval):
            with contextlib.suppress(Exception), session_scope(self.engine) as db:
                db.exec(update(Run).where(col(Run.id) == run_id,
                                          col(Run.lease_owner) == self.worker_id)
                        .values(lease_expires_at=time.time() + self.settings.lease_seconds))

    def _on_event(self, run_id: str, paths: RunPaths, event: dict[str, Any]) -> None:
        etype = event.get("type")
        if etype not in STAGE_EVENTS:
            return
        try:
            with session_scope(self.engine) as db:
                run = db.get(Run, run_id)
                if run is None:
                    return
                if event.get("stage"):
                    run.current_stage = str(event["stage"])
                    db.add(run)
                sync_stage_runs(db, run_id, paths)
                if etype == "stage_completed" and event.get("stage") == "ingest":
                    sync_documents(db, run.workspace_id, paths)
        except Exception as exc:  # never let bookkeeping kill a run
            log.warning("event_sync_failed", run_id=run_id, error=type(exc).__name__)

    def _finish(self, run_id: str, paths: RunPaths, outcome: str, *, interrupted: bool) -> str:
        now = time.time()
        with session_scope(self.engine) as db:
            run = db.get(Run, run_id)
            if run is None:
                return outcome
            sync_stage_runs(db, run_id, paths)
            if outcome == "cancelled" and interrupted:
                state = "queued"  # worker shutdown, not a user cancel: resume later
            elif outcome == "cancelled":
                state = "cancelled"
                run.error_code, run.error_message = "E_CANCELLED", _msg("E_CANCELLED")
            elif outcome == "failed":
                state = "failed"
                run.error_code = failed_stage_error(paths)
                run.error_message = _msg(run.error_code)
            else:
                state = outcome  # completed | awaiting_approval
            run.state = state
            run.lease_owner, run.lease_expires_at = None, None
            if state in ("completed", "failed", "cancelled"):
                run.finished_at = now
            if state == "completed":
                run.current_stage = None
            db.add(run)
            audit(db, "worker", f"run.{state}", "run", run_id, error_code=run.error_code)
            if state == "completed":
                adapter = register_adapter(db, run, paths)
                existing = db.exec(select(Adapter.id).where(Adapter.run_id == run_id)).first()
                with contextlib.suppress(OSError):
                    append_event(paths, {"run_id": run_id, "type": "run_completed",
                                         "adapter_id": adapter.id if adapter else existing})
        log.info("run_finished", run_id=run_id, state=state)
        return state


class _CancelPoll:
    """``cancel_requested`` callable: DB flag (throttled) or worker shutdown."""

    def __init__(self, engine: Engine, run_id: str, stop: threading.Event,
                 interval: float = 0.5) -> None:
        self.engine, self.run_id, self.stop, self.interval = engine, run_id, stop, interval
        self.user_cancelled = False
        self._last = 0.0

    def __call__(self) -> bool:
        if self.user_cancelled:
            return True
        now = time.monotonic()
        if now - self._last >= self.interval:
            self._last = now
            with Session(self.engine) as db:
                run = db.get(Run, self.run_id)
                self.user_cancelled = run is None or bool(run.cancel_requested)
        return self.user_cancelled or self.stop.is_set()


def _msg(code: str | None) -> str | None:
    if code is None:
        return None
    try:
        return USER_MESSAGES[ErrorCode(code)]
    except ValueError:
        return USER_MESSAGES[ErrorCode.INTERNAL]


def failed_stage_error(paths: RunPaths) -> str:
    try:
        manifest = read_manifest(paths)
    except (OSError, ValueError):
        return ErrorCode.INTERNAL.value
    for name in STAGE_ORDER:
        entry = manifest.stages[name]
        if entry.status == "failed" and entry.error:
            try:
                return ErrorCode(entry.error).value
            except ValueError:
                return ErrorCode.INTERNAL.value
    return ErrorCode.INTERNAL.value


def sync_stage_runs(db: Session, run_id: str, paths: RunPaths) -> None:
    """Mirror manifest stage entries into ``stage_run``."""
    try:
        manifest = read_manifest(paths)
    except (OSError, ValueError):
        return
    rows = {r.stage: r for r in db.exec(select(StageRun).where(StageRun.run_id == run_id)).all()}
    for name in STAGE_ORDER:
        entry = manifest.stages[name]
        row = rows.get(name) or StageRun(run_id=run_id, stage=name)
        row.state, row.started_at, row.finished_at = entry.status, entry.started_at, \
            entry.finished_at
        row.output_hash, row.metrics_json, row.error = entry.output_hash, dict(entry.metrics), \
            entry.error
        db.add(row)


def sync_documents(db: Session, workspace_id: str, paths: RunPaths) -> None:
    """Copy per-document ingest status/pages from documents.parquet into the DB."""
    parquet = paths.stage("ingest") / "documents.parquet"
    if not parquet.exists():
        return
    from zllf.core.errors import USER_MESSAGES as MSG
    from zllf.core.schemas import DocumentRecord, read_parquet

    records = {r.path: r for r in read_parquet(parquet, DocumentRecord)}
    for doc in db.exec(select(Document).where(Document.workspace_id == workspace_id)).all():
        rec = records.get(doc.path)
        if rec is None or (doc.status == "skipped" and doc.error):
            continue
        doc.status, doc.pages = rec.status, rec.n_pages or None
        err = None
        if rec.error:
            try:
                err = MSG[ErrorCode(rec.error)]
            except ValueError:
                err = MSG[ErrorCode.INGEST_PARSE_FAILED] if rec.status == "failed" else None
        doc.error = err
        db.add(doc)


def register_adapter(db: Session, run: Run, paths: RunPaths) -> Adapter | None:
    """Register the packaged adapter of a completed run (idempotent)."""
    if db.exec(select(Adapter).where(Adapter.run_id == run.id)).first() is not None:
        return None
    meta: dict[str, Any] | None = None
    try:
        from zllf.stages.package.registry import register_adapter as package_meta

        meta = package_meta(paths)
    except ImportError:
        meta = _local_adapter_meta(paths)
    except Exception as exc:
        log.warning("adapter_registration_failed", run_id=run.id, error=type(exc).__name__)
        meta = _local_adapter_meta(paths)
    if not meta:
        return None
    try:
        manifest = read_manifest(paths)
        base_model = (manifest.config.get("models") or {}).get("student")
    except (OSError, ValueError):
        base_model = None
    adapter = Adapter(
        id=str(meta.get("adapter_id") or f"adp-{run.id}"), run_id=run.id,
        workspace_id=run.workspace_id, base_model=str(meta.get("base_model") or base_model or ""),
        path=str(meta["path"]), sha256=str(meta.get("sha256") or ""),
        size_bytes=int(meta.get("size_bytes") or 0),
        gates_passed=bool(meta.get("gates_passed")),
    )
    db.add(adapter)
    audit(db, "worker", "adapter.register", "adapter", adapter.id, run_id=run.id,
          gates_passed=adapter.gates_passed)
    return adapter


def _local_adapter_meta(paths: RunPaths) -> dict[str, Any] | None:
    from zllf.api.artefacts import gates_passed
    from zllf.core.hashing import sha256_dir

    for adapter in (paths.stage("package") / "adapter", paths.stage("train") / "adapter"):
        files = [p for p in adapter.rglob("*") if p.is_file()] if adapter.is_dir() else []
        if files:
            return {"path": str(adapter), "sha256": sha256_dir(adapter),
                    "size_bytes": sum(p.stat().st_size for p in files),
                    "gates_passed": gates_passed(paths)}
    return None


class WorkerThread:
    """Run a :class:`Worker` in a daemon thread (``zllf serve-api --with-worker``)."""

    def __init__(self, settings: Settings, engine: Engine) -> None:
        self.worker = Worker(settings, engine)
        self._thread = threading.Thread(target=self.worker.run_forever, daemon=True,
                                         name="zllf-worker")

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        self.worker.stop()
        self._thread.join(timeout=timeout)
