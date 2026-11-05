"""Run control (create/approve/cancel/resume), SSE events and result artefacts."""

from __future__ import annotations

import asyncio
import contextlib
import shutil
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse
from sqlalchemy.engine import Engine
from sqlmodel import Session, col, select
from sse_starlette.sse import EventSourceResponse

from zllf.api import artefacts
from zllf.api.deps import DB, CurrentUser, SettingsDep, get_run, get_workspace
from zllf.api.errors import ApiError, conflict_state
from zllf.api.schemas import RedactionSummary, RunCreate, RunOut, RunSummary, Scorecard, TrialOut
from zllf.api.views import current_stage, load_manifest, run_adapter_id, run_out, run_summary
from zllf.core.config import load_config
from zllf.core.manifest import RunPaths, append_event
from zllf.db.engine import audit
from zllf.db.models import ACTIVE_RUN_STATES, TERMINAL_RUN_STATES, Document, Run

router = APIRouter(prefix="/api", tags=["runs"])


@router.post("/workspaces/{workspace_id}/runs", response_model=RunOut, status_code=201)
def create_run(
    workspace_id: str, body: RunCreate, db: DB, user: CurrentUser, settings: SettingsDep
) -> RunOut:
    from zllf.pipeline import init_run, new_run_id

    ws = get_workspace(db, user, workspace_id)
    if body.mode == "C" and not body.json_schema:
        raise ApiError(422, "E_SCHEMA_REQUIRED", "Structured extraction needs a form (JSON "
                       "schema). Pick a template or paste one.")
    hardware = body.hardware or settings.hardware
    if hardware not in settings.hardware_names():
        raise ApiError(422, "E_VALIDATION", f"Unknown hardware profile {hardware!r}.")
    active = db.exec(select(Run.id).where(Run.workspace_id == ws.id)
                     .where(col(Run.state).in_(ACTIVE_RUN_STATES))).first()
    if active:
        raise ApiError(409, "E_RUN_ACTIVE", "A run is already in progress for this workspace. "
                       "Wait for it to finish or cancel it first.")
    usable = db.exec(select(Document.id).where(Document.workspace_id == ws.id)
                     .where(col(Document.status).in_(("uploaded", "parsed")))).first()
    if not usable:
        raise ApiError(409, "E_NO_DOCUMENTS", "There are no documents in this workspace yet. "
                       "Upload documents first.")

    overrides: dict[str, Any] = {}
    if body.json_schema is not None:
        overrides["json_schema"] = body.json_schema
    if body.budget_seconds is not None:
        overrides["budget"] = {"total_seconds": body.budget_seconds}
    try:
        base = load_config(profile=f"mode_{body.mode.lower()}", hardware=hardware,
                           config_dir=settings.config_dir)
        if base.models.backend != "fake" and ws.base_model != base.models.student:
            overrides["models"] = {"student": ws.base_model}
        config = load_config(profile=f"mode_{body.mode.lower()}", hardware=hardware,
                             overrides=overrides, config_dir=settings.config_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise ApiError(422, "E_VALIDATION", "The run settings are not valid.") from exc

    corpus = settings.corpus_dir(ws.id)
    corpus.mkdir(parents=True, exist_ok=True)
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = new_run_id()
    paths = init_run(settings.runs_dir, corpus, config, run_id=run_id)
    gold = settings.gold_path(ws.id)
    if gold.is_file():  # snapshot: later gold edits do not change this run's evaluation
        shutil.copyfile(gold, paths.root / "gold.jsonl")
    run = Run(id=run_id, workspace_id=ws.id, mode=body.mode, profile=config.profile_name,
              hardware=hardware, state="queued", budget_s=config.budget.total_seconds,
              seed=config.seed, run_dir=str(paths.root), created_by=user.id)
    db.add(run)
    audit(db, user.id, "run.create", "run", run_id, workspace_id=ws.id, mode=body.mode,
          hardware=hardware)
    db.flush()
    return run_out(run, db)


@router.get("/workspaces/{workspace_id}/runs", response_model=list[RunSummary])
def list_runs(workspace_id: str, db: DB, user: CurrentUser) -> list[RunSummary]:
    get_workspace(db, user, workspace_id)
    runs = db.exec(select(Run).where(Run.workspace_id == workspace_id)
                   .order_by(col(Run.created_at).desc())).all()
    return [run_summary(r, load_manifest(r)) for r in runs]


@router.get("/runs/{run_id}", response_model=RunOut)
def read_run(run_id: str, db: DB, user: CurrentUser) -> RunOut:
    return run_out(get_run(db, user, run_id), db)


@router.post("/runs/{run_id}/approve", response_model=RunOut)
def approve_run(run_id: str, db: DB, user: CurrentUser) -> RunOut:
    from zllf.pipeline import approve

    run = get_run(db, user, run_id)
    if run.state != "awaiting_approval":
        raise conflict_state("This run is not waiting for approval.")
    paths = RunPaths(Path(run.run_dir))
    approve(paths)
    emit_run_event(paths, run.id, "approved", stage="sanitize")
    run.state = "queued"
    db.add(run)
    audit(db, user.id, "run.approve", "run", run.id)
    return run_out(run, db)


@router.post("/runs/{run_id}/cancel", response_model=RunOut)
def cancel_run(run_id: str, db: DB, user: CurrentUser) -> RunOut:
    """Queued/awaiting runs are cancelled at once; running ones when the worker stops them."""
    run = get_run(db, user, run_id)
    if run.state not in ACTIVE_RUN_STATES:
        raise conflict_state("Only a run that is in progress can be cancelled.")
    run.cancel_requested = True
    if run.state in ("queued", "awaiting_approval"):
        import time

        run.state = "cancelled"
        run.finished_at = time.time()
        run.error_code = "E_CANCELLED"
        run.error_message = "The job was cancelled."
        emit_run_event(RunPaths(Path(run.run_dir)), run.id, "cancelled",
                       stage=current_stage(run, load_manifest(run)))
    db.add(run)
    audit(db, user.id, "run.cancel", "run", run.id)
    return run_out(run, db)


@router.post("/runs/{run_id}/resume", response_model=RunOut)
def resume_run(run_id: str, db: DB, user: CurrentUser) -> RunOut:
    run = get_run(db, user, run_id)
    if run.state not in ("failed", "cancelled"):
        raise conflict_state("Only a cancelled or failed run can be resumed.")
    active = db.exec(select(Run.id).where(Run.workspace_id == run.workspace_id)
                     .where(col(Run.state).in_(ACTIVE_RUN_STATES))).first()
    if active:
        raise ApiError(409, "E_RUN_ACTIVE", "Another run is in progress for this workspace.")
    run.state = "queued"
    run.cancel_requested = False
    run.error_code = None
    run.error_message = None
    run.finished_at = None
    db.add(run)
    emit_run_event(RunPaths(Path(run.run_dir)), run.id, "resumed")
    audit(db, user.id, "run.resume", "run", run.id)
    return run_out(run, db)


# ------------------------------------------------------------------ events (SSE)
def emit_run_event(paths: RunPaths, run_id: str, type_: str, **payload: Any) -> None:
    """Append a control event (approved/resumed/cancelled) to the run's event log."""
    with contextlib.suppress(OSError):
        append_event(paths, {"run_id": run_id, "type": type_,
                             **{k: v for k, v in payload.items() if v is not None}})


def _run_state(engine: Engine, run_id: str) -> str | None:
    with Session(engine) as s:
        run = s.get(Run, run_id)
        return run.state if run else None


async def _follow_events(
    request: Request, engine: Engine, run_id: str, events_path: Path, last_id: int
) -> AsyncIterator[dict[str, Any]]:
    pos, line_no, buf = 0, 0, b""
    while True:
        if await request.is_disconnected():
            return
        data = b""
        try:
            with events_path.open("rb") as f:
                f.seek(pos)
                data = f.read(1024 * 1024)
        except FileNotFoundError:
            pass
        if data:
            pos += len(data)
            buf += data
            *lines, buf = buf.split(b"\n")
            for raw in lines:
                if not raw.strip():
                    continue
                line_no += 1
                if line_no <= last_id:
                    continue
                yield {"event": "message", "id": str(line_no), "data": raw.decode("utf-8")}
            continue
        state = await asyncio.to_thread(_run_state, engine, run_id)
        if state is None or state in TERMINAL_RUN_STATES:
            # one last read so events written just before the state change are not lost
            try:
                if events_path.stat().st_size > pos:
                    continue
            except FileNotFoundError:
                pass
            return
        await asyncio.sleep(0.5)


@router.get("/runs/{run_id}/events")
def run_events(run_id: str, request: Request, db: DB, user: CurrentUser) -> EventSourceResponse:
    run = get_run(db, user, run_id)
    header = request.headers.get("last-event-id", "")
    last_id = int(header) if header.isdigit() else 0
    events_path = RunPaths(Path(run.run_dir)).events
    return EventSourceResponse(
        _follow_events(request, request.app.state.engine, run.id, events_path, last_id),
        ping=15,
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


# ------------------------------------------------------------------ results
@router.get("/runs/{run_id}/redactions", response_model=RedactionSummary)
def redactions(run_id: str, db: DB, user: CurrentUser) -> RedactionSummary:
    run = get_run(db, user, run_id)
    manifest = load_manifest(run)
    status = manifest.stages["sanitize"].status if manifest else "pending"
    paths = RunPaths(Path(run.run_dir))
    if status not in ("completed", "awaiting_approval") or not paths.stage("sanitize").exists():
        raise ApiError(404, "E_NOT_READY", "The privacy filter has not finished yet.")
    return artefacts.redaction_summary(paths)


@router.get("/runs/{run_id}/trials", response_model=list[TrialOut])
def trials(run_id: str, db: DB, user: CurrentUser) -> list[TrialOut]:
    run = get_run(db, user, run_id)
    return artefacts.trials(RunPaths(Path(run.run_dir)))


@router.get("/runs/{run_id}/scorecard", response_model=Scorecard)
def scorecard(run_id: str, db: DB, user: CurrentUser) -> Scorecard:
    run = get_run(db, user, run_id)
    manifest = load_manifest(run)
    paths = RunPaths(Path(run.run_dir))
    done = manifest is not None and manifest.stages["evaluate"].status == "completed"
    if not done or not artefacts.scorecard_ready(paths):
        raise ApiError(404, "E_NOT_READY", "The quality checks have not finished yet.")
    card = artefacts.scorecard(run.id, paths)
    card.adapter_id = run_adapter_id(db, run.id)
    return card


def _report(run_id: str, db: Session, user: Any, name: str, media_type: str) -> FileResponse:
    run = get_run(db, user, run_id)
    path = RunPaths(Path(run.run_dir)).stage("package") / name
    if not path.is_file():
        raise ApiError(404, "E_NOT_READY", "The report is created when the run has finished.")
    return FileResponse(path, media_type=media_type, filename=f"zllf-{run.id}-{name}")


@router.get("/runs/{run_id}/report.pdf")
def report_pdf(run_id: str, db: DB, user: CurrentUser) -> FileResponse:
    return _report(run_id, db, user, "report.pdf", "application/pdf")


@router.get("/runs/{run_id}/report.json")
def report_json(run_id: str, db: DB, user: CurrentUser) -> FileResponse:
    return _report(run_id, db, user, "report.json", "application/json")
