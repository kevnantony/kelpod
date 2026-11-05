"""Sequential, resumable pipeline runner (used by the CLI and the worker).

Resume rule: a stage is skipped when its manifest entry is ``completed`` and its
input hash is unchanged; otherwise it and all later stages re-run.
"""

from __future__ import annotations

import time
import traceback
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from zllf.core.config import RunConfig
from zllf.core.errors import CancelledError, ZLLFError
from zllf.core.hashing import sha256_json
from zllf.core.logging import get_logger
from zllf.core.manifest import (
    STAGE_ORDER,
    Manifest,
    RunPaths,
    read_manifest,
    stage_input_hash,
    stage_output,
    write_manifest,
)
from zllf.stages import get_stage
from zllf.stages.base import EventSink, RunContext

Outcome = Literal["completed", "awaiting_approval", "failed", "cancelled"]


def new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


def init_run(runs_dir: Path, corpus_dir: Path, config: RunConfig, run_id: str | None = None,
             ) -> RunPaths:
    run_id = run_id or new_run_id()
    paths = RunPaths(runs_dir / run_id)
    paths.root.mkdir(parents=True, exist_ok=False)
    manifest = Manifest(
        run_id=run_id,
        seed=config.seed,
        config=config.model_dump(mode="json"),
        corpus_dir=str(corpus_dir.resolve()),
    )
    write_manifest(paths, manifest)
    return paths


def approve(paths: RunPaths) -> None:
    manifest = read_manifest(paths)
    manifest.approved = True
    if manifest.stages["sanitize"].status == "awaiting_approval":
        manifest.stages["sanitize"].status = "completed"
    write_manifest(paths, manifest)


def run_pipeline(
    paths: RunPaths,
    *,
    until: str | None = None,
    cancel_requested: Callable[[], bool] = lambda: False,
    event_sink: EventSink | None = None,
) -> Outcome:
    """Run (or resume) all stages up to and including ``until``."""
    manifest = read_manifest(paths)
    config = RunConfig.model_validate(manifest.config)
    config_hash = sha256_json(manifest.config)
    ctx = RunContext(
        run_id=manifest.run_id,
        paths=paths,
        corpus_dir=Path(manifest.corpus_dir),
        config=config,
        cancel_requested=cancel_requested,
        event_sink=event_sink,
    )
    log = get_logger("zllf.pipeline", run_id=manifest.run_id)
    stages = STAGE_ORDER[: STAGE_ORDER.index(until) + 1] if until else STAGE_ORDER
    invalidated = False

    for name in stages:
        entry = manifest.stages[name]
        input_hash = stage_input_hash(paths, name, config_hash)
        if name == "synth" and not manifest.approved:
            entry_s = manifest.stages["sanitize"]
            entry_s.status = "awaiting_approval"
            write_manifest(paths, manifest)
            ctx.emit("awaiting_approval", stage="sanitize")
            return "awaiting_approval"
        if (
            not invalidated
            and entry.status in ("completed", "awaiting_approval")
            and entry.input_hash == input_hash
            and paths.stage(name).exists()
        ):
            continue
        invalidated = True
        entry.status = "running"
        entry.started_at = time.time()
        entry.error = None
        write_manifest(paths, manifest)
        ctx.emit("stage_started", stage=name)
        log.info("stage_started", stage=name)
        try:
            ctx.check_cancelled()
            stage = get_stage(name)
            with stage_output(paths, name) as out_dir:
                result = stage.run(ctx, out_dir)
        except CancelledError:
            entry.status = "failed"
            entry.error = "cancelled"
            write_manifest(paths, manifest)
            ctx.emit("cancelled", stage=name)
            return "cancelled"
        except Exception as exc:
            entry.status = "failed"
            entry.error = exc.code.value if isinstance(exc, ZLLFError) else type(exc).__name__
            entry.finished_at = time.time()
            write_manifest(paths, manifest)
            ctx.emit("stage_failed", stage=name, error=entry.error, detail=str(exc)[:500])
            log.error("stage_failed", stage=name, error=entry.error,
                      trace=traceback.format_exc(limit=8))
            return "failed"
        from zllf.core.hashing import sha256_dir

        entry.status = "completed"
        entry.finished_at = time.time()
        entry.input_hash = input_hash
        entry.output_hash = sha256_dir(paths.stage(name))
        entry.metrics = result.metrics
        write_manifest(paths, manifest)
        ctx.emit("stage_completed", stage=name, metrics=result.metrics)
        log.info("stage_completed", stage=name)
        if result.needs_approval and not manifest.approved:
            entry.status = "awaiting_approval"
            write_manifest(paths, manifest)
            ctx.emit("awaiting_approval", stage=name)
            return "awaiting_approval"
    return "completed"
