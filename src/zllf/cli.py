"""``zllf`` command line (Typer).

Exit codes for ``run``/``resume``: 0 completed, 2 awaiting approval, 1 failed, 130 cancelled.
"""

from __future__ import annotations

import json
import signal
import sys
import threading
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from zllf.core.manifest import STAGE_ORDER, RunPaths

app = typer.Typer(name="zllf", help="Zero-Leak LoRA Factory", no_args_is_help=True,
                  add_completion=False)
console = Console(stderr=True)

EXIT_CODES = {"completed": 0, "awaiting_approval": 2, "failed": 1, "cancelled": 130}
Mode = Annotated[str, typer.Option("--mode", help="Goal profile: A (Q&A), B (style), C (extract)")]


def _settings() -> Any:
    from zllf.api.settings import get_settings

    return get_settings()


class _Display:
    """Rich progress bars fed by pipeline events (no document text is ever printed)."""

    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet
        self.progress = Progress(SpinnerColumn(), TextColumn("{task.description:<10}"),
                                 BarColumn(), TextColumn("{task.fields[status]}"),
                                 TimeElapsedColumn(), console=console, disable=quiet)
        self.tasks: dict[str, Any] = {}

    def __enter__(self) -> _Display:
        self.progress.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.progress.stop()

    def _task(self, stage: str) -> Any:
        if stage not in self.tasks:
            self.tasks[stage] = self.progress.add_task(stage, total=None, status="")
        return self.tasks[stage]

    def __call__(self, event: dict[str, Any]) -> None:
        etype, stage = event.get("type"), event.get("stage")
        if not stage:
            return
        task = self._task(str(stage))
        if etype == "stage_started":
            self.progress.update(task, status="running")
        elif etype == "progress" and event.get("total"):
            self.progress.update(task, total=event["total"], completed=event.get("done", 0),
                                 status=str(event.get("message") or "")[:40])
        elif etype == "stage_completed":
            total = self.progress.tasks[task].total or 1
            self.progress.update(task, total=total, completed=total, status="done")
        elif etype == "stage_failed":
            self.progress.update(task, status=f"[red]failed: {event.get('error')}")
        elif etype == "awaiting_approval":
            self.progress.update(task, status="[yellow]awaiting approval")
        elif etype == "cancelled":
            self.progress.update(task, status="[red]cancelled")


def _drive(paths: RunPaths, *, until: str | None, auto_approve: bool, quiet: bool) -> str:
    """Run/resume the pipeline; with auto_approve, approve and continue after sanitize."""
    from zllf.pipeline import approve, run_pipeline

    cancelled = threading.Event()
    previous = signal.getsignal(signal.SIGINT)

    def _on_sigint(_sig: int, _frame: Any) -> None:
        if cancelled.is_set():
            raise KeyboardInterrupt
        console.print("[yellow]Cancelling after the current step… (Ctrl-C again to abort)")
        cancelled.set()

    in_main = threading.current_thread() is threading.main_thread()
    if in_main:
        signal.signal(signal.SIGINT, _on_sigint)
    try:
        with _Display(quiet) as display:
            outcome = run_pipeline(paths, until=until, cancel_requested=cancelled.is_set,
                                   event_sink=display)
            if outcome == "awaiting_approval" and auto_approve:
                approve(paths)
                outcome = run_pipeline(paths, until=until, cancel_requested=cancelled.is_set,
                                       event_sink=display)
    finally:
        if in_main:
            signal.signal(signal.SIGINT, previous)
    return outcome


def _report_outcome(paths: RunPaths, outcome: str) -> None:
    msg = {
        "completed": "[green]Run completed.",
        "awaiting_approval": "[yellow]Waiting for redaction approval. Review the redaction "
        f"summary, then run: zllf approve {paths.root} && zllf resume {paths.root}",
        "failed": "[red]Run failed. Fix the cause, then: zllf resume " + str(paths.root),
        "cancelled": "[red]Run cancelled. Resume with: zllf resume " + str(paths.root),
    }[outcome]
    console.print(msg)
    typer.echo(json.dumps({"run_dir": str(paths.root), "outcome": outcome}))


def _check_until(until: str | None) -> None:
    if until is not None and until not in STAGE_ORDER:
        raise typer.BadParameter(f"must be one of {', '.join(STAGE_ORDER)}", param_hint="--until")


@app.command()
def run(
    corpus: Annotated[Path, typer.Option("--corpus", exists=True, file_okay=False,
                                         help="Folder with the documents")],
    mode: Mode = "A",
    hardware: Annotated[str | None, typer.Option(help="fake | toy | mac (default ZLLF_HARDWARE)")]
    = None,
    runs_dir: Annotated[Path | None, typer.Option("--runs-dir")] = None,
    auto_approve: Annotated[bool, typer.Option("--auto-approve",
                                               help="Approve redactions without review")] = False,
    until: Annotated[str | None, typer.Option(help="Stop after this stage")] = None,
    json_schema: Annotated[Path | None, typer.Option("--json-schema", exists=True,
                                                     dir_okay=False, help="Mode C schema")] = None,
    budget_seconds: Annotated[int | None, typer.Option("--budget-seconds")] = None,
    seed: Annotated[int | None, typer.Option()] = None,
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
) -> None:
    """Run the pipeline on a folder, standalone (no database or API needed)."""
    from zllf.core.config import load_config
    from zllf.pipeline import init_run

    _check_until(until)
    mode = mode.upper()
    if mode not in ("A", "B", "C"):
        raise typer.BadParameter("must be A, B or C", param_hint="--mode")
    settings = _settings()
    overrides: dict[str, Any] = {}
    if json_schema is not None:
        overrides["json_schema"] = json.loads(json_schema.read_text(encoding="utf-8"))
    if budget_seconds is not None:
        overrides["budget"] = {"total_seconds": budget_seconds}
    if seed is not None:
        overrides["seed"] = seed
    config = load_config(profile=f"mode_{mode.lower()}", hardware=hardware or settings.hardware,
                         overrides=overrides)
    runs = (runs_dir or settings.runs_dir).expanduser()
    runs.mkdir(parents=True, exist_ok=True)
    paths = init_run(runs, corpus, config)
    console.print(f"Run [bold]{paths.root.name}[/] → {paths.root}")
    outcome = _drive(paths, until=until, auto_approve=auto_approve, quiet=quiet)
    _report_outcome(paths, outcome)
    raise typer.Exit(EXIT_CODES[outcome])


def _run_paths(run_dir: Path) -> RunPaths:
    paths = RunPaths(run_dir)
    if not paths.manifest.exists():
        raise typer.BadParameter(f"{run_dir} has no manifest.json", param_hint="RUN_DIR")
    return paths


@app.command()
def resume(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    auto_approve: Annotated[bool, typer.Option("--auto-approve")] = False,
    until: Annotated[str | None, typer.Option()] = None,
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
) -> None:
    """Resume a run from its last good stage."""
    _check_until(until)
    paths = _run_paths(run_dir)
    outcome = _drive(paths, until=until, auto_approve=auto_approve, quiet=quiet)
    _report_outcome(paths, outcome)
    raise typer.Exit(EXIT_CODES[outcome])


@app.command("approve")
def approve_cmd(run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)]) -> None:
    """Approve the redaction policy of a run waiting after sanitize."""
    from zllf.pipeline import approve

    paths = _run_paths(run_dir)
    approve(paths)
    console.print(f"[green]Approved.[/] Continue with: zllf resume {run_dir}")


@app.command()
def status(run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
           as_json: Annotated[bool, typer.Option("--json")] = False) -> None:
    """Show per-stage status of a run directory."""
    from zllf.core.manifest import read_manifest

    manifest = read_manifest(_run_paths(run_dir))
    if as_json:
        typer.echo(json.dumps({"run_id": manifest.run_id, "approved": manifest.approved,
                               "stages": {k: v.model_dump() for k, v in manifest.stages.items()}},
                              default=str))
        return
    table = Table(title=f"Run {manifest.run_id}")
    for col_name in ("stage", "status", "seconds", "error"):
        table.add_column(col_name)
    for name in STAGE_ORDER:
        e = manifest.stages[name]
        secs = f"{e.finished_at - e.started_at:.1f}" if e.started_at and e.finished_at else ""
        table.add_row(name, e.status, secs, e.error or "")
    Console().print(table)


@app.command("serve-api")
def serve_api(
    host: Annotated[str | None, typer.Option(help="Default 127.0.0.1 (ZLLF_HOST)")] = None,
    port: Annotated[int | None, typer.Option(help="Default 8000 (ZLLF_PORT)")] = None,
    with_worker: Annotated[bool, typer.Option("--with-worker",
                                              help="Also run the worker in-process")] = False,
) -> None:
    """Start the HTTP API + web UI."""
    import uvicorn

    from zllf.api.app import create_app
    from zllf.core.logging import configure_logging

    configure_logging()
    settings = _settings()
    application = create_app(settings, start_worker=with_worker)
    uvicorn.run(application, host=host or settings.host, port=port or settings.port,
                log_level="warning", access_log=False, server_header=False)


@app.command()
def worker(once: Annotated[bool, typer.Option("--once", help="Process one run and exit")]
           = False) -> None:
    """Start the background worker (claims queued runs from the database)."""
    from zllf.db.engine import init_db
    from zllf.worker.loop import Worker

    settings = _settings()
    settings.ensure_dirs()
    w = Worker(settings, init_db(settings.db_url))
    if once:
        processed = w.run_once()
        w.close()
        raise typer.Exit(0 if processed else 3)

    def _stop(_sig: int, _frame: Any) -> None:
        console.print("[yellow]Stopping worker after the current step…")
        w.stop()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    w.run_forever()


@app.command("verify-package")
def verify_package(
    package_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    pubkey: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
) -> None:
    """Verify the signature and hashes of a packaged adapter (06_package)."""
    try:
        from zllf.stages.package.signing import verify_package_cli
    except ImportError:
        console.print("[red]Package verification is not available in this build.")
        raise typer.Exit(1) from None
    raise typer.Exit(verify_package_cli(package_dir, pubkey))


@app.command()
def hardware(name: Annotated[str | None, typer.Option("--profile",
                                                      help="Hardware config name")] = None
             ) -> None:
    """Print the detected hardware profile as JSON."""
    from zllf.api.hardware import hardware_profile

    settings = _settings()
    profile = hardware_profile(name or settings.hardware, settings.root)
    typer.echo(profile.model_dump_json(indent=2))


@app.command("create-admin")
def create_admin(
    username: Annotated[str, typer.Option(prompt=True)],
    password: Annotated[str, typer.Option(prompt=True, hide_input=True,
                                          confirmation_prompt=True)],
) -> None:
    """Create (or reset the password of) an administrator account."""
    from sqlmodel import select

    from zllf.api.security import hash_password
    from zllf.db.engine import audit, init_db, session_scope
    from zllf.db.models import User

    settings = _settings()
    if len(password) < settings.min_password_length:
        console.print(f"[red]Password must have at least {settings.min_password_length} "
                      "characters.")
        raise typer.Exit(1)
    settings.ensure_dirs()
    with session_scope(init_db(settings.db_url)) as db:
        user = db.exec(select(User).where(User.username == username.strip())).first()
        if user is None:
            user = User(username=username.strip(), password_hash=hash_password(password),
                        role="admin")
            action = "user.create_admin"
        else:
            user.password_hash, user.role, user.disabled = hash_password(password), "admin", False
            action = "user.reset_admin"
        db.add(user)
        db.flush()
        audit(db, "cli", action, "user", user.id)
    console.print(f"[green]Administrator {username!r} is ready.")


def main() -> None:  # pragma: no cover
    app(sys.argv[1:])


if __name__ == "__main__":  # pragma: no cover
    main()
