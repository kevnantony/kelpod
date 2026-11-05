"""Worker: claiming/leases, crash recovery + resume, shutdown re-queue."""

from __future__ import annotations

import os
import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from sqlmodel import Session
from zllf_testkit import CALLS, Behaviour, install_dummy_stages, make_settings

from zllf.core.config import load_config
from zllf.core.manifest import RunPaths, read_manifest, write_manifest
from zllf.core.offline import block_egress
from zllf.db.engine import init_db, session_scope
from zllf.db.models import Run, User, Workspace
from zllf.pipeline import init_run, run_pipeline
from zllf.worker.loop import Worker


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    install_dummy_stages(monkeypatch)
    settings = make_settings(tmp_path)
    settings.ensure_dirs()
    engine = init_db(settings.db_url)
    corpus = settings.corpus_dir("ws1")
    corpus.mkdir(parents=True)
    (corpus / "a.txt").write_text("hello")
    with session_scope(engine) as db:
        db.add(User(id="u1", username="u", password_hash="x", role="admin"))
        db.add(Workspace(id="ws1", name="W", base_model="fake-student", created_by="u1"))
    with block_egress():
        yield settings, engine


def _new_run(settings: Any, engine: Any, **fields: Any) -> tuple[str, RunPaths]:
    config = load_config(hardware="fake")
    paths = init_run(settings.runs_dir, settings.corpus_dir("ws1"), config)
    with session_scope(engine) as db:
        db.add(Run(id=paths.root.name, workspace_id="ws1", mode="A", profile="mode_a",
                   hardware="fake", run_dir=str(paths.root), **fields))
    return paths.root.name, paths


def _state(engine: Any, run_id: str) -> Run:
    with Session(engine) as db:
        run = db.get(Run, run_id)
        assert run is not None
        return run


def test_claim_is_exclusive(env: Any) -> None:
    settings, engine = env
    run_id, _ = _new_run(settings, engine)
    a, b = Worker(settings, engine, worker_id="a"), Worker(settings, engine, worker_id="b")
    assert a.claim_next() == run_id
    assert b.claim_next() is None
    run = _state(engine, run_id)
    assert run.state == "running" and run.lease_owner == "a" and run.lease_expires_at


def test_crash_resume_skips_completed_stages(env: Any) -> None:
    settings, engine = env
    dead_owner = f"{socket.gethostname()}:{2**22 - 7}:1"  # pid that does not exist
    run_id, paths = _new_run(settings, engine, state="running", lease_owner=dead_owner,
                             lease_expires_at=time.time() + 3600, started_at=time.time())
    # simulate a crash in the middle of sanitize: ingest committed, sanitize "running"
    assert run_pipeline(paths, until="ingest") == "completed"
    manifest = read_manifest(paths)
    manifest.stages["sanitize"].status = "running"
    write_manifest(paths, manifest)
    CALLS.clear()

    worker = Worker(settings, engine)
    assert worker.recover_stale() == 1
    assert _state(engine, run_id).state == "queued"
    assert worker.run_once()
    assert _state(engine, run_id).state == "awaiting_approval"
    assert CALLS["ingest"] == 0 and CALLS["sanitize"] == 1


def test_live_lease_is_not_stolen(env: Any) -> None:
    settings, engine = env
    owner = f"{socket.gethostname()}:{os.getpid()}:other"
    run_id, _ = _new_run(settings, engine, state="running", lease_owner=owner,
                         lease_expires_at=time.time() + 3600)
    assert Worker(settings, engine).recover_stale() == 0
    assert _state(engine, run_id).state == "running"


def test_shutdown_requeues_running_run(env: Any) -> None:
    settings, engine = env
    run_id, _ = _new_run(settings, engine)
    Behaviour.block_stage = "ingest"
    worker = Worker(settings, engine)
    t = threading.Thread(target=worker.run_once)
    t.start()
    assert Behaviour.started.wait(10)
    worker.stop()  # SIGTERM path
    t.join(10)
    worker.close()
    run = _state(engine, run_id)
    assert run.state == "queued" and run.lease_owner is None
    Behaviour.block_stage = None
    worker2 = Worker(settings, engine)
    assert worker2.run_once()
    assert _state(engine, run_id).state == "awaiting_approval"


def test_failed_run_records_error_code(env: Any) -> None:
    settings, engine = env
    run_id, _ = _new_run(settings, engine)
    Behaviour.fail_stage = "ingest"
    Worker(settings, engine).run_once()
    run = _state(engine, run_id)
    assert run.state == "failed" and run.error_code == "E_TRAIN_FAILED"
    assert run.finished_at is not None


def test_run_forever_stops_on_signal(env: Any) -> None:
    settings, engine = env
    run_id, _ = _new_run(settings, engine)
    worker = Worker(settings, engine)
    t = threading.Thread(target=worker.run_forever)
    t.start()
    for _ in range(200):
        if _state(engine, run_id).state == "awaiting_approval":
            break
        time.sleep(0.02)
    worker.stop()
    t.join(5)
    assert not t.is_alive()
    assert _state(engine, run_id).state == "awaiting_approval"
