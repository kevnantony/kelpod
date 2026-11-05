"""Run directory layout and the atomic, resumable per-stage manifest (NFR-REL-05)."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from zllf.core.hashing import sha256_dir

STAGE_ORDER: tuple[str, ...] = ("ingest", "sanitize", "synth", "train", "evaluate", "package")
STAGE_DIRS: dict[str, str] = {
    "ingest": "01_ingest",
    "sanitize": "02_sanitize",
    "synth": "03_synth",
    "train": "04_train",
    "evaluate": "05_eval",
    "package": "06_package",
}

StageStatus = Literal["pending", "running", "completed", "failed", "awaiting_approval"]


class StageEntry(BaseModel):
    status: StageStatus = "pending"
    started_at: float | None = None
    finished_at: float | None = None
    input_hash: str | None = None
    output_hash: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class Manifest(BaseModel):
    run_id: str
    created_at: float = Field(default_factory=time.time)
    seed: int
    config: dict[str, Any]
    corpus_dir: str
    approved: bool = False  # redaction policy approved by the user
    stages: dict[str, StageEntry] = Field(
        default_factory=lambda: {name: StageEntry() for name in STAGE_ORDER}
    )


class RunPaths:
    """All paths inside ``runs/<run_id>/``."""

    def __init__(self, run_dir: Path) -> None:
        self.root = run_dir

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def events(self) -> Path:
        return self.logs / "events.jsonl"

    def stage(self, name: str) -> Path:
        """Committed output directory of a stage (read inputs from here)."""
        return self.root / STAGE_DIRS[name]

    def work(self, name: str) -> Path:
        """Persistent scratch for a stage (checkpoints survive crashes; removed on success)."""
        path = self.root / f"{STAGE_DIRS[name]}.work"
        path.mkdir(parents=True, exist_ok=True)
        return path


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_manifest(paths: RunPaths, manifest: Manifest) -> None:
    _atomic_write_text(paths.manifest, manifest.model_dump_json(indent=2))


def read_manifest(paths: RunPaths) -> Manifest:
    return Manifest.model_validate_json(paths.manifest.read_text(encoding="utf-8"))


def stage_input_hash(paths: RunPaths, stage: str, config_hash: str) -> str:
    """Inputs of a stage = config + committed outputs of all earlier stages."""
    from zllf.core.hashing import sha256_json

    parts: dict[str, Any] = {"config": config_hash}
    for name in STAGE_ORDER[: STAGE_ORDER.index(stage)]:
        out = paths.stage(name)
        parts[name] = sha256_dir(out) if out.exists() else None
    return sha256_json(parts)


@contextmanager
def stage_output(paths: RunPaths, stage: str) -> Iterator[Path]:
    """Yield a temp dir; on success atomically replace ``paths.stage(stage)`` with it.

    A crash inside the block leaves the previously committed output untouched.
    """
    final = paths.stage(stage)
    tmp = paths.root / f".tmp-{STAGE_DIRS[stage]}-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        yield tmp
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    old = paths.root / f".old-{STAGE_DIRS[stage]}-{os.getpid()}"
    if final.exists():
        os.replace(final, old)
    os.replace(tmp, final)
    shutil.rmtree(old, ignore_errors=True)


def append_event(paths: RunPaths, event: dict[str, Any]) -> None:
    paths.logs.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"ts": time.time(), **event}, default=str)
    with paths.events.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
