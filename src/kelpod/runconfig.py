"""Run configuration: which segmentation backend and which checkpoint the pipeline uses.

The repository root holds ``kelpod-run.toml``::

    segment_backend = "learned"
    ocr_backend = "rapidocr"
    [[checkpoints]]              # one per training seed
    seed = 1
    path = "~/kelpod-models/kelpod-seg/v1-s1/model.onnx"
    sha256 = "..."

Checkpoint resolution for the learned backend (first match wins):

1. ``KELPOD_SEG_CHECKPOINT`` (a path; no hash check);
2. the ``[[checkpoints]]`` entry whose ``seed`` equals ``KELPOD_SEG_SEED``;
3. the first ``[[checkpoints]]`` entry.

There is **no silent fallback** to the classical segmenter: a missing config, a missing file or
a hash mismatch raises :class:`CheckpointError` with what to do. ``KELPOD_RUN_CONFIG`` points to
another config file.
"""

from __future__ import annotations

import hashlib
import os
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ENV_CONFIG = "KELPOD_RUN_CONFIG"
ENV_CHECKPOINT = "KELPOD_SEG_CHECKPOINT"
ENV_SEED = "KELPOD_SEG_SEED"
CONFIG_NAME = "kelpod-run.toml"
DEFAULT_SEGMENT_BACKEND = "learned"


class CheckpointError(RuntimeError):
    """The learned backend has no usable checkpoint."""


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    sha256: str | None
    seed: int | None


def config_path() -> Path:
    env = os.environ.get(ENV_CONFIG)
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parents[2] / CONFIG_NAME  # repo root (src layout)


def load_config() -> dict:
    p = config_path()
    if not p.is_file():
        return {}
    with open(p, "rb") as f:
        return tomllib.load(f)


def default_segment_backend() -> str:
    return str(load_config().get("segment_backend", DEFAULT_SEGMENT_BACKEND))


@lru_cache(maxsize=8)
def _sha256(path: str, mtime_ns: int, size: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def resolve_checkpoint() -> Checkpoint:
    env = os.environ.get(ENV_CHECKPOINT)
    if env:
        p = Path(env).expanduser()
        if not p.is_file():
            raise CheckpointError(f"{ENV_CHECKPOINT}={env} does not exist")
        return Checkpoint(p, None, None)
    cfg = load_config()
    entries = cfg.get("checkpoints") or []
    if not entries:
        raise CheckpointError(
            "the learned segmentation backend needs a checkpoint: set "
            f"{ENV_CHECKPOINT}=/path/to/model.onnx, or list [[checkpoints]] in "
            f"{config_path()} (see training/README.md); or choose segment_backend='classical' "
            "explicitly"
        )
    seed = os.environ.get(ENV_SEED)
    if seed is not None:
        match = [e for e in entries if str(e.get("seed")) == seed]
        if not match:
            raise CheckpointError(f"no checkpoint with seed {seed} in {config_path()}")
        entry = match[0]
    else:
        entry = entries[0]
    p = Path(str(entry["path"])).expanduser()
    if not p.is_absolute():
        p = config_path().parent / p
    if not p.is_file():
        raise CheckpointError(
            f"checkpoint {p} (from {config_path()}) not found; copy it there or set "
            f"{ENV_CHECKPOINT}"
        )
    want = entry.get("sha256")
    if want:
        st = p.stat()
        got = _sha256(str(p), st.st_mtime_ns, st.st_size)
        if got != want:
            raise CheckpointError(f"checkpoint {p} has sha256 {got}, {config_path()} expects {want}")
    return Checkpoint(p, want, int(entry["seed"]) if "seed" in entry else None)
