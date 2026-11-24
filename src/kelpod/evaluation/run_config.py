"""Declared run configuration of a worktree for locked evaluation (``kelpod-run.toml``).

A worktree's pipeline normally runs with its defaults. A candidate that needs something else (the
learned segmentation backend, a different interpreter, trained weights) declares it in a
versioned file at the repo root, so the locked run is reproducible from the commit alone::

    # kelpod-run.toml
    version = 1
    segment_backend = "learned"          # passed to kelpod.pipeline.convert
    ocr_backend = "rapidocr"             # passed to kelpod.pipeline.convert
    python_project = "training"          # uv project whose .venv runs the pipeline ("." = root)
    manifests = ["training/manifests/synth.json"]   # training data of the checkpoints

    [[checkpoints]]                      # one per training seed (EVAL_PROTOCOL §4 rule 5)
    path = "~/kelpod-models/kelpod-seg/r34-s0/model.pt"
    sha256 = "..."
    seed = 0                             # label only
    # manifests = [...]                  # optional, overrides the top-level list

An absent file means today's defaults (convert() defaults, the root project, no checkpoint).
Checkpoints live **outside** the worktree and outside every evaluation data directory; each is
verified by SHA-256, copied to a private read-only temp file and handed to the pipeline only
through ``KELPOD_SEG_CHECKPOINT``. The plan's temp dir still contains only the input file.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import subprocess
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

FILENAME = "kelpod-run.toml"
VERSION = 1
CHECKPOINT_ENV = "KELPOD_SEG_CHECKPOINT"
"""The only variable through which a checkpoint reaches the pipeline (kelpod.segment.learned)."""

_TOP_KEYS = {"version", "segment_backend", "ocr_backend", "python_project", "manifests",
             "checkpoints"}  # fmt: skip
_CKPT_KEYS = {"path", "sha256", "seed", "manifests"}
_BACKEND_RE = re.compile(r"^[A-Za-z0-9_.@-]{1,64}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class RunConfigError(ValueError):
    """The run config is malformed or violates an integrity rule: the run must not start."""


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    sha256: str
    seed: str
    manifests: tuple[str, ...] | None = None


@dataclass
class RunConfig:
    segment_backend: str | None = None
    ocr_backend: str | None = None
    python_project: str = "."
    manifests: tuple[str, ...] | None = None
    checkpoints: list[Checkpoint] = field(default_factory=list)
    source: str = "defaults (no kelpod-run.toml)"
    text: str = ""
    state: str = "absent"
    """git state of the file: committed, modified, untracked or absent (set by the caller)."""

    def runner_args(self) -> list[str]:
        """Extra arguments for ``locked_runner.py`` (none for the defaults, so a worktree without
        a config is called exactly as before)."""
        out = []
        if self.segment_backend:
            out += ["--segment-backend", self.segment_backend]
        if self.ocr_backend:
            out += ["--ocr-backend", self.ocr_backend]
        return out

    def summary(self) -> str:
        parts = [
            f"segment_backend={self.segment_backend or 'default'}",
            f"ocr_backend={self.ocr_backend or 'default'}",
            f"python_project={self.python_project}",
            f"checkpoints={len(self.checkpoints)}",
        ]
        return ", ".join(parts)


def _rel_paths(v, what: str) -> tuple[str, ...]:
    """Training manifests are part of the commit: relative paths inside the worktree only."""
    if not isinstance(v, list) or not all(isinstance(x, str) and x for x in v):
        raise RunConfigError(f"{what} must be a list of non-empty strings")
    for x in v:
        if Path(x).is_absolute() or x.startswith("~") or ".." in Path(x).parts:
            raise RunConfigError(f"{what}: {x!r} must be a relative path inside the worktree")
    return tuple(v)


def parse(text: str, *, source: str = FILENAME) -> RunConfig:
    """Parse and validate the TOML text (no file system checks; see :func:`check`)."""
    try:
        d = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise RunConfigError(f"{source}: invalid TOML: {exc}") from exc
    unknown = set(d) - _TOP_KEYS
    if unknown:
        raise RunConfigError(f"{source}: unknown keys {sorted(unknown)}")
    if d.get("version") != VERSION:
        raise RunConfigError(f"{source}: version must be {VERSION}")
    cfg = RunConfig(source=source, text=text)
    for key in ("segment_backend", "ocr_backend"):
        if key in d:
            v = d[key]
            if not isinstance(v, str) or not _BACKEND_RE.match(v):
                raise RunConfigError(f"{source}: {key} must match {_BACKEND_RE.pattern}")
            setattr(cfg, key, v)
    proj = d.get("python_project", ".")
    if not isinstance(proj, str):
        raise RunConfigError(f"{source}: python_project must be a string")
    p = Path(proj)
    if p.is_absolute() or ".." in p.parts:
        raise RunConfigError(
            f"{source}: python_project must be a relative path inside the worktree"
        )
    cfg.python_project = proj
    if "manifests" in d:
        cfg.manifests = _rel_paths(d["manifests"], f"{source}: manifests")
    ckpts = d.get("checkpoints", [])
    if not isinstance(ckpts, list):
        raise RunConfigError(f"{source}: checkpoints must be an array of tables")
    for k, c in enumerate(ckpts):
        if not isinstance(c, dict):
            raise RunConfigError(f"{source}: checkpoints[{k}] must be a table")
        unknown = set(c) - _CKPT_KEYS
        if unknown:
            raise RunConfigError(f"{source}: checkpoints[{k}]: unknown keys {sorted(unknown)}")
        if not isinstance(c.get("path"), str) or not c["path"]:
            raise RunConfigError(f"{source}: checkpoints[{k}].path is required")
        sha = c.get("sha256")
        if not isinstance(sha, str) or not _SHA_RE.match(sha):
            raise RunConfigError(f"{source}: checkpoints[{k}].sha256 must be 64 lowercase hex")
        man = c.get("manifests")
        cfg.checkpoints.append(
            Checkpoint(
                path=Path(os.path.expanduser(c["path"])),
                sha256=sha,
                seed=str(c.get("seed", k)),
                manifests=None if man is None else _rel_paths(man, f"checkpoints[{k}].manifests"),
            )
        )
    shas = [c.sha256 for c in cfg.checkpoints]
    if len(set(shas)) != len(shas):
        raise RunConfigError(
            f"{source}: two checkpoints have the same sha256; seeds must be distinct models"
        )
    return cfg


def git_state(wt: Path) -> str:
    """'committed', 'modified', 'untracked' or 'absent' for the worktree's run config."""
    f = wt / FILENAME
    if not f.exists():
        tracked = subprocess.run(
            ["git", "-C", str(wt), "ls-files", "--error-unmatch", FILENAME],
            capture_output=True, check=False,
        ).returncode == 0  # fmt: skip
        return "modified" if tracked else "absent"
    st = subprocess.run(
        ["git", "-C", str(wt), "status", "--porcelain", "--", FILENAME],
        capture_output=True, text=True, check=False,
    ).stdout.strip()  # fmt: skip
    if not st:
        return "committed"
    return "untracked" if st.startswith("??") else "modified"


def load(wt: Path) -> RunConfig:
    """The worktree's run config (defaults if there is no ``kelpod-run.toml``)."""
    f = wt / FILENAME
    if not f.is_file():
        return RunConfig()
    return parse(f.read_text(), source=str(f))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def forbidden_roots(manifests: dict[str, dict]) -> list[Path]:
    """Directories a checkpoint must never come from: the evaluation data root, the sealed
    directory and every split's data directory."""
    from kelpod.evaluation.splits import data_root, split_data_dir

    roots = [data_root(), Path.home() / "kelpod-data" / "sealed"]
    for m in manifests.values():
        if m.get("data_dir"):
            roots.append(split_data_dir(m))
    return [Path(os.path.realpath(r)) for r in roots]


def checkpoint_manifests(wt: Path, cfg: RunConfig, ck: Checkpoint) -> list[Path]:
    """Training manifests that describe ``ck``'s training data: the checkpoint's own list, else
    the config's list, else every ``training/manifests/*.json`` of the worktree."""
    rel = ck.manifests if ck.manifests is not None else cfg.manifests
    if rel is None:
        return sorted((wt / "training" / "manifests").glob("*.json"))
    return [wt / r for r in rel]


def check(
    wt: Path, cfg: RunConfig, manifests: dict[str, dict], *, other_worktrees: list[Path] = ()
) -> list[str]:
    """Integrity problems of a run config (empty = may run). Verifies every checkpoint's location
    and SHA-256 and the overlap of its training manifests with the test splits."""
    from kelpod.evaluation.splits import check_training_overlap

    problems = []
    proj = wt / cfg.python_project
    if not (proj / "pyproject.toml").is_file():
        problems.append(f"python_project {cfg.python_project!r}: no pyproject.toml in {proj}")
    roots = forbidden_roots(manifests)
    trees = [Path(os.path.realpath(t)) for t in (wt, *other_worktrees)]
    for ck in cfg.checkpoints:
        real = Path(os.path.realpath(ck.path))
        tag = f"checkpoint seed {ck.seed} ({ck.path})"
        if any(real.is_relative_to(t) for t in trees):
            problems.append(f"{tag}: must live outside the worktrees under evaluation")
        if any(real.is_relative_to(r) for r in roots):
            problems.append(f"{tag}: lies in an evaluation data or sealed directory")
        if not real.is_file():
            problems.append(f"{tag}: file not found")
            continue
        got = sha256_file(real)
        if got != ck.sha256:
            problems.append(f"{tag}: sha256 mismatch (declared {ck.sha256[:12]}, file {got[:12]})")
        mans = checkpoint_manifests(wt, cfg, ck)
        missing = [str(m) for m in mans if not m.is_file()]
        if missing:
            problems.append(f"{tag}: training manifest(s) not found: {', '.join(missing)}")
        if not mans:
            problems.append(f"{tag}: no training manifest; leak hygiene cannot be checked")
        problems += [
            f"{tag}: {p}" for p in check_training_overlap([m for m in mans if m.is_file()],
                                                          manifests)
        ]  # fmt: skip
    return problems


def stage_checkpoint(ck: Checkpoint, dest_dir: Path) -> Path:
    """Copy a verified checkpoint to a private read-only file (outside every plan's temp dir),
    re-checking the hash of the copy, so it cannot change while the run is going."""
    dest_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    dest = dest_dir / f"seed-{re.sub(r'[^A-Za-z0-9_.-]', '_', ck.seed)}{ck.path.suffix}"
    shutil.copyfile(os.path.realpath(ck.path), dest)
    os.chmod(dest, stat.S_IRUSR)
    got = sha256_file(dest)
    if got != ck.sha256:
        raise RunConfigError(f"checkpoint seed {ck.seed}: sha256 of the staged copy is {got[:12]}")
    return dest


def staging_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="kelpod-ckpt-"))
