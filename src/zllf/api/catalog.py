"""Model catalogue: models referenced by configs/hardware/*.yaml + local HF cache presence."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml

from zllf.api.schemas import ModelInfo
from zllf.api.settings import Settings

ModelRole = Literal["student", "teacher", "aux"]
_ROLE_KEYS: tuple[tuple[str, ModelRole], ...] = (
    ("student", "student"), ("teacher", "teacher"), ("judge", "teacher"),
              ("nli", "aux"), ("embedder", "aux"))


def hf_cache_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    home = os.environ.get("HF_HOME")
    return (Path(home).expanduser() if home else Path.home() / ".cache" / "huggingface") / "hub"


def _snapshot(model_id: str) -> Path | None:
    repo = hf_cache_dir() / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if not repo.is_dir():
        return None
    snaps = sorted((p for p in repo.iterdir() if p.is_dir() and any(p.iterdir())),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return snaps[0] if snaps else None


def _license(snapshot: Path | None) -> str | None:
    if snapshot is None:
        return None
    readme = snapshot / "README.md"
    try:
        text = readme.read_text(encoding="utf-8", errors="ignore")[:20000]
    except OSError:
        return None
    match = re.match(r"^---\s*\n(.*?)\n---", text, re.S)
    if not match:
        return None
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return None
    lic = meta.get("license") if isinstance(meta, dict) else None
    return str(lic) if lic else None


def list_models(settings: Settings) -> list[ModelInfo]:
    """Every model named in a hardware profile. Fake models are listed only on ``fake``."""
    seen: dict[str, ModelInfo] = {}
    for path in sorted((settings.config_dir / "hardware").glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        models = data.get("models") or {}
        fake = models.get("backend") == "fake"
        if fake and settings.hardware != "fake":
            continue
        for key, role in _ROLE_KEYS:
            model_id = models.get(key)
            if not model_id or model_id in seen:
                continue
            if fake or str(model_id).startswith("fake"):
                seen[model_id] = ModelInfo(id=model_id, role=role, license=None, installed=True)
                continue
            snap = _snapshot(model_id)
            seen[model_id] = ModelInfo(id=model_id, role=role, license=_license(snap),
                                       installed=snap is not None)
    order = {"student": 0, "teacher": 1, "aux": 2}
    return sorted(seen.values(), key=lambda m: (order[m.role], m.id))
