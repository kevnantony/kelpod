"""Adapter registry metadata (P1.7.1). Pure function over the run dir: the API owns the DB."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from zllf.core.errors import ErrorCode, ZLLFError
from zllf.core.hashing import sha256_dir
from zllf.core.manifest import RunPaths
from zllf.stages.package import artefacts as A
from zllf.stages.package.signing import SIG_NAME


def adapter_id_for(run_id: str) -> str:
    """Stable adapter id derived from the run id."""
    return f"adp-{run_id}"


def register_adapter(run_paths: RunPaths) -> dict[str, Any]:
    """Metadata for the adapter table: id, path, size, sha256, gate outcome, signature.

    Prefers the packaged adapter (``06_package/adapter``) and falls back to
    ``04_train/adapter``. Raises ``PACKAGE_FAILED`` if neither exists.
    """
    package_dir = run_paths.stage("package")
    adapter: Path = package_dir / "adapter"
    if not A.dir_files(adapter):
        adapter = A.adapter_dir(run_paths)
    files = A.dir_files(adapter)
    if not files:
        raise ZLLFError(ErrorCode.PACKAGE_FAILED, "no adapter to register")
    manifest = A.load_manifest(run_paths)
    run_id = manifest.run_id if manifest else run_paths.root.name
    gates = A.load_gates(run_paths) or []
    base_model = None
    if manifest:
        base_model = (manifest.config.get("models") or {}).get("student")
    packaged = (package_dir / "SHA256SUMS").exists()
    return {
        "adapter_id": adapter_id_for(run_id),
        "run_id": run_id,
        "path": str(adapter),
        "package_dir": str(package_dir) if packaged else None,
        "base_model": base_model,
        "size_bytes": sum(p.stat().st_size for p in files),
        "sha256": sha256_dir(adapter),
        "gates_passed": A.overall_ready(gates),
        "gates": {g.gate: g.passed for g in gates},
        "signed": (package_dir / SIG_NAME).exists(),
        "report_path": str(package_dir / "report.json") if packaged else None,
    }
