"""``report.json``: the immutable audit report of a run (FR-09, P2.7.1).

Built only from committed artefacts of earlier stages + the run manifest. Contains
counts, hashes, parameters and metrics — never document text or redacted values
(NFR-SEC-06). Validated against ``report.schema.json`` before it is written.
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import sys
from functools import cache
from importlib import metadata
from pathlib import Path
from typing import Any

import jsonschema

import zllf
from zllf.core.config import REPO_ROOT, RunConfig
from zllf.core.errors import ErrorCode, ZLLFError
from zllf.core.hashing import sha256_dir, sha256_file, sha256_json
from zllf.core.manifest import STAGE_DIRS, STAGE_ORDER, RunPaths
from zllf.stages.package import artefacts as A

SCHEMA_VERSION = "1.0"
SCHEMA_PATH = Path(__file__).with_name("report.schema.json")
ADAPTER_SIZE_LIMIT_BYTES = 500 * 1024 * 1024  # FR-08
DEFAULT_MODELS_MANIFEST = REPO_ROOT / "models" / "manifest.json"
DEFAULT_BENCH_DIR = REPO_ROOT / "bench"

GATE_LABELS: dict[str, str] = {
    "G1_domain": "Accuracy on your documents",
    "G2_forgetting": "Keeps general skills",
    "G3_hallucination": "Admits when it doesn't know",
    "G4_memorization": "Doesn't leak private data",
    "G5_schema": "Correct output format",
}

KEY_LIBRARIES: tuple[str, ...] = (
    "torch", "transformers", "peft", "mlx", "mlx-lm", "vllm", "sentence-transformers",
    "optuna", "docling", "presidio-analyzer", "spacy", "numpy", "pandas", "pyarrow",
    "pydantic", "huggingface-hub", "fpdf2", "matplotlib", "cyclonedx-python-lib",
    "cryptography",
)

# Every artefact earlier stages are expected to write (CLAUDE.md contract table).
EXPECTED_ARTEFACTS: dict[str, tuple[str, ...]] = {
    "ingest": ("documents.parquet", "pages.parquet", "chunks.parquet", "ingest_report.json"),
    "sanitize": ("chunks_clean.parquet", "redactions.parquet", "pii_fingerprints.json",
                 "policy.yaml", "sanitize_report.json"),
    "synth": ("train.jsonl", "val.jsonl", "test.jsonl", "canaries.json", "taxonomy.json",
              "synth_report.json"),
    "train": ("hw_profile.json", "trials.json", "adapter", "train_report.json"),
    "evaluate": ("predictions.jsonl", "results.json", "gates.json", "eval_report.json"),
}


@cache
def load_schema() -> dict[str, Any]:
    schema: dict[str, Any] = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return schema


def validate_report(report: dict[str, Any]) -> None:
    """Raise ``ZLLFError(PACKAGE_FAILED)`` if ``report`` violates the JSON Schema."""
    try:
        jsonschema.validate(report, load_schema())
    except jsonschema.ValidationError as exc:
        where = "/".join(str(p) for p in exc.absolute_path) or "<root>"
        raise ZLLFError(ErrorCode.PACKAGE_FAILED,
                        f"report.json invalid at {where}: {exc.message[:200]}") from exc


# ---------------------------------------------------------------- sections
def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return dt.datetime.fromtimestamp(ts, tz=dt.UTC).isoformat(timespec="seconds")


def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def versions_section() -> dict[str, Any]:
    return {
        "zllf": zllf.__version__,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "byteorder": sys.byteorder,
        },
        "libraries": {name: _dist_version(name) for name in KEY_LIBRARIES},
    }


def inputs_section(paths: RunPaths) -> dict[str, Any]:
    docs = A.load_documents(paths)
    if docs is None:
        return {"present": False}
    by_status: dict[str, int] = {}
    for d in docs:
        by_status[str(d["status"])] = by_status.get(str(d["status"]), 0) + 1
    ingest_report = A.read_json(paths.stage("ingest") / "ingest_report.json")
    return {
        "present": True,
        "n_documents": len(docs),
        "by_status": dict(sorted(by_status.items())),
        "n_pages": sum(int(d.get("n_pages") or 0) for d in docs),
        "n_ocr_pages": sum(int(d.get("n_ocr_pages") or 0) for d in docs),
        "n_chars": sum(int(d.get("n_chars") or 0) for d in docs),
        "corpus_sha256": sha256_json(sorted(str(d["sha256"]) for d in docs)),
        "documents": docs,
        "report": A.numeric_only(ingest_report),
    }


def redaction_section(paths: RunPaths) -> dict[str, Any]:
    counts = A.redaction_counts(paths)
    report = A.numeric_only(A.read_json(paths.stage("sanitize") / "sanitize_report.json"))
    if counts is None and report is None:
        return {"present": False}
    return {"present": True, "counts": counts, "report": report}


def synthesis_section(paths: RunPaths) -> dict[str, Any]:
    splits = A.synth_counts(paths)
    report = A.numeric_only(A.read_json(paths.stage("synth") / "synth_report.json"))
    if splits is None and report is None:
        return {"present": False}
    canaries = A.read_json(paths.stage("synth") / "canaries.json")
    files = {}
    for name in A.SPLIT_FILES:
        path = paths.stage("synth") / name
        if path.exists():
            files[name] = sha256_file(path)
    return {
        "present": True,
        "splits": splits,
        "n_canaries": len(canaries) if isinstance(canaries, list) else None,
        "files_sha256": files,
        "report": report,
    }


def training_section(paths: RunPaths) -> dict[str, Any]:
    train_dir = paths.stage("train")
    hw = A.load_hw_profile(paths)
    trials = A.load_trials(paths)
    train_report = A.read_json(train_dir / "train_report.json")
    if hw is None and trials is None and train_report is None:
        return {"present": False}
    return {
        "present": True,
        "hardware_profile": hw.model_dump(mode="json") if hw else None,
        "n_trials": len(trials) if trials is not None else None,
        "trials": [t.model_dump(mode="json") for t in trials or []],
        "final_params": A.final_params(train_report, trials),
        "loss_curves": A.loss_curves(train_report),
        "report": A.numeric_only(train_report),
    }


def evaluation_section(paths: RunPaths) -> dict[str, Any]:
    gates = A.load_gates(paths) or []
    results = A.read_json(paths.stage("evaluate") / "results.json")
    metrics = A.normalize_metrics(results)
    return {
        "present": True,
        "ready": A.overall_ready(gates),
        "gates": [
            {
                "gate": g.gate,
                "label": GATE_LABELS.get(g.gate, g.gate),
                "passed": g.passed,
                "value": g.value,
                "threshold": g.threshold,
                "details": A.numeric_only(g.details) or {},
            }
            for g in gates
        ],
        "metrics": {v: {m: mv.to_json() for m, mv in ms.items()} for v, ms in metrics.items()},
        "results_present": results is not None,
        "report": A.numeric_only(A.read_json(paths.stage("evaluate") / "eval_report.json")),
    }


def timings_section(paths: RunPaths) -> dict[str, Any]:
    manifest = A.load_manifest(paths)
    out: dict[str, Any] = {}
    for name in STAGE_ORDER:
        entry = manifest.stages.get(name) if manifest else None
        if entry is None:
            out[name] = {"status": "unknown", "started_at": None, "finished_at": None,
                         "seconds": None}
            continue
        seconds = None
        if entry.started_at is not None and entry.finished_at is not None:
            seconds = round(max(entry.finished_at - entry.started_at, 0.0), 3)
        out[name] = {"status": entry.status, "started_at": _iso(entry.started_at),
                     "finished_at": _iso(entry.finished_at), "seconds": seconds}
    return out


def adapter_section(adapter: Path) -> dict[str, Any]:
    files = A.dir_files(adapter)
    size = sum(p.stat().st_size for p in files)
    return {
        "path": "adapter",
        "sha256": sha256_dir(adapter),
        "size_bytes": size,
        "n_files": len(files),
        "within_size_limit": size < ADAPTER_SIZE_LIMIT_BYTES,
        "files": [{"path": p.relative_to(adapter).as_posix(), "sha256": sha256_file(p),
                   "size_bytes": p.stat().st_size} for p in files],
    }


def model_roles(config: RunConfig) -> dict[str, str]:
    """role -> model id for every model the run used."""
    m = config.models
    return {"student": m.student, "teacher": m.teacher, "judge": m.judge or m.teacher,
            "nli": m.nli, "embedder": m.embedder}


def load_models_manifest(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """``models/manifest.json`` entries by model name (empty if absent)."""
    data = A.read_json(path or DEFAULT_MODELS_MANIFEST)
    entries = data.get("models") if isinstance(data, dict) else None
    out: dict[str, dict[str, Any]] = {}
    for entry in entries or []:
        if isinstance(entry, dict) and isinstance(entry.get("name"), str):
            out[entry["name"]] = entry
    return out


def bench_datasets(bench_dir: Path | None = None) -> dict[str, Path]:
    """Replay/anchor benchmark directories that exist and contain files."""
    bench_dir = bench_dir or DEFAULT_BENCH_DIR
    return {name: bench_dir / name for name in ("replay", "anchor")
            if A.dir_files(bench_dir / name)}


def _dataset_license(directory: Path) -> str | None:
    data = A.read_json(directory / "manifest.json")
    if isinstance(data, dict):
        if isinstance(data.get("license"), str):
            return str(data["license"])
        lic = {str(e["license"]) for e in data.get("datasets", []) or []
               if isinstance(e, dict) and isinstance(e.get("license"), str)}
        if lic:
            return " AND ".join(sorted(lic))
    return None


def licenses_section(config: RunConfig, models_manifest: dict[str, dict[str, Any]],
                     bench: dict[str, Path]) -> dict[str, Any]:
    models: dict[str, dict[str, Any]] = {}
    for role, name in model_roles(config).items():
        entry = models.setdefault(name, {"name": name, "roles": [], "license": None,
                                         "revision": None, "source": "unknown"})
        entry["roles"].append(role)
        if name.startswith("fake"):
            entry["source"] = "fake-test-model"
        elif (info := models_manifest.get(name)) is not None:
            entry.update(license=info.get("license"), revision=info.get("revision"),
                         source="models/manifest.json")
    datasets = [{"name": f"bench/{name}", "license": _dataset_license(path),
                 "source": "bench"} for name, path in sorted(bench.items())]
    datasets.append({"name": "synthetic", "license": "proprietary (derived from customer "
                     "documents)", "source": "03_synth"})
    return {"models": sorted(models.values(), key=lambda e: e["name"]), "datasets": datasets}


def artefacts_section(paths: RunPaths) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for stage, names in EXPECTED_ARTEFACTS.items():
        for name in names:
            path = paths.stage(stage) / name
            out[f"{STAGE_DIRS[stage]}/{name}"] = (
                bool(A.dir_files(path)) if name == "adapter" else path.is_file())
    return out


# ---------------------------------------------------------------- assembly
def build_report(
    paths: RunPaths,
    config: RunConfig,
    *,
    run_id: str | None = None,
    signing_key_id: str | None = None,
    workspace: str | None = None,
    models_manifest: Path | None = None,
    bench_dir: Path | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Assemble the audit report. Raises ``PACKAGE_FAILED`` if adapter/gates are missing."""
    A.require_inputs(paths)
    manifest = A.load_manifest(paths)
    config_json = manifest.config if manifest else config.model_dump(mode="json")
    adapter = A.adapter_dir(paths)
    evaluation = evaluation_section(paths)
    run_created = manifest.created_at if manifest else None
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id or (manifest.run_id if manifest else paths.root.name),
        "workspace": workspace,
        "created_at": _iso(now if now is not None else dt.datetime.now(dt.UTC).timestamp()),
        "run_created_at": _iso(run_created),
        "mode": config.mode,
        "profile": config.profile_name,
        "hardware": config.hardware_name,
        "ready": evaluation["ready"],
        "versions": versions_section(),
        "config": config_json,
        "config_sha256": sha256_json(config_json),
        "seeds": {"seed": manifest.seed if manifest else config.seed,
                  "derivation": "per-stage RNG = sha256(f'{seed}:{salt}')"},
        "inputs": inputs_section(paths),
        "redaction": redaction_section(paths),
        "synthesis": synthesis_section(paths),
        "training": training_section(paths),
        "evaluation": evaluation,
        "timings": timings_section(paths),
        "adapter": adapter_section(adapter),
        "licenses": licenses_section(config, load_models_manifest(models_manifest),
                                     bench_datasets(bench_dir)),
        "artefacts": artefacts_section(paths),
        "signing": {"algorithm": "Ed25519", "key_id": signing_key_id,
                    "manifest": "SHA256SUMS", "signature": "SHA256SUMS.sig"},
    }
    return report


def write_report(report: dict[str, Any], path: Path) -> None:
    """Validate against the schema, then write pretty, key-sorted JSON."""
    validate_report(report)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8")
