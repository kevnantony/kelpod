"""Tolerant readers for the artefacts of earlier stages (the run-directory contract).

Everything here is read-only and defensive: optional artefacts that are missing or
malformed come back as ``None``/empty, so the package stage can still produce an
audit report that records what was absent. Only :func:`require_inputs` raises.

Privacy (NFR-SEC-06): free-form report files written by other stages are reduced with
:func:`numeric_only` before they enter the audit report, so no stray text survives.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zllf.core.errors import ErrorCode, ZLLFError
from zllf.core.manifest import Manifest, RunPaths, read_manifest
from zllf.core.schemas import (
    Canary,
    EvalPrediction,
    GateResult,
    HardwareProfile,
    SynthExample,
    TrialResult,
)

VARIANTS: tuple[str, ...] = ("base", "base_rag", "adapter", "adapter_rag")
GATE_ORDER: tuple[str, ...] = (
    "G1_domain", "G2_forgetting", "G3_hallucination", "G4_memorization", "G5_schema",
)


# ---------------------------------------------------------------- generic helpers
def read_json(path: Path) -> Any | None:
    """Parse a JSON file; ``None`` if it is missing or unreadable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def numeric_only(obj: Any, *, max_depth: int = 6) -> Any:
    """Keep only numbers/bools (and dicts/lists of them). Strings are dropped.

    Used on free-form stage reports so the audit report can never carry document text.
    Returns ``None`` when nothing numeric is left.
    """
    if max_depth < 0:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int | float):
        return obj if not (isinstance(obj, float) and not math.isfinite(obj)) else None
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            kept = numeric_only(value, max_depth=max_depth - 1)
            if kept is not None and kept != {} and kept != []:
                out[str(key)] = kept
        return out or None
    if isinstance(obj, list | tuple):
        items = [numeric_only(v, max_depth=max_depth - 1) for v in obj]
        items = [v for v in items if v is not None]
        return items or None
    return None


def dir_files(path: Path) -> list[Path]:
    """All regular files below ``path`` in sorted order (empty if missing)."""
    if not path.is_dir():
        return []
    return sorted(p for p in path.rglob("*") if p.is_file())


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in dir_files(path))


def jsonl_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    yield obj


# ---------------------------------------------------------------- run-level
def load_manifest(paths: RunPaths) -> Manifest | None:
    try:
        return read_manifest(paths)
    except (OSError, ValueError):
        return None


def adapter_dir(paths: RunPaths) -> Path:
    return paths.stage("train") / "adapter"


def require_inputs(paths: RunPaths) -> None:
    """Raise ``PACKAGE_FAILED`` unless the adapter and the gate results exist."""
    adapter = adapter_dir(paths)
    if not dir_files(adapter):
        raise ZLLFError(ErrorCode.PACKAGE_FAILED, f"no trained adapter in {adapter}")
    gates = paths.stage("evaluate") / "gates.json"
    if not gates.exists():
        raise ZLLFError(ErrorCode.PACKAGE_FAILED, f"gate results missing: {gates}")
    if load_gates(paths) is None:
        raise ZLLFError(ErrorCode.PACKAGE_FAILED, f"gate results unreadable: {gates}")


def load_gates(paths: RunPaths) -> list[GateResult] | None:
    data = read_json(paths.stage("evaluate") / "gates.json")
    if isinstance(data, dict):  # tolerate {"gates": [...]}
        data = data.get("gates")
    if not isinstance(data, list):
        return None
    try:
        gates = [GateResult.model_validate(g) for g in data]
    except ValueError:
        return None
    return sorted(gates, key=lambda g: GATE_ORDER.index(g.gate))


def overall_ready(gates: list[GateResult] | None) -> bool:
    return bool(gates) and all(g.passed for g in gates or [])


# ---------------------------------------------------------------- 01_ingest
def load_documents(paths: RunPaths) -> list[dict[str, Any]] | None:
    """documents.parquet rows without the file path (file names may contain PII)."""
    path = paths.stage("ingest") / "documents.parquet"
    if not path.exists():
        return None
    import pandas as pd

    try:
        df = pd.read_parquet(path)
    except (OSError, ValueError):
        return None
    keep = ["doc_id", "sha256", "mime", "n_pages", "n_ocr_pages", "n_chars", "status",
            "language"]
    rows: list[dict[str, Any]] = []
    for rec in df.to_dict(orient="records"):
        row: dict[str, Any] = {}
        for key in keep:
            value = rec.get(key)
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, float) and value != value:
                value = None
            row[key] = value
        rows.append(row)
    return sorted(rows, key=lambda r: str(r.get("doc_id")))


# ---------------------------------------------------------------- 02_sanitize
def redaction_counts(paths: RunPaths) -> dict[str, Any] | None:
    """Counts by entity type and action from redactions.parquet (never values)."""
    path = paths.stage("sanitize") / "redactions.parquet"
    if not path.exists():
        return None
    import pandas as pd

    try:
        df = pd.read_parquet(path, columns=["chunk_id", "entity_type", "action"])
    except (OSError, ValueError, KeyError):
        return None
    by_type = {str(k): int(v) for k, v in df["entity_type"].value_counts().sort_index().items()}
    by_action = {str(k): int(v) for k, v in df["action"].value_counts().sort_index().items()}
    return {
        "n_redactions": len(df),
        "n_chunks_affected": int(df["chunk_id"].nunique()),
        "by_entity_type": by_type,
        "by_action": by_action,
    }


# ---------------------------------------------------------------- 03_synth
SPLIT_FILES: tuple[str, ...] = ("train.jsonl", "val.jsonl", "test.jsonl")


def synth_counts(paths: RunPaths) -> dict[str, Any] | None:
    """Per-split counts by task and answerability, computed from the JSONL files."""
    synth = paths.stage("synth")
    out: dict[str, Any] = {}
    for name in SPLIT_FILES:
        path = synth / name
        if not path.exists():
            continue
        by_task: dict[str, int] = {}
        n = n_unanswerable = 0
        for row in _iter_jsonl(path):
            n += 1
            task = str(row.get("task", "unknown"))
            by_task[task] = by_task.get(task, 0) + 1
            if row.get("answerable") is False:
                n_unanswerable += 1
        out[name.removesuffix(".jsonl")] = {
            "n": n, "n_unanswerable": n_unanswerable, "by_task": dict(sorted(by_task.items())),
        }
    return out or None


def load_test_examples(paths: RunPaths) -> dict[str, SynthExample]:
    path = paths.stage("synth") / "test.jsonl"
    out: dict[str, SynthExample] = {}
    if not path.exists():
        return out
    for row in _iter_jsonl(path):
        try:
            ex = SynthExample.model_validate(row)
        except ValueError:
            continue
        out[ex.example_id] = ex
    return out


def load_canary_secrets(paths: RunPaths) -> list[str]:
    data = read_json(paths.stage("synth") / "canaries.json")
    if not isinstance(data, list):
        return []
    secrets = []
    for row in data:
        try:
            secrets.append(Canary.model_validate(row).secret)
        except ValueError:
            continue
    return [s for s in secrets if s]


# ---------------------------------------------------------------- 04_train
def load_hw_profile(paths: RunPaths) -> HardwareProfile | None:
    data = read_json(paths.stage("train") / "hw_profile.json")
    try:
        return HardwareProfile.model_validate(data) if isinstance(data, dict) else None
    except ValueError:
        return None


def load_trials(paths: RunPaths) -> list[TrialResult] | None:
    data = read_json(paths.stage("train") / "trials.json")
    if isinstance(data, dict):
        data = data.get("trials")
    if not isinstance(data, list):
        return None
    trials = []
    for row in data:
        try:
            trials.append(TrialResult.model_validate(row))
        except ValueError:
            continue
    return sorted(trials, key=lambda t: t.number)


def final_params(train_report: Any, trials: list[TrialResult] | None) -> dict[str, Any] | None:
    """Final hyper-parameters: from train_report.json, else the best completed trial."""
    if isinstance(train_report, dict):
        for key in ("final_params", "best_params", "params"):
            value = train_report.get(key)
            if isinstance(value, dict):
                return {k: v for k, v in value.items()
                        if isinstance(v, int | float | str | bool) or v is None}
    done = [t for t in trials or [] if t.state == "complete" and t.objective is not None]
    if not done:
        return None
    best = min(done, key=lambda t: (t.objective, t.number))
    return best.params.model_dump(mode="json")


def _as_points(series: Any) -> list[list[float]] | None:
    """Normalise a curve to ``[[step, value], ...]``."""
    if not isinstance(series, list) or not series:
        return None
    points: list[list[float]] = []
    for i, item in enumerate(series):
        if isinstance(item, int | float) and not isinstance(item, bool):
            points.append([float(i), float(item)])
        elif isinstance(item, list | tuple) and len(item) >= 2:
            try:
                points.append([float(item[0]), float(item[1])])
            except (TypeError, ValueError):
                continue
        elif isinstance(item, dict):
            step = item.get("step", item.get("iteration", item.get("epoch", i)))
            value = next((item[k] for k in ("loss", "value", "train_loss", "val_loss")
                          if isinstance(item.get(k), int | float)), None)
            if value is not None and isinstance(step, int | float):
                points.append([float(step), float(value)])
    points = [p for p in points if math.isfinite(p[0]) and math.isfinite(p[1])]
    return points or None


def loss_curves(train_report: Any) -> dict[str, list[list[float]]]:
    """Extract loss curves from train_report.json in any of the shapes we accept.

    Accepted: ``{"loss_curves": {"train": [...], "val": [...]}}``, ``{"loss_curve": [...]}``,
    ``{"history": [{"step", "train_loss", "val_loss"}]}`` where series are lists of floats,
    ``[step, loss]`` pairs or ``{"step", "loss"}`` dicts.
    """
    out: dict[str, list[list[float]]] = {}
    if not isinstance(train_report, dict):
        return out
    curves = train_report.get("loss_curves")
    if isinstance(curves, dict):
        for name, series in curves.items():
            pts = _as_points(series)
            if pts:
                out[str(name)] = pts
    for key, name in (("loss_curve", "train"), ("train_loss", "train"), ("val_loss", "val")):
        pts = _as_points(train_report.get(key))
        if pts and name not in out:
            out[name] = pts
    history = train_report.get("history")
    if isinstance(history, list):
        for metric in ("train_loss", "val_loss", "loss"):
            series = [{"step": h.get("step", i), "loss": h[metric]}
                      for i, h in enumerate(history)
                      if isinstance(h, dict) and isinstance(h.get(metric), int | float)]
            pts = _as_points(series)
            name = "train" if metric in ("train_loss", "loss") else "val"
            if pts and name not in out:
                out[name] = pts
    return dict(sorted(out.items()))


# ---------------------------------------------------------------- 05_eval
@dataclass(frozen=True)
class MetricValue:
    value: float
    ci_low: float | None = None
    ci_high: float | None = None
    n: int | None = None

    def to_json(self) -> dict[str, Any]:
        return {"value": self.value, "ci_low": self.ci_low, "ci_high": self.ci_high, "n": self.n}


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _leaf(obj: Any) -> MetricValue | None:
    if (v := _num(obj)) is not None:
        return MetricValue(v)
    if not isinstance(obj, dict):
        return None
    value = next((_num(obj[k]) for k in ("value", "mean", "point", "score") if k in obj), None)
    if value is None:
        return None
    lo = hi = None
    for key in ("ci", "ci95", "ci_95"):
        ci = obj.get(key)
        if isinstance(ci, list | tuple) and len(ci) == 2:
            lo, hi = _num(ci[0]), _num(ci[1])
    for lo_key, hi_key in (("ci_low", "ci_high"), ("lo", "hi"), ("lower", "upper")):
        if lo_key in obj or hi_key in obj:
            lo, hi = _num(obj.get(lo_key)), _num(obj.get(hi_key))
    n = obj.get("n")
    return MetricValue(value, lo, hi, int(n) if isinstance(n, int) else None)


def normalize_metrics(results: Any) -> dict[str, dict[str, MetricValue]]:
    """Turn results.json into ``{variant: {metric_path: MetricValue}}``.

    The variant may appear at any nesting level (``variants.base.domain_test.rougeL``,
    ``rougeL.base`` …); the remaining keys form the metric path joined with ``/``.
    """
    out: dict[str, dict[str, MetricValue]] = {}

    def walk(obj: Any, path: tuple[str, ...], depth: int) -> None:
        if depth > 6:
            return
        leaf = _leaf(obj)
        variant = next((p for p in path if p in VARIANTS), None)
        if leaf is not None and variant is not None:
            rest = [p for p in path if p != variant and p not in ("variants", "metrics")]
            if rest:
                out.setdefault(variant, {})["/".join(rest)] = leaf
            return
        if isinstance(obj, dict):
            for key, value in obj.items():
                walk(value, (*path, str(key)), depth + 1)

    walk(results, (), 0)
    return {v: dict(sorted(out[v].items())) for v in VARIANTS if v in out}


@dataclass(frozen=True)
class Example:
    item_id: str
    suite: str
    question: str
    reference: str | None
    outputs: dict[str, str]


def load_examples(paths: RunPaths, limit: int = 6) -> list[Example]:
    """Side-by-side answers for the scorecard, from 05_eval predictions.

    Questions come from the synthetic test split (masked text). Canary prompts are
    never shown. Deterministic: domain items first, sorted by item id.
    """
    path = paths.stage("evaluate") / "predictions.jsonl"
    if not path.exists():
        return []
    by_item: dict[str, dict[str, EvalPrediction]] = {}
    for row in _iter_jsonl(path):
        try:
            pred = EvalPrediction.model_validate(row)
        except ValueError:
            continue
        if pred.suite == "canary" or pred.suite.startswith("anchor"):
            continue
        by_item.setdefault(pred.item_id, {})[pred.variant] = pred
    tests = load_test_examples(paths)
    order = {"domain_test": 0, "gold": 1, "unanswerable": 2}
    candidates = []
    for item_id, preds in by_item.items():
        ex = tests.get(item_id)
        if ex is None or "base" not in preds or "adapter" not in preds:
            continue
        question = next((m.content for m in ex.messages if m.role == "user"), None)
        if not question:
            continue
        suite = next(iter(preds.values())).suite
        candidates.append(Example(
            item_id=item_id, suite=suite, question=question, reference=ex.reference_answer,
            outputs={v: preds[v].output for v in VARIANTS if v in preds},
        ))
    candidates.sort(key=lambda e: (order.get(e.suite, 9), e.item_id))
    # prefer a mix: mostly domain, at least one unanswerable when available
    picked = [e for e in candidates if e.suite != "unanswerable"][: max(limit - 1, 1)]
    picked += [e for e in candidates if e.suite == "unanswerable"][: limit - len(picked)]
    return picked[:limit]
