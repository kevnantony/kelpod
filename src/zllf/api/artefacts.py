"""Readers for stage artefacts in a run directory (redactions, trials, scorecard)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from zllf.api.schemas import (
    GateOut,
    RedactionSample,
    RedactionSummary,
    Scorecard,
    ScorecardExample,
    Span,
    TrialOut,
)
from zllf.core.logging import get_logger
from zllf.core.manifest import RunPaths

log = get_logger("zllf.api.artefacts")
VARIANTS = ("base", "base_rag", "adapter", "adapter_rag")
MAX_SAMPLES = 20
MAX_EXAMPLES = 10


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------ redactions
def redaction_summary(paths: RunPaths) -> RedactionSummary:
    stage_dir = paths.stage("sanitize")
    try:  # helper owned by the sanitize stage, when available
        from zllf.stages.sanitize.preview import redaction_preview

        return RedactionSummary.model_validate(redaction_preview(paths, n_samples=MAX_SAMPLES))
    except ImportError:
        pass
    except Exception as exc:  # incompatible helper -> local fallback
        log.warning("redaction_preview_failed", error=type(exc).__name__)
    return _local_redaction_summary(stage_dir, paths)


def _local_redaction_summary(stage_dir: Path, paths: RunPaths) -> RedactionSummary:
    from zllf.core.schemas import Chunk, Redaction, read_parquet

    redactions: list[Redaction] = []
    if (stage_dir / "redactions.parquet").exists():
        redactions = read_parquet(stage_dir / "redactions.parquet", Redaction)
    clean: list[Chunk] = []
    if (stage_dir / "chunks_clean.parquet").exists():
        clean = read_parquet(stage_dir / "chunks_clean.parquet", Chunk)
    report = _read_json(stage_dir / "sanitize_report.json") or {}

    counts: dict[str, int] = defaultdict(int)
    by_chunk: dict[str, list[Redaction]] = defaultdict(list)
    dropped: set[str] = set()
    for r in redactions:
        counts[r.entity_type] += 1
        by_chunk[r.chunk_id].append(r)
        if r.action == "drop_chunk":
            dropped.add(r.chunk_id)
    total = report.get("chunks_total") or report.get("chunks_in")
    if not isinstance(total, int):
        total = len(clean) + len(dropped)
    n_dropped = report.get("chunks_dropped")
    if not isinstance(n_dropped, int):
        n_dropped = len(dropped)

    samples: list[RedactionSample] = []
    for chunk in sorted(clean, key=lambda c: c.chunk_id):
        if len(samples) >= MAX_SAMPLES:
            break
        reds = sorted((r for r in by_chunk.get(chunk.chunk_id, []) if r.action == "mask"),
                      key=lambda r: r.start)
        spans = _placeholder_spans(chunk.text, reds)
        if spans:
            samples.append(RedactionSample(chunk_id=chunk.chunk_id, text_redacted=chunk.text,
                                           spans=spans))
    return RedactionSummary(counts_by_type=dict(counts), chunks_dropped=n_dropped,
                            chunks_total=total, samples=samples)


def _placeholder_spans(masked_text: str, reds: list[Any]) -> list[Span]:
    """Locate each redaction's placeholder in the (already masked) text, in order."""
    spans: list[Span] = []
    pos = 0
    for r in reds:
        token = r.placeholder or f"<{r.entity_type}>"
        idx = masked_text.find(token, pos)
        if idx < 0:
            continue
        spans.append(Span(start=idx, end=idx + len(token), entity_type=r.entity_type,
                          action=r.action))
        pos = idx + len(token)
    return spans


# ------------------------------------------------------------------ trials
def trials(paths: RunPaths) -> list[TrialOut]:
    candidates = [paths.stage("train") / "trials.json",
                  paths.root / "04_train.work" / "trials.json"]
    for path in candidates:
        data = _read_json(path)
        if isinstance(data, list):
            out: list[TrialOut] = []
            for row in data:
                if not isinstance(row, dict):
                    continue
                metrics = {k: float(v) for k, v in (row.get("metrics") or {}).items()
                           if isinstance(v, int | float)}
                out.append(TrialOut(
                    number=int(row.get("number", len(out))), state=str(row.get("state", "")),
                    objective=row.get("objective"), params=dict(row.get("params") or {}),
                    metrics=metrics, seconds=float(row.get("seconds") or 0.0),
                ))
            return out
    return []


# ------------------------------------------------------------------ scorecard
def scorecard_ready(paths: RunPaths) -> bool:
    return (paths.stage("evaluate") / "gates.json").exists()


def _numeric(d: Any) -> dict[str, float]:
    if not isinstance(d, dict):
        return {}
    return {str(k): float(v) for k, v in d.items()
            if isinstance(v, int | float) and not isinstance(v, bool)}


def _flatten_suites(variant: Any) -> dict[str, float]:
    """``{suite: {metric: {"value", "ci": [lo, hi], "n"}}}`` -> ``{"suite.metric": value, …}``.

    CI bounds are added as ``"suite.metric.ci_low"`` / ``".ci_high"``. Flat numeric dicts
    (``{metric: value}``) pass through unchanged.
    """
    out: dict[str, float] = {}
    if not isinstance(variant, dict):
        return out
    for suite, metrics in variant.items():
        if isinstance(metrics, int | float) and not isinstance(metrics, bool):
            out[str(suite)] = float(metrics)
            continue
        if not isinstance(metrics, dict):
            continue
        for metric, val in metrics.items():
            key = f"{suite}.{metric}"
            if isinstance(val, dict):
                if isinstance(val.get("value"), int | float):
                    out[key] = float(val["value"])
                ci = val.get("ci")
                if isinstance(ci, list | tuple) and len(ci) == 2 and all(
                        isinstance(x, int | float) for x in ci):
                    out[f"{key}.ci_low"], out[f"{key}.ci_high"] = float(ci[0]), float(ci[1])
            elif isinstance(val, int | float) and not isinstance(val, bool):
                out[key] = float(val)
    return out


def _variant_metrics(results: Any) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {v: {} for v in VARIANTS}
    if not isinstance(results, dict):
        return out
    for container in (results.get("variants"), results.get("metrics"), results):
        if isinstance(container, dict) and any(v in container for v in VARIANTS):
            for v in VARIANTS:
                out[v] = _flatten_suites(container.get(v))
            return out
    return out


def _last_user(messages: Any) -> str:
    if isinstance(messages, list):
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                return str(m.get("content", ""))
    return ""


def _examples(paths: RunPaths, results: Any) -> list[ScorecardExample]:
    if isinstance(results, dict) and isinstance(results.get("examples"), list):
        out = []
        for ex in results["examples"][:MAX_EXAMPLES]:
            try:
                out.append(ScorecardExample.model_validate(ex))
            except ValueError:
                continue
        if out:
            return out
    preds_path = paths.stage("evaluate") / "predictions.jsonl"
    if not preds_path.exists():
        return []
    by_item: dict[str, dict[str, Any]] = defaultdict(dict)
    order: list[str] = []
    with preds_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                p = json.loads(line)
            except ValueError:
                continue
            if p.get("suite") not in ("domain_test", "gold"):
                continue
            item = p.get("item_id")
            if item not in by_item:
                order.append(item)
            by_item[item][p.get("variant")] = p.get("output", "")
    # prompts: evaluated items (05_eval/items.jsonl), else the synthetic test split
    prompts: dict[str, tuple[str, str | None]] = {}
    for path, id_key, msg_key, ref_key in (
        (paths.stage("evaluate") / "items.jsonl", "item_id", "prompt", "reference"),
        (paths.stage("synth") / "test.jsonl", "example_id", "messages", "reference_answer"),
    ):
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                key = row.get(id_key)
                if key in by_item and key not in prompts:
                    prompts[key] = (_last_user(row.get(msg_key)), row.get(ref_key))
    out = []
    for item in order:
        outputs = by_item[item]
        if "base" not in outputs or "adapter" not in outputs:
            continue
        prompt, reference = prompts.get(item, ("", None))
        if not prompt:
            continue
        out.append(ScorecardExample(prompt=prompt, base=outputs["base"],
                                    adapter=outputs["adapter"], reference=reference))
        if len(out) >= MAX_EXAMPLES:
            break
    return out


def scorecard(run_id: str, paths: RunPaths) -> Scorecard:
    gates_raw = _read_json(paths.stage("evaluate") / "gates.json")
    gates = []
    for g in gates_raw if isinstance(gates_raw, list) else []:
        try:
            gates.append(GateOut.model_validate({"details": {}, **g}))
        except ValueError:
            continue
    results = _read_json(paths.stage("evaluate") / "results.json")
    return Scorecard(
        run_id=run_id,
        gates=gates,
        all_passed=bool(gates) and all(g.passed for g in gates),
        metrics=_variant_metrics(results),
        examples=_examples(paths, results),
    )


def gates_passed(paths: RunPaths) -> bool:
    data = _read_json(paths.stage("evaluate") / "gates.json")
    return isinstance(data, list) and bool(data) and all(
        isinstance(g, dict) and g.get("passed") is True for g in data
    )
