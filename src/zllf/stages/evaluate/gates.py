"""Gate engine (P2.6.7): pure functions from per-item scores to :class:`GateResult`.

Thresholds come from ``config.gates``. ``details`` carry evidence item ids (capped).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from zllf.core.config import GatesConfig
from zllf.core.schemas import GateResult
from zllf.stages.evaluate.metrics import paired_relative_ci, relative_improvement

MAX_EVIDENCE = 50

# How to read GateResult.value against its threshold (also written to details["direction"]).
DIRECTION: dict[str, str] = {
    "G1_domain": "higher_is_better",        # relative improvement over base
    "G2_forgetting": "lower_is_better",     # worst per-suite drop in points
    "G3_hallucination": "higher_is_better",  # abstention rate
    "G4_memorization": "lower_is_better",   # canary extractions + raw-PII hits
    "G5_schema": "higher_is_better",        # schema-valid rate
}


def _gate(gate: str, *, passed: bool, value: float, threshold: float,
          details: dict[str, Any]) -> GateResult:
    return GateResult(gate=gate, passed=passed, value=value,  # type: ignore[arg-type]
                      threshold=threshold, details={"direction": DIRECTION[gate], **details})


def _cap(ids: Sequence[str]) -> list[str]:
    return list(ids)[:MAX_EVIDENCE]


def gate_domain(
    *,
    suite: str,
    metric: str,
    item_ids: Sequence[str],
    adapter: Sequence[float],
    base: Sequence[float],
    base_rag: Sequence[float] | None,
    gates: GatesConfig,
    resamples: int,
    rng: np.random.Generator,
) -> GateResult:
    """G-1: relative improvement of the adapter over base on the primary metric.

    Passes iff point estimate >= ``domain_rel_improvement_min`` AND the lower bound of the
    paired-bootstrap 95 % CI is > 0 (PRD §11.2.5). Comparison with base+RAG is recorded as
    a warning only (R2).
    """
    thr = gates.domain_rel_improvement_min
    if len(item_ids) == 0:
        return _gate("G1_domain", passed=False, value=0.0, threshold=thr,
                          details={"suite": suite, "metric": metric, "error": "no items"})
    point, lo, hi = paired_relative_ci(adapter, base, resamples, rng)
    a_mean, b_mean = float(np.mean(adapter)), float(np.mean(base))
    worse = [i for i, a, b in zip(item_ids, adapter, base, strict=True) if a < b]
    details: dict[str, object] = {
        "suite": suite, "metric": metric, "n": len(item_ids),
        "adapter_mean": a_mean, "base_mean": b_mean,
        "rel_improvement": point, "rel_ci_low": lo, "rel_ci_high": hi,
        "denominator_floor": 0.01,
        "items_adapter_worse_than_base": _cap(worse),
    }
    if base_rag is not None and len(base_rag) == len(item_ids):
        r_mean = float(np.mean(base_rag))
        rel_rag = relative_improvement(a_mean, r_mean)
        details["vs_base_rag"] = {
            "base_rag_mean": r_mean, "rel_improvement": rel_rag,
            "adapter_ge_rag": a_mean >= r_mean, "warning_only": True,
        }
        if a_mean < r_mean:
            details["warning"] = "adapter scores below the base+RAG baseline"
    passed = bool(point >= thr and lo > 0)
    return _gate("G1_domain", passed=passed, value=float(point), threshold=thr,
                      details=details)


@dataclass
class AnchorScores:
    item_ids: list[str]
    base: list[float]
    adapter: list[float]
    extra: dict[str, object] = field(default_factory=dict)


def gate_forgetting(per_suite: Mapping[str, AnchorScores], gates: GatesConfig) -> GateResult:
    """G-2: per anchor suite, drop = (base accuracy - adapter accuracy) in points; fail if any
    suite drops more than ``forgetting_max_drop_points``. Value = worst drop."""
    thr = gates.forgetting_max_drop_points
    if not per_suite:
        return _gate("G2_forgetting", passed=False, value=0.0, threshold=thr,
                          details={"error": "no anchor suites available"})
    suites: dict[str, object] = {}
    worst = -float("inf")
    failing: list[str] = []
    for name, s in sorted(per_suite.items()):
        b = 100.0 * float(np.mean(s.base)) if s.base else 0.0
        a = 100.0 * float(np.mean(s.adapter)) if s.adapter else 0.0
        drop = b - a
        worst = max(worst, drop)
        regressed = [i for i, x, y in zip(s.item_ids, s.base, s.adapter, strict=True) if y < x]
        suites[name] = {"n": len(s.item_ids), "base_points": b, "adapter_points": a,
                        "drop_points": drop, "items_regressed": _cap(regressed), **s.extra}
        if drop > thr:
            failing.append(name)
    return _gate("G2_forgetting", passed=not failing, value=float(worst),
                      threshold=thr, details={"suites": suites, "failing_suites": failing})


def gate_hallucination(item_ids: Sequence[str], labels: Sequence[str],
                       gates: GatesConfig, n_judge_fallback: int = 0) -> GateResult:
    """G-3: on unanswerable prompts, abstention rate >= ``abstention_min`` and fabrication
    rate <= ``fabrication_max``. Value = abstention rate."""
    thr = gates.abstention_min
    n = len(labels)
    if n == 0:
        return _gate("G3_hallucination", passed=False, value=0.0, threshold=thr,
                          details={"error": "no unanswerable items"})
    abst = sum(1 for x in labels if x == "abstain") / n
    fab = sum(1 for x in labels if x == "fabricate") / n
    fabricated = [i for i, x in zip(item_ids, labels, strict=True) if x == "fabricate"]
    hedged = [i for i, x in zip(item_ids, labels, strict=True) if x == "hedge"]
    passed = abst >= thr and fab <= gates.fabrication_max
    return _gate("G3_hallucination", passed=bool(passed), value=abst, threshold=thr,
                      details={"n": n, "abstention_rate": abst, "fabrication_rate": fab,
                               "fabrication_max": gates.fabrication_max,
                               "fabricated_items": _cap(fabricated),
                               "hedged_items": _cap(hedged),
                               "judge_fallbacks": n_judge_fallback})


def gate_memorization(
    *,
    n_canaries: int,
    extracted: Sequence[str],
    exposures: Mapping[str, Mapping[str, float]],
    pii_hit_items: Sequence[str],
    pii_status: str,
    gates: GatesConfig,
) -> GateResult:
    """G-4: canaries extracted <= ``canary_extractions_max`` and raw-PII hits <=
    ``raw_pii_hits_max``. Value = canary extractions + PII hits; threshold = sum of maxima.
    Exposure is reported only."""
    thr = float(gates.canary_extractions_max + gates.raw_pii_hits_max)
    ext = sorted(set(extracted))
    hits = list(pii_hit_items)
    passed = len(ext) <= gates.canary_extractions_max and len(hits) <= gates.raw_pii_hits_max
    passed = passed and not pii_status.startswith("error")  # fail closed on a broken scan
    adapter_exp = [e["adapter_exposure"] for e in exposures.values() if "adapter_exposure" in e]
    details: dict[str, object] = {
        "n_canaries": n_canaries, "canaries_extracted": ext,
        "canary_extractions_max": gates.canary_extractions_max,
        "exposure": dict(exposures),
        "max_adapter_exposure": max(adapter_exp) if adapter_exp else None,
        "pii_scan": pii_status, "pii_hit_items": _cap(hits), "pii_hits": len(hits),
        "raw_pii_hits_max": gates.raw_pii_hits_max,
    }
    if n_canaries == 0:
        details["warning"] = "no canaries were provided by the synth stage"
    if pii_status != "ok":
        details["warning_pii"] = f"raw-PII fingerprint scan not performed ({pii_status})"
    return _gate("G4_memorization", passed=bool(passed),
                      value=float(len(ext) + len(hits)), threshold=thr, details=details)


def gate_schema(item_ids: Sequence[str], valid: Sequence[bool], gates: GatesConfig,
                schema_available: bool = True) -> GateResult:
    """G-5 (Mode C): valid-JSON (schema-conformant) rate >= ``schema_valid_min``."""
    thr = gates.schema_valid_min
    if not valid:
        return _gate("G5_schema", passed=False, value=0.0, threshold=thr,
                          details={"error": "no extraction items"})
    rate = sum(valid) / len(valid)
    invalid = [i for i, v in zip(item_ids, valid, strict=True) if not v]
    details: dict[str, object] = {"n": len(valid), "invalid_items": _cap(invalid)}
    if not schema_available:
        details["warning"] = "config.json_schema missing: checked JSON object syntax only"
    return _gate("G5_schema", passed=rate >= thr, value=rate, threshold=thr,
                      details=details)
