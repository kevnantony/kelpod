"""Paired bootstrap and the promotion rule of ``docs/EVAL_PROTOCOL.md`` §4 (rules 2-5).

Rule 1 (integrity) lives in ``scripts/promotion_check.sh``; rule 6 (budget) in
``scripts/locked_eval.py`` because it needs the test log.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

N_RESAMPLES = 10_000
RATE_TOL = 0.01  # -1 pp
SCORE_TOL = 0.01  # -0.01 IoU / F1
ERROR_TOL = 0.05  # +5 % relative
RUNTIME_TOL = 0.25  # +25 % relative (median)
P95_LIMIT_S = 180.0
GAIN_CC_PP = 0.03  # +3 pp correct-closed
GAIN_EDIT_REL = -0.10  # -10 % edit cost
GAIN_IOU = 0.02  # +0.02 room IoU
GENERALISATION_SPLITS = ("test-synth", "test-ood", "fresh")


@dataclass
class Delta:
    """Candidate minus baseline over paired plans (``relative``: stat(c) / stat(b) - 1)."""

    metric: str
    n: int
    base: float | None
    cand: float | None
    delta: float | None
    lo: float | None
    hi: float | None
    relative: bool = False
    stat: str = "mean"

    def to_dict(self) -> dict:
        return asdict(self)


def paired_bootstrap(
    cand: list[float | None],
    base: list[float | None],
    *,
    metric: str = "",
    relative: bool = False,
    stat: str = "mean",
    n_resamples: int = N_RESAMPLES,
    seed: int = 0,
    alpha: float = 0.05,
) -> Delta:
    """95 % percentile CI of the paired difference, resampling plans with replacement.

    Plans where either side is None are dropped (the metric is undefined there). For a
    relative delta whose baseline statistic is 0 the CI is of the absolute difference and
    ``relative`` is reported False.
    """
    if len(cand) != len(base):
        raise ValueError("paired samples must have the same length")
    pairs = [(c, b) for c, b in zip(cand, base, strict=True) if c is not None and b is not None]
    if not pairs:
        return Delta(metric, 0, None, None, None, None, None, relative, stat)
    c = np.array([p[0] for p in pairs], float)
    b = np.array([p[1] for p in pairs], float)
    f = np.median if stat == "median" else np.mean
    bc, bb = float(f(c)), float(f(b))
    if relative and bb == 0.0:
        relative = False
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(c), size=(n_resamples, len(c)))
    sc, sb = f(c[idx], axis=1), f(b[idx], axis=1)
    if relative:
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(sb > 0, sc / sb - 1.0, np.where(sc > 0, np.inf, 0.0))
        point = bc / bb - 1.0
    else:
        d = sc - sb
        point = bc - bb
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    return Delta(metric, len(c), bb, bc, float(point), float(lo), float(hi), relative, stat)


def compare_metric(
    metric: str,
    direction: int,
    kind: str,
    cand: list[float | None],
    base: list[float | None],
    *,
    seed: int = 0,
    n_resamples: int = N_RESAMPLES,
) -> Delta:
    """Bootstrap with the statistic the rule needs: absolute mean for rates and scores,
    relative mean for errors and edit cost, relative median for runtime."""
    if kind == "runtime":
        return paired_bootstrap(
            cand, base, metric=metric, relative=True, stat="median", seed=seed,
            n_resamples=n_resamples,
        )  # fmt: skip
    return paired_bootstrap(
        cand, base, metric=metric, relative=kind == "error", seed=seed, n_resamples=n_resamples
    )


def regression(d: Delta, direction: int, kind: str) -> str | None:
    """Rule 3: a message if the CI of Δ lies entirely beyond the tolerance, else None."""
    if d.lo is None or kind == "info":
        return None
    if not d.relative and kind in ("error", "runtime"):
        # baseline statistic was 0: any CI entirely above 0 is a regression
        if d.lo > 0:
            return f"{d.metric}: was 0 on the baseline, CI of Δ [{d.lo:+.3g}, {d.hi:+.3g}] > 0"
        return None
    if kind in ("rate", "score"):
        tol = RATE_TOL if kind == "rate" else SCORE_TOL
        if direction > 0 and d.hi < -tol:
            return f"{d.metric}: CI of Δ [{d.lo:+.4f}, {d.hi:+.4f}] entirely below -{tol}"
        if direction < 0 and d.lo > tol:
            return f"{d.metric}: CI of Δ [{d.lo:+.4f}, {d.hi:+.4f}] entirely above +{tol}"
        return None
    tol = RUNTIME_TOL if kind == "runtime" else ERROR_TOL
    if d.lo > tol:
        return f"{d.metric}: CI of relative Δ [{d.lo:+.1%}, {d.hi:+.1%}] entirely above +{tol:.0%}"
    return None


@dataclass
class RuleResult:
    rule: str
    passed: bool | None
    """None = not applicable / report only."""
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Decision:
    rules: list[RuleResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed is not False for r in self.rules) and any(
            r.rule.startswith("2") and r.passed for r in self.rules
        )

    def to_dict(self) -> dict:
        return {"passed": self.passed, "rules": [r.to_dict() for r in self.rules]}


def _fmt_ci(d: Delta, pct: bool = False) -> str:
    if d.delta is None:
        return "n/a"
    if pct or d.relative:
        return f"{d.delta:+.1%} [{d.lo:+.1%}, {d.hi:+.1%}]"
    return f"{d.delta:+.4f} [{d.lo:+.4f}, {d.hi:+.4f}]"


def gain_on_primary(primary: dict[str, Delta]) -> tuple[bool, str, str | None]:
    """Rule 2 on the primary split. Returns (passed, detail, criterion that passed)."""
    cc, ec, iou = primary.get("correct_closed"), primary.get("edit_cost"), primary.get("room_iou")
    parts = []
    if cc is not None and cc.delta is not None:
        ok = cc.lo > 0 and cc.delta >= GAIN_CC_PP
        parts.append(f"Δ correct-closed {_fmt_ci(cc)} (need lo > 0 and Δ >= +3 pp)")
        if ok:
            return True, "; ".join(parts), "correct_closed"
    if ec is not None and ec.delta is not None and ec.relative:
        ok = ec.delta <= GAIN_EDIT_REL and ec.hi < 0
        parts.append(f"Δ edit cost {_fmt_ci(ec)} (alt: <= -10 %, CI < 0)")
        if ok:
            return True, "; ".join(parts), "edit_cost"
    if iou is not None and iou.delta is not None:
        ok = iou.delta >= GAIN_IOU and iou.lo > 0
        parts.append(f"Δ room IoU {_fmt_ci(iou)} (alt: >= +0.02, CI > 0)")
        if ok:
            return True, "; ".join(parts), "room_iou"
    return False, "; ".join(parts) or "no primary data", None


def decide(
    deltas: dict[str, dict[str, Delta]],
    kinds: dict[str, dict[str, tuple[int, str]]],
    *,
    primary: str,
    p95_cand: dict[str, float | None],
    families: dict[str, str],
    test_real_pages: int = 0,
) -> Decision:
    """Rules 2-4 over the per-split deltas.

    ``deltas[split][metric]``; ``kinds[split][metric] = (direction, kind)``; ``primary`` is the
    key of the pooled primary split in ``deltas``; ``families[split]`` maps each split to
    ``test-synth``, ``test-ood``, ``fresh``, ``test-real`` or ``val`` (rule 4).
    """
    dec = Decision()
    ok, detail, criterion = gain_on_primary(deltas.get(primary, {}))
    dec.rules.append(RuleResult("2 real gain on primary", ok, f"{primary}: {detail}"))

    regs = []
    for split, ds in deltas.items():
        if split == primary:
            continue
        for m, d in ds.items():
            direction, kind = kinds[split][m]
            msg = regression(d, direction, kind)
            if msg:
                regs.append(f"{split}: {msg}")
        p95 = p95_cand.get(split)
        if p95 is not None and p95 >= P95_LIMIT_S:
            regs.append(f"{split}: runtime p95 {p95:.1f} s >= {P95_LIMIT_S:.0f} s")
    dec.rules.append(
        RuleResult("3 no hidden regression", not regs, "; ".join(regs) or "none on any split")
    )

    crit = criterion or "correct_closed"
    sign = -1 if crit == "edit_cost" else 1
    fam_gain: dict[str, bool] = {}
    for split, ds in deltas.items():
        fam = families.get(split)
        if split == primary or fam not in GENERALISATION_SPLITS or crit not in ds:
            continue
        d = ds[crit]
        if d.delta is not None:
            fam_gain[fam] = fam_gain.get(fam, False) or sign * d.delta > 0
    n_gain = sum(fam_gain.values())
    detail = ", ".join(f"{k}: {'gain' if v else 'no gain'}" for k, v in sorted(fam_gain.items()))
    if test_real_pages >= 40 and not any(f == "test-real" for f in families.values()):
        detail += " (test-real exists but was not run)"
    dec.rules.append(
        RuleResult(
            "4 generalises",
            n_gain >= 2,
            f"Δ {crit} > 0 on {n_gain} of {len(fam_gain)} available families "
            f"({detail or 'none'}); need 2",
        )
    )
    return dec
