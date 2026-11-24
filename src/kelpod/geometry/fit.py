"""Least-squares fit of the wall line coordinates to the dimension chains (Maßketten).

Every dimension label with a measured span (two tick positions) becomes one equation
``(X_j + o_j) - (X_i + o_i) = value`` where ``X`` are wall line coordinates and ``o`` the offset of
the feature the tick sits on (a wall face ``±t/2`` or the centreline ``0``). The ticks are matched
in the drawing's own geometry, so a plan that was drawn a few percent off still matches; the written
values then pull the walls to the right lengths. Weak priors keep unconstrained lines where they
were drawn. Contradicting (misread) labels are removed greedily, leave-one-out.

Both conventions (ticks on faces, ticks on centrelines) are tried; the one that explains more
labels (then: lower residual against the drawing as drawn) wins.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .lines import Line, Piece
from .params import GeometryParams


@dataclass
class DimM:
    axis: str  # "x": measures x (horizontal chain), "y": measures y (vertical chain)
    lo: float
    hi: float
    value: float
    text: str = ""


@dataclass
class FitResult:
    coords: list[float]
    mode: str | None
    n_used: int
    n_dims: int
    n_outliers: int
    residual_before_m: float | None
    residual_m: float | None


@dataclass
class _Eq:
    i: int
    oi: float
    j: int
    oj: float
    value: float


def _features(lines: list[Line], pieces: list[Piece], axis: str, mode: str):
    orient = "v" if axis == "x" else "h"
    feats: dict[tuple[int, float], float] = {}
    for p in pieces:
        if lines[p.line].orient != orient:
            continue
        offs = (0.0,) if mode == "centrelines" else (-p.t / 2, p.t / 2)
        for o in offs:
            key = (p.line, round(o, 4))
            feats[key] = lines[p.line].coord + key[1]
    return [(k[0], k[1], v) for k, v in feats.items()]


def _match(feats, pos: float, tol: float):
    best, best_d = None, tol
    for f in feats:
        d = abs(f[2] - pos)
        if d <= best_d:
            best, best_d = f, d
    return best


def _build(lines, pieces, dims, mode, prm) -> list[_Eq]:
    feats = {ax: _features(lines, pieces, ax, mode) for ax in ("x", "y")}
    eqs = []
    for d in dims:
        fl = _match(feats[d.axis], d.lo, prm.dim_match_tol_m)
        fh = _match(feats[d.axis], d.hi, prm.dim_match_tol_m)
        if fl is None or fh is None or fl[0] == fh[0] or fh[2] <= fl[2]:
            continue
        measured = fh[2] - fl[2]
        if abs(measured - d.value) > max(prm.dim_max_change_m, prm.dim_max_change_rel * d.value):
            continue
        eqs.append(_Eq(fl[0], fl[1], fh[0], fh[1], d.value))
    return eqs


def _residuals(x: np.ndarray, eqs: list[_Eq]) -> np.ndarray:
    return np.array([(x[e.j] + e.oj) - (x[e.i] + e.oi) - e.value for e in eqs])


def _solve(x0: np.ndarray, eqs: list[_Eq], prm: GeometryParams) -> np.ndarray:
    """Linear least squares: label equations (weight 1) plus weak priors towards the drawing."""
    n = len(x0)
    a = np.zeros((len(eqs) + n, n))
    b = np.zeros(len(eqs) + n)
    for k, e in enumerate(eqs):
        a[k, e.j] += 1.0
        a[k, e.i] -= 1.0
        b[k] = e.value - e.oj + e.oi
    w = prm.prior_weight
    a[len(eqs) :, :] = np.eye(n) * w
    b[len(eqs) :] = x0 * w
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _fit_robust(x0: np.ndarray, eqs: list[_Eq], prm: GeometryParams):
    """Fit, then greedily drop the label whose removal best restores consistency (ties: the
    solution that stays closest to the drawing). Returns (x, kept equations)."""
    active = list(eqs)
    max_drop = int(prm.dim_max_reject_share * len(eqs))
    x = _solve(x0, active, prm)
    while len(eqs) - len(active) < max_drop and len(active) > 1:
        worst = float(np.max(np.abs(_residuals(x, active))))
        if worst <= prm.dim_consistent_m:
            break
        best = None
        for k in range(len(active)):
            rest = active[:k] + active[k + 1 :]
            xk = _solve(x0, rest, prm)
            score = (
                round(float(np.max(np.abs(_residuals(xk, rest)))), 4),
                float(np.linalg.norm(xk - x0)),
            )
            if best is None or score < best[0]:
                best = (score, k, xk)
        if best is None or best[0][0] >= worst - 1e-4:
            break
        active.pop(best[1])
        x = best[2]
    return x, active


def _rms(r: np.ndarray) -> float:
    return float(math.sqrt(float(np.mean(r**2)))) if len(r) else 0.0


def fit_dimensions(
    lines: list[Line], pieces: list[Piece], dims: list[DimM], prm: GeometryParams
) -> FitResult:
    x0 = np.array([ln.coord for ln in lines], dtype=float)
    best: FitResult | None = None
    for mode in ("faces", "centrelines"):
        eqs = _build(lines, pieces, dims, mode, prm)
        if not eqs:
            continue
        x, keep = _fit_robust(x0, eqs, prm)
        n_out = len(eqs) - len(keep)
        res = FitResult(
            coords=[float(v) for v in x],
            mode=mode,
            n_used=len(keep),
            n_dims=len(dims),
            n_outliers=n_out,
            residual_before_m=_rms(_residuals(x0, keep)),
            residual_m=_rms(_residuals(x, keep)),
        )
        if (
            best is None
            or res.n_used > best.n_used
            or (
                res.n_used == best.n_used
                and (res.residual_before_m or 0) < (best.residual_before_m or 0)
            )
        ):
            best = res
    if best is None:
        return FitResult([float(v) for v in x0], None, 0, len(dims), 0, None, None)
    return best
