"""Joint decoding of dimension chains (Maßketten).

A German plan says every length more than once. The parts of a chain add up to the overall
dimension (*Gesamtmaß*) on the next chain level, nested chains share their end ticks through the
extension lines, the same overall length is written on opposite facades, and every value matches
its pixel span at one page-wide px/m scale. The greedy reader ignores all of that and keeps one
guess per label. This module chooses the readings jointly:

* **Chains.** Labels whose dimension lines coincide form one chain; its ticks come from the line
  finder (:mod:`kelpod.ocr.dimlines`), so ticks of unread labels are known too.
* **Unknowns.** One integer coordinate ``X`` (mm) per tick, per orientation.
* **Options per label.** Each n-best value (:mod:`kelpod.ocr.nbest`, plus a lost-superscript-5
  variant) on each nearby tick pair. Choosing an option enforces ``X_b - X_a = value``. Its cost
  combines the recogniser probability, the fit to the pixel span and the label position. Or the
  label is dropped as misread, at a fixed cost.
* **Links.** Ticks of different chains at the same position (same extension line, the facade
  corners) are equal unless that link is broken, at a cost.
* **Prior.** Every tick interval should stay near its drawn length (soft, linear beyond a
  tolerance).

CP-SAT (OR-Tools, Apache-2.0) finds the cheapest consistent choice. Afterwards every tick
interval without a label whose two ends are tied together by chosen equalities gets its value from
the arithmetic, exactly (a misread label there is ``corrected``, a missing one ``inferred``).

The decoder only changes labels it can place on a chain; everything else keeps the greedy result.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from kelpod.contracts import DimensionLabel, Point, TextItem
from kelpod.ocr.dimlines import DimensionLineFinder, LineTicks
from kelpod.ocr.engines import cw_to_original, quad_size
from kelpod.ocr.parse import parse_length_m, parse_number


@dataclass
class ChainParams:
    tick_merge_px: float = 2.0
    """Ticks of one chain closer than this are one tick."""
    same_line_px: float = 2.5
    same_line_rel: float = 0.3
    """Two labels sit on the same dimension line if the line rows differ by at most
    max(px, rel x text height)."""
    tick_diag_min: float = 0.7
    tick_ext_min: float = 0.8
    """A dimension tick: 45° stroke score >= diag_min and an extension line of at least
    ext_min x text height."""
    link_px: float = 3.0
    """Ticks of different chains this close are linked (same extension line / corner)."""
    max_skip: int = 2
    """A label may span up to this many tick intervals (a spurious tick in between)."""
    fit_tol_px: float = 2.5
    fit_tol_rel: float = 0.02
    """Soft tolerance of value x px/m against the tick span."""
    fit_max_px: float = 6.0
    fit_max_rel: float = 0.08
    """Hard limit: options further off than max(px, rel x span) are not considered."""
    drop_cost: float = 3.0
    drop_cost_conf: float = 3.0
    """Dropping a label costs drop_cost + drop_cost_conf x its top n-best probability."""
    skip_cost: float = 1.5
    displaced_cost: float = 1.0
    superscript_cost: float = 0.7
    superscript_cost_blind: float = 2.5
    min_prob: float = 1e-3
    confident_prob: float = 0.8
    robust_margin: float = 1.0
    """Corrected / inferred values are only emitted if every unchecked label they depend on
    would cost at least this much more to read differently."""
    tail_prob: float = 0.3
    """Probability factor for reading a 4th decimal as a tick stub and dropping it."""
    max_options: int = 12
    max_values: int = 200
    """At most this many candidate coordinates per node."""
    domain_mm: float = 60.0
    domain_rel: float = 0.01
    """A tick coordinate stays within domain_mm + domain_rel x (distance from the first tick
    of its orientation) of its drawn position."""
    time_limit_s: float = 2.0
    min_value_m: float = 0.02


SCALE = 100  # objective costs are integers: cost x SCALE


@dataclass
class Candidate:
    """One dimension text with its readings and (if found) the line under it."""

    item: TextItem
    angle: float
    greedy_value: float | None
    values: list[tuple[float, float]]
    """``(value_m, cost)``, cost 0 for the most probable reading."""
    ticks: LineTicks | None = None
    vertical: bool = False


@dataclass
class Chain:
    vertical: bool
    y: float
    ticks: list[float]
    text_h: float
    members: list[int] = field(default_factory=list)


_SMALL_GLYPH = re.compile(r"[0-9⁰¹²³⁴⁵⁶⁷⁸⁹'\"`´°*]")
"""A glyph after the number that may be a superscript digit the recogniser could not read."""
_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_JUNK = str.maketrans({"，": ",", "、": "", "。": "", "．": ".", " ": ""})


def reading_values(item: TextItem, p: ChainParams | None = None) -> list[tuple[float, float]]:
    """Candidate lengths of a text item with costs ``-ln(p / p_best)`` (0 = best).

    Uses the greedy text and the n-best alternatives; only whole millimetres. A two-digit
    centimetre value (``36``) or a two-decimal metre value (``1,08``) may have lost a superscript
    ``5`` (``36⁵``, ``1,08⁵``), the usual half centimetre on German plans. That variant costs
    ``superscript_cost`` when the n-best list shows a small extra glyph after the number (``36²``,
    ``363``, ``36'``), else ``superscript_cost_blind``.
    """
    p = p or ChainParams()
    readings = list(item.alternatives)
    texts = {t for t, _ in readings}
    if item.text not in texts:
        best = max((q for _, q in readings), default=item.confidence)
        readings.insert(0, (item.text, max(best, item.confidence)))
    clean = [(t.translate(_JUNK).strip(".,:;'\"`"), t.translate(_JUNK), q) for t, q in readings]
    probs: dict[float, float] = {}
    variants: dict[float, tuple[float, str]] = {}
    tails = []  # a tick or extension-line stub read as a trailing digit: "0,3651"
    for t, raw, q in clean:
        if len(t) >= 4 and re.search(r"[,.]\d{3}\d$", t):
            tails.append((t[:-1], raw, q * p.tail_prob))
    for t, raw, q in clean + tails:
        v = parse_length_m(t)
        if v is None or abs(v * 1000 - round(v * 1000)) > 1e-6:
            continue
        probs[v] = probs.get(v, 0.0) + max(q, 0.0)
        n = parse_number(t)
        if n is not None and not n.superscript and (
            (n.has_comma and n.decimal_digits == 2) or (not n.has_comma and n.integer_digits == 2)
        ):
            w = round(v + 0.005, 4)
            if w not in variants or q > variants[w][0]:
                variants[w] = (q, t)
    if not probs:
        return []
    pmax = max(probs.values())
    out = {v: -math.log(max(q, p.min_prob * pmax) / pmax) for v, q in probs.items()}
    for w, (q, t) in variants.items():
        trailing = any(raw.startswith(t) and _SMALL_GLYPH.search(raw[len(t):])
                       for _, raw, _ in clean)
        extra = p.superscript_cost if trailing else p.superscript_cost_blind
        c = -math.log(max(q, p.min_prob * pmax) / pmax) + extra
        if w not in out or c < out[w]:
            out[w] = c
    return sorted(out.items(), key=lambda kv: kv[1])


def _cluster(xs: list[float], tol: float) -> list[float]:
    out: list[list[float]] = []
    for x in sorted(xs):
        if out and x - out[-1][-1] <= tol:
            out[-1].append(x)
        else:
            out.append([x])
    return [float(np.mean(g)) for g in out]


def group_chains(
    cands: list[Candidate], p: ChainParams, finder: DimensionLineFinder | None = None
) -> list[Chain]:
    """Labels on the same dimension line (same orientation, same row, shared ticks) -> chains."""
    idx = [i for i, c in enumerate(cands) if c.ticks is not None and len(c.ticks.ticks) >= 2]
    parent = {i: i for i in idx}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a_k, a in enumerate(idx):
        for b in idx[a_k + 1 :]:
            ca, cb = cands[a], cands[b]
            tol = max(p.same_line_px, p.same_line_rel * min(ca.ticks.text_h, cb.ticks.text_h))
            if ca.vertical != cb.vertical or abs(ca.ticks.y - cb.ticks.y) > tol:
                continue
            ta, tb = ca.ticks.ticks, cb.ticks.ticks
            if any(abs(x - y) <= p.tick_merge_px for x in ta for y in tb):
                parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for i in idx:
        groups.setdefault(find(i), []).append(i)
    chains = []
    for members in groups.values():
        cs = [cands[i] for i in members]
        ticks = _cluster([t for c in cs for t in c.ticks.ticks], p.tick_merge_px)
        ch = Chain(
            vertical=cs[0].vertical,
            y=float(np.median([c.ticks.y for c in cs])),
            ticks=ticks,
            text_h=float(np.median([c.ticks.text_h for c in cs])),
            members=sorted(members),
        )
        if finder is not None:
            ch.ticks = filter_ticks(finder, ch, p)
        if len(ch.ticks) >= 2:
            chains.append(ch)
    return chains


def _run_from(col: np.ndarray, gap: int) -> int:
    """Length of the ink run at the start of ``col`` (bool), bridging gaps up to ``gap``."""
    n, miss = 0, 0
    for k, v in enumerate(col):
        if v:
            n, miss = k + 1, 0
        else:
            miss += 1
            if miss > gap:
                break
    return n


def tick_evidence(
    finder: DimensionLineFinder, vertical: bool, y: float, x: float, text_h: float
) -> tuple[float, int]:
    """``(diagonal score, extension length)`` of a crossing at ``x`` on the line at row ``y``
    (upright frame). The diagonal score is the share of a short 45° stroke found on both sides
    of the line (either direction); the extension length is the longer perpendicular ink run
    that starts at the line (px)."""
    ink, dil = finder.rotated if vertical else (finder.ink, finder.dil)
    H, W = ink.shape
    yi, xi = int(round(y)), int(round(x))
    if not (0 <= yi < H and 0 <= xi < W):
        return 0.0, 0
    L = max(3, int(round(0.3 * text_h)))
    half = 1  # the line itself (thin line, anti-aliased)

    def diag(dx: int) -> float:
        hits = []
        for sgn in (-1, 1):
            n = 0
            for k in range(1, L + 1):
                yy, xx = yi + sgn * (half + k), xi - sgn * dx * (half + k)
                n += 0 <= yy < H and 0 <= xx < W and bool(dil[yy, xx])
            hits.append(n / L)
        return min(hits)

    d = max(diag(1), diag(-1))
    reach = int(3 * text_h)
    x0, x1 = max(0, xi - 1), min(W, xi + 2)
    up = dil[max(0, yi - half - reach) : max(0, yi - half), x0:x1].any(axis=1)[::-1]
    dn = dil[yi + half + 1 : yi + half + 1 + reach, x0:x1].any(axis=1)
    return d, max(_run_from(up, 1), _run_from(dn, 1))


def filter_ticks(finder: DimensionLineFinder, ch: Chain, p: ChainParams) -> list[float]:
    """Keep the ticks that look like dimension ticks: a 45° stroke plus an extension line
    towards the wall. Falls back to 45° strokes alone, then to all crossings, when a chain has
    fewer than two such ticks (plans drawn without extension lines or with other tick marks).
    Glyphs of labels sitting on the line and extension lines of outer chains that only pass
    through are dropped this way."""
    ev = [tick_evidence(finder, ch.vertical, ch.y, t, ch.text_h) for t in ch.ticks]
    strong = [t for t, (d, e) in zip(ch.ticks, ev, strict=True)
              if d >= p.tick_diag_min and e >= p.tick_ext_min * ch.text_h]
    if len(strong) >= 2:
        return strong
    diag = [t for t, (d, _) in zip(ch.ticks, ev, strict=True) if d >= p.tick_diag_min]
    if len(diag) >= 2:
        return diag
    return ch.ticks


@dataclass
class _Option:
    cand: int
    chain: int
    i: int
    j: int
    value_m: float
    cost: float


def _options(c: Candidate, k: int, ch: Chain, px_per_m: float, p: ChainParams) -> list[_Option]:
    t = ch.ticks
    cx, th = c.ticks.cx, ch.text_h
    tw = quad_size(c.item.quad)[0]
    out = []
    for i in range(len(t)):
        for j in range(i + 1, min(len(t), i + 1 + p.max_skip)):
            span = t[j] - t[i]
            off = abs((t[i] + t[j]) / 2 - cx)
            if off <= span / 2 + 0.5 * th:
                pos = 0.3 * off / (span / 2 + th)
            elif span < tw + 2 * th and off <= span / 2 + 2.5 * tw + 2 * th:
                pos = p.displaced_cost  # short span: the label sits beside it
            else:
                continue
            pos += p.skip_cost * (j - i - 1)
            for v, vc in c.values:
                err = abs(v * px_per_m - span)
                if err > max(p.fit_max_px, p.fit_max_rel * span):
                    continue
                tol = max(p.fit_tol_px, p.fit_tol_rel * span)
                fit = 0.5 * min((err / tol) ** 2, 9.0)
                out.append(_Option(k, -1, i, j, v, vc + pos + fit))
    out.sort(key=lambda o: o.cost)
    return out[: p.max_options]


_DIGIT_FIX = str.maketrans("OoDQlI|", "0000111")


def digits_compatible(text: str, value_m: float) -> bool:
    """True if the digits read in ``text`` appear in order in ``value_m`` written to the mm
    (``36`` / ``36⁵`` -> 0.365, ``605`` -> 1.605, ``15`` -> 0.175): the reading is a partial
    view of that value, so correcting it is a repair, not a replacement."""
    got = re.sub(r"[^0-9]", "", text.translate(_DIGIT_FIX).translate(_SUP))
    if not got:
        return True
    want = f"{value_m:.3f}".replace(".", "")
    it = iter(want)
    return all(ch in it for ch in got)


def _confident(c: Candidate, p: ChainParams) -> bool:
    """A clear, complete-looking reading: high n-best probability and at least three digits."""
    top = max((q for _, q in c.item.alternatives), default=c.item.confidence)
    digits = len(re.sub(r"[^0-9]", "", c.item.text))
    return top >= p.confident_prob and digits >= 3


def _to_image(finder: DimensionLineFinder, pt: Point, vertical: bool) -> Point:
    if not vertical:
        return (float(pt[0]), float(pt[1]))
    return cw_to_original(pt, finder.ink.shape[0])


@dataclass
class ChainResult:
    labels: dict[int, DimensionLabel]
    """Candidate index -> decoded label (only candidates that sit on a chain)."""
    dropped: set[int]
    """Candidates on a chain that were judged misread and could not be corrected."""
    inferred: list[DimensionLabel]
    notes: list[str]


def decode(
    cands: list[Candidate],
    finder: DimensionLineFinder,
    px_per_m: float,
    params: ChainParams | None = None,
) -> ChainResult:
    """Jointly decode all candidates that sit on a dimension chain (see the module docstring)."""
    from ortools.sat.python import cp_model

    p = params or ChainParams()
    chains = group_chains(cands, p, finder)
    if not chains or not (px_per_m > 0 and math.isfinite(px_per_m)):
        return ChainResult({}, set(), [], [])
    chain_of = {k: ci for ci, ch in enumerate(chains) for k in ch.members}
    opts: list[_Option] = []
    for k, ci in chain_of.items():
        for o in _options(cands[k], k, chains[ci], px_per_m, p):
            o.chain = ci
            opts.append(o)

    mm = 1000.0 / px_per_m  # px -> mm
    # 1. nodes: ticks of different chains at the same position (extension lines, corners)
    #    are one coordinate
    keys = [(ci, ti) for ci, ch in enumerate(chains) for ti in range(len(ch.ticks))]
    node_of = {k: k for k in keys}

    def find(a):
        while node_of[a] != a:
            node_of[a] = node_of[node_of[a]]
            a = node_of[a]
        return a

    for a in range(len(chains)):
        for b in range(a + 1, len(chains)):
            if chains[a].vertical != chains[b].vertical:
                continue
            for ia, ta in enumerate(chains[a].ticks):
                for ib, tb in enumerate(chains[b].ticks):
                    if abs(ta - tb) <= p.link_px:
                        node_of[find((a, ia))] = find((b, ib))
    root = {k: find(k) for k in keys}
    drawn: dict[tuple[int, int], float] = {}
    for k in keys:
        drawn.setdefault(root[k], []).append(chains[k[0]].ticks[k[1]] * mm)
    drawn = {n: float(np.mean(v)) for n, v in drawn.items()}
    lo = {o: min((drawn[root[(ci, 0)]] for ci, ch in enumerate(chains) if ch.vertical == o),
                 default=0.0) for o in (False, True)}
    vert_of = {root[(ci, ti)]: chains[ci].vertical for ci, ti in keys}

    def window(n) -> float:
        return p.domain_mm + p.domain_rel * (drawn[n] - lo[vert_of[n]])

    # 2. candidate coordinates per node (mm): everything reachable from an anchor through the
    #    label options without leaving the node's window around its drawn position
    edges = [(root[(o.chain, o.i)], root[(o.chain, o.j)], int(round(o.value_m * 1000)))
             for o in opts]
    adj: dict = {}
    for a, b, v in edges:
        adj.setdefault(a, []).append((b, v))
        adj.setdefault(b, []).append((a, -v))
    values: dict = {}
    for n in sorted(drawn, key=lambda n: drawn[n]):
        if n in values:
            continue
        anchor = n  # one fixed value: the decoding is invariant to a common shift
        values[n] = {int(round(drawn[n]))}
        queue = [(n, int(round(drawn[n])))]
        while queue:
            a, u = queue.pop()
            for b, v in adj.get(a, []):
                w = u + v
                if b == anchor or abs(w - drawn[b]) > window(b):
                    continue
                vb = values.setdefault(b, set())
                if w not in vb and len(vb) < p.max_values:
                    vb.add(w)
                    queue.append((b, w))
    for n in drawn:
        values.setdefault(n, set()).add(int(round(drawn[n])))

    # 3. pure Boolean model: one value per node, one option (or drop) per label
    m = cp_model.CpModel()
    xb = {n: {u: m.NewBoolVar("") for u in sorted(vs)} for n, vs in values.items()}
    for n in xb:
        m.AddExactlyOne(list(xb[n].values()))
    obj = []
    b_opt = []
    by_cand: dict[int, list[int]] = {}
    for oi, (o, (a, b, v)) in enumerate(zip(opts, edges, strict=True)):
        bo = m.NewBoolVar("")
        b_opt.append(bo)
        by_cand.setdefault(o.cand, []).append(oi)
        for u, xa in xb[a].items():  # option and X_a = u  =>  X_b = u + v
            xbv = xb[b].get(u + v)
            m.AddBoolOr([bo.Not(), xa.Not()] + ([xbv] if xbv is not None else []))
        obj.append(int(round(o.cost * SCALE)) * bo)
    for k in chain_of:
        d = m.NewBoolVar("")
        m.AddExactlyOne([b_opt[oi] for oi in by_cand.get(k, [])] + [d])
        it = cands[k].item
        top = max((q for _, q in it.alternatives), default=it.confidence)
        obj.append(int(round((p.drop_cost + p.drop_cost_conf * top) * SCALE)) * d)
    for ci, ch in enumerate(chains):  # one label per tick interval of a chain
        for e in range(len(ch.ticks) - 1):
            cover = [b_opt[oi] for oi, o in enumerate(opts) if o.chain == ci and o.i <= e < o.j]
            if len(cover) > 1:
                m.AddAtMostOne(cover)
    m.Minimize(sum(obj))
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.max_time_in_seconds = p.time_limit_s
    status = solver.Solve(m)
    stats = (solver.StatusName(status), round(solver.WallTime(), 3),
             sum(len(v) for v in values.values()))
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return ChainResult({}, set(), [], ["chain decoding found no solution; greedy readings kept"])
    xnode = {n: next(u for u, bv in xb[n].items() if solver.Value(bv)) for n in xb}
    xval = {k: xnode[root[k]] for k in keys}

    # 4. which ticks are tied together by chosen equalities (exact differences), and how firmly
    chosen: dict[int, _Option] = {}
    g = nx.Graph()
    g.add_nodes_from(xnode)
    for oi, o in enumerate(opts):
        if solver.Value(b_opt[oi]):
            chosen[o.cand] = o
            a, b, _ = edges[oi]
            if g.has_edge(a, b):
                g[a][b]["labels"].append(o.cand)
            else:
                g.add_edge(a, b, labels=[o.cand])
    bridges = {frozenset(e) for e in nx.bridges(g)} if g.number_of_edges() else set()

    def margin(k: int) -> float:
        """Extra cost of the cheapest other value for label ``k``."""
        o = chosen[k]
        alts = [opts[oi].cost for oi in by_cand.get(k, [])
                if abs(opts[oi].value_m - o.value_m) > 1e-6]
        return min(alts, default=math.inf) - o.cost

    def weak(a, b) -> bool:
        """An equality that nothing else checks (a bridge, one label) and that a cheap
        alternative reading would shift."""
        labs = g[a][b]["labels"]
        return frozenset((a, b)) in bridges and len(labs) == 1 and margin(labs[0]) < p.robust_margin

    def tied(k1, k2) -> bool:
        """The difference of the two ticks follows from the chosen labels, and robustly so."""
        a, b = root[k1], root[k2]
        if a == b:
            return True
        if not nx.has_path(g, a, b):
            return False
        path = nx.shortest_path(g, a, b)
        return not any(weak(u, v) for u, v in zip(path[:-1], path[1:], strict=True))

    import os
    if os.environ.get("KELPOD_CHAIN_DEBUG"):
        for k in sorted(chain_of):
            c = cands[k]
            ch = chains[chain_of[k]]
            o = chosen.get(k)
            alts = [(round(opts[oi].value_m, 4), opts[oi].i, opts[oi].j, round(opts[oi].cost, 2))
                    for oi in by_cand.get(k, [])][:6]
            print(chain_of[k], ch.vertical, round(ch.y), [round(t) for t in ch.ticks], repr(c.item.text),
                  "->", None if o is None else (o.value_m, o.i, o.j, round(o.cost, 2)), alts)
    labels: dict[int, DimensionLabel] = {}
    n_corr = 0
    for k, o in chosen.items():
        c, ch = cands[k], chains[o.chain]
        line = (_to_image(finder, (ch.ticks[o.i], ch.y), ch.vertical),
                _to_image(finder, (ch.ticks[o.j], ch.y), ch.vertical))
        corrected = c.greedy_value is None or abs(o.value_m - c.greedy_value) > 1e-6
        n_corr += corrected
        labels[k] = DimensionLabel(
            value_m=o.value_m, text=c.item.text, anchor=c.item.center, angle_deg=c.angle,
            line=line, confidence=c.item.confidence * (0.8 if corrected else 1.0),
            origin="corrected" if corrected else "read",
        )
    # intervals without a chosen label whose ends are tied: exact value from the arithmetic
    covered = {(o.chain, e) for o in chosen.values() for e in range(o.i, o.j)}
    dropped_at: dict[tuple[int, int], int] = {}
    for k in chain_of:
        if k in chosen:
            continue
        ch_opts = [opts[oi] for oi in by_cand.get(k, []) if opts[oi].j == opts[oi].i + 1]
        c = cands[k]
        ci = chain_of[k]
        if ch_opts:
            best = min(ch_opts, key=lambda o: abs((chains[ci].ticks[o.i] + chains[ci].ticks[o.j])
                                                  / 2 - c.ticks.cx))
            dropped_at.setdefault((ci, best.i), k)
        else:  # no option at all: the interval under the text centre
            t = chains[ci].ticks
            for e in range(len(t) - 1):
                if t[e] <= c.ticks.cx <= t[e + 1]:
                    dropped_at.setdefault((ci, e), k)
    inferred: list[DimensionLabel] = []
    dropped: set[int] = {k for k in chain_of if k not in chosen}
    # a clear reading that was dropped is kept as read (greedy) unless another label took
    # its interval
    where = {k: key for key, k in dropped_at.items()}
    keep_greedy: set[int] = {k for k in dropped if _confident(cands[k], p)
                             and where.get(k) not in covered}
    n_inf = 0
    for ci, ch in enumerate(chains):
        for e in range(len(ch.ticks) - 1):
            if (ci, e) in covered or not tied((ci, e), (ci, e + 1)):
                continue
            v = (xval[ci, e + 1] - xval[ci, e]) / 1000.0
            span = ch.ticks[e + 1] - ch.ticks[e]
            if v < p.min_value_m or abs(v * px_per_m - span) > max(p.fit_max_px,
                                                                  p.fit_max_rel * span):
                continue
            line = (_to_image(finder, (ch.ticks[e], ch.y), ch.vertical),
                    _to_image(finder, (ch.ticks[e + 1], ch.y), ch.vertical))
            k = dropped_at.get((ci, e))
            if k is not None:
                c = cands[k]
                if digits_compatible(c.item.text, v):  # partial read: repair it
                    labels[k] = DimensionLabel(
                        value_m=round(v, 4), text=c.item.text, anchor=c.item.center,
                        angle_deg=c.angle, line=line, confidence=c.item.confidence * 0.6,
                        origin="corrected",
                    )
                    dropped.discard(k)
                    keep_greedy.discard(k)
                    n_corr += 1
                    continue
                if _confident(c, p):
                    # a clear reading that fits neither its span nor the arithmetic: more
                    # likely a missed or extra tick than a misread, so infer nothing here
                    continue
                keep_greedy.discard(k)  # a fragment: replace it by the inferred value
            mid = ((ch.ticks[e] + ch.ticks[e + 1]) / 2, ch.y - 0.8 * ch.text_h)
            inferred.append(DimensionLabel(
                value_m=round(v, 4), text="", anchor=_to_image(finder, mid, ch.vertical),
                angle_deg=90.0 if ch.vertical else 0.0, line=line, confidence=0.5,
                origin="inferred",
            ))
            n_inf += 1
    dropped -= keep_greedy
    notes = [
        f"chain decoding: {len(chains)} chain(s), {len(chain_of)} label(s) on them, "
        f"{n_corr} corrected, {len(dropped)} dropped as misread, {n_inf} inferred from the "
        f"chain arithmetic (CP-SAT {stats[0]}, {stats[1]} s, {stats[2]} node values)"
    ]
    return ChainResult(labels, dropped, inferred, notes)


__all__ = ["Candidate", "Chain", "ChainParams", "ChainResult", "decode", "group_chains",
           "reading_values"]
