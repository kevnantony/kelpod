"""Step 5: consistency checks on a plan graph (NetworkX + Shapely), in plain language."""

from __future__ import annotations

import networkx as nx
from shapely.geometry import Polygon

from kelpod.contracts import ConsistencyReport, PlanGraph

STAMP_MISMATCH_REL = 0.10


def _signed_area(poly: list[tuple[float, float]]) -> float:
    s = 0.0
    for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1], strict=True):
        s += x0 * y1 - x1 * y0
    return s / 2


def check(graph: PlanGraph) -> ConsistencyReport:
    """Components, dangling nodes, node and room closure, stamp-area mismatches."""
    issues: list[str] = []
    ids = {n.id for n in graph.nodes}
    pos = {n.id: (n.x, n.y) for n in graph.nodes}
    g = nx.MultiGraph()
    g.add_nodes_from(ids)
    for w in graph.walls:
        if w.start not in ids or w.end not in ids:
            issues.append(f"Wall {w.id} refers to a missing node.")
            continue
        if w.start == w.end:
            issues.append(f"Wall {w.id} starts and ends at the same node.")
            continue
        g.add_edge(w.start, w.end, key=w.id)

    n_comp = nx.number_connected_components(g) if len(g) else 0
    deg = dict(g.degree())
    dangling = sorted(n for n, d in deg.items() if d == 1)
    isolated = sorted(n for n, d in deg.items() if d == 0)
    node_closure = sum(1 for d in deg.values() if d >= 2) / len(deg) if deg else 0.0

    if not graph.walls:
        issues.append("No walls were found.")
    if n_comp > 1:
        issues.append(
            f"The walls form {n_comp} separate groups; a single storey should be one "
            "connected group."
        )
    if dangling:
        x, y = pos[dangling[0]]
        issues.append(
            f"{len(dangling)} wall end(s) do not meet another wall, e.g. at "
            f"({x:.2f}, {y:.2f}) m (nodes {', '.join(map(str, dangling[:8]))})."
        )
    if isolated:
        issues.append(f"{len(isolated)} node(s) have no wall attached.")

    valid = 0
    for r in graph.rooms:
        label = f"'{r.name}'" if r.name else str(r.id)
        ok = len(r.polygon) >= 3
        if ok:
            poly = Polygon(r.polygon)
            ok = poly.is_valid and poly.area > 1e-6 and _signed_area(list(r.polygon)) > 0
        if ok:
            valid += 1
        else:
            issues.append(f"Room {label} is not a valid counter-clockwise polygon.")
        if r.stamp_area_m2 and r.stamp_area_m2 > 0:
            rel = abs(r.area_m2 - r.stamp_area_m2) / r.stamp_area_m2
            if rel > STAMP_MISMATCH_REL:
                issues.append(
                    f"Room {label}: fitted area {r.area_m2:.2f} m² differs from the stamp "
                    f"{r.stamp_area_m2:.2f} m² by {rel * 100:.0f} %."
                )
    room_closure = valid / len(graph.rooms) if graph.rooms else 0.0
    if graph.walls and not graph.rooms:
        issues.append("No closed room was found.")

    walls = {w.id: w for w in graph.walls}
    for o in graph.openings:
        w = walls.get(o.wall_id)
        if w is None or w.start not in pos or w.end not in pos:
            issues.append(f"Opening {o.id} sits on a missing wall.")
            continue
        (x0, y0), (x1, y1) = pos[w.start], pos[w.end]
        length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        if o.offset_m < -1e-6 or o.offset_m + o.width_m > length + 1e-6:
            issues.append(f"Opening {o.id} ({o.kind.value}) extends beyond its wall.")

    return ConsistencyReport(
        n_nodes=len(graph.nodes),
        n_walls=len(graph.walls),
        n_rooms=len(graph.rooms),
        n_components=n_comp,
        dangling_nodes=dangling,
        node_closure=node_closure,
        room_closure=room_closure,
        dimension_residual_m=graph.report.dimension_residual_m if graph.report else None,
        issues=issues,
    )
