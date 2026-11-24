"""ResPlan (CC BY 4.0 data, MIT code, https://github.com/m-agour/ResPlan) -> :class:`VectorPlan`.

The pickle is untrusted: it is read with a restricted unpickler that only allows the three
globals the file uses (``shapely.io.from_wkb``, numpy scalar and dtype; checked with
``pickletools`` without unpickling). Plans are normalised to a ~256-unit canvas, so the scale is
set per plan from ``wall_depth`` (typical wall thickness in plan units), mapped to a random
German wall thickness of 0.20-0.30 m. Canonical splits from the repo's ``split.json``.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from shapely import affinity
from shapely.ops import unary_union

from kelpod_train.render import VectorPlan, _polys, opening_box

PKL = "resplan/ResPlan.pkl"
SPLIT = "src/ResPlan/split.json"
_ALLOWED = {("shapely.io", "from_wkb"), ("numpy._core.multiarray", "scalar"), ("numpy", "dtype")}
NAMES = {"bathroom": "Bad", "bedroom": "Schlafen", "kitchen": "Küche", "living": "Wohnen"}
IGNORE = ("balcony",)


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) not in _ALLOWED:
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")
        return super().find_class(module, name)


def load_all(data: Path) -> list[dict]:
    with open(Path(data) / PKL, "rb") as f:
        return _SafeUnpickler(f).load()


def plan_to_vector(p: dict, rng: np.random.Generator) -> VectorPlan | None:
    wd = float(p.get("wall_depth") or 0)
    if wd <= 0 or p.get("wall") is None or p["wall"].is_empty:
        return None
    s = float(rng.uniform(0.20, 0.30)) / wd  # metres per unit

    def sc(g):
        return affinity.scale(g, s, -s, origin=(0, 0))  # y flipped to "y up"

    walls = [sc(g) for g in _polys(p["wall"])]
    walls_u = unary_union(walls)

    def get(key):
        g = p.get(key)
        return [] if g is None or g.is_empty else [sc(x) for x in _polys(g)]

    doors = [opening_box(g, walls_u) for g in get("door") + get("front_door")]
    windows = [opening_box(g, walls_u) for g in get("window")]
    # "inner" is the whole interior: labelled as room first, named rooms drawn over it
    rooms = [(g, None) for g in get("inner")] + [(g, NAMES[k]) for k in NAMES for g in get(k)]
    ignore = [g for k in IGNORE for g in get(k)]
    return VectorPlan(walls=walls, doors=doors, windows=windows, rooms=rooms, ignore=ignore)


def iter_plans(
    data: Path, *, split: str = "train", limit: int | None = None, seed: int = 0
) -> Iterator[tuple[str, VectorPlan]]:
    plans = {int(p["id"]): p for p in load_all(data)}  # split.json lists plan ids
    ids = json.loads((Path(data) / SPLIT).read_text())[split]
    rng = np.random.default_rng(seed)
    ids = list(ids)
    rng.shuffle(ids)
    n = 0
    for i in ids:
        if int(i) not in plans:
            continue
        v = plan_to_vector(plans[i], np.random.default_rng([int(i), 2]))
        if v is None:
            continue
        yield str(i), v
        n += 1
        if limit and n >= limit:
            break
