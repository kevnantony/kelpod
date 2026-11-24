"""Swiss Dwellings v3.0.0 (CC BY 4.0, https://zenodo.org/records/7788422) -> :class:`VectorPlan`.

One plan per ``floor_id`` (all apartments of a storey, as in a German *Geschossplan*). Geometry is
WKT in metres. Split by ``site_id`` (so the same building never lands in train and test):
``crc32(site_id) % 10 == 0`` test, ``== 1`` val, else train.
"""

from __future__ import annotations

import zlib
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from shapely import wkt
from shapely.ops import unary_union

from kelpod_train.render import VectorPlan, opening_box

CSV = "swiss-dwellings/swiss-dwellings-v3.0.0/geometries.csv"
NAMES = {
    "BATHROOM": "Bad",
    "BEDROOM": "Schlafen",
    "LIVING_ROOM": "Wohnen",
    "LIVING_DINING": "Wohnen",
    "KITCHEN": "Küche",
    "KITCHEN_DINING": "Küche",
    "DINING": "Essen",
    "CORRIDOR": "Flur",
    "CORRIDORS_AND_HALLS": "Flur",
    "STOREROOM": "Abst.",
    "STAIRCASE": "Treppe",
    "OFFICE": "Büro",
}
ROOM_NAMES = ("Zimmer", "Kind", "Schlafen", "Arbeiten", "Gast")
IGNORE = {"BALCONY", "LOGGIA", "TERRACE", "GARDEN", "PATIO", "WINTERGARTEN", "ARCADE"}
NOT_DRAWN = {"SHAFT", "VOID", "OUTDOOR_VOID", "LIGHTWELL", "AIR", "ELEVATOR", "TRANSPORT_SHAFT"}
FIXTURES = {
    "BATHTUB", "SINK", "TOILET", "SHOWER", "KITCHEN", "STAIRS", "BUILT_IN_FURNITURE",
    "WASHING_MACHINE",
}


def split_of(site_id: int) -> str:
    h = zlib.crc32(str(int(site_id)).encode()) % 10
    return "test" if h == 0 else "val" if h == 1 else "train"


def _geoms(rows) -> list:
    out = []
    for g in rows:
        try:
            geom = wkt.loads(g)
        except Exception:  # noqa: BLE001 - malformed rows are skipped
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
        if not geom.is_empty and geom.area > 1e-4:
            out.append(geom)
    return out


def floor_to_plan(df, rng: np.random.Generator) -> VectorPlan:
    sub = df.entity_subtype
    et = df.entity_type
    walls = _geoms(df[(et == "separator") & (sub == "WALL")].geometry)
    walls_u = unary_union(walls)
    doors = [
        opening_box(g, walls_u)
        for g in _geoms(df[(et == "opening") & sub.isin(["DOOR", "ENTRANCE_DOOR"])].geometry)
    ]
    windows = [opening_box(g, walls_u) for g in _geoms(df[(et == "opening") & (sub == "WINDOW")].geometry)]
    rooms, ignore = [], []
    for s, g in zip(df[et == "area"].entity_subtype, df[et == "area"].geometry, strict=True):
        gs = _geoms([g])
        if not gs or s in NOT_DRAWN:
            continue
        if s in IGNORE:
            ignore += gs
            continue
        name = NAMES.get(s) or (str(rng.choice(ROOM_NAMES)) if s == "ROOM" else None)
        rooms.append((gs[0], name))
    fixtures = _geoms(df[(et == "feature") & sub.isin(FIXTURES)].geometry)
    railings = _geoms(df[(et == "separator") & (sub == "RAILING")].geometry)
    columns = _geoms(df[(et == "separator") & (sub == "COLUMN")].geometry)
    return VectorPlan(
        walls=walls, doors=doors, windows=windows, rooms=rooms, ignore=ignore,
        fixtures=fixtures, railings=railings, columns=columns,
    )


def iter_plans(
    data: Path, *, split: str = "train", limit: int | None = None, seed: int = 0
) -> Iterator[tuple[str, VectorPlan]]:
    import pandas as pd

    cols = ["site_id", "floor_id", "entity_type", "entity_subtype", "geometry"]
    df = pd.read_csv(Path(data) / CSV, usecols=cols)
    df = df.drop_duplicates(subset=["floor_id", "entity_type", "geometry"])
    sites = df.groupby("floor_id").site_id.first()
    floors = [f for f, s in sites.items() if split_of(s) == split]
    rng = np.random.default_rng(seed)
    rng.shuffle(floors)
    if limit:
        floors = floors[:limit]
    want = set(floors)
    groups = {f: g for f, g in df[df.floor_id.isin(want)].groupby("floor_id")}
    for f in floors:
        yield f"{int(f)}", floor_to_plan(groups[f], np.random.default_rng([int(f), 1]))
