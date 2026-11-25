"""Loaders for the out-of-distribution splits: Swiss Dwellings, ResPlan, CubiCasa5k.

Licences (see ``CREDITS.md``): Swiss Dwellings v3.0.0 CC BY 4.0, ResPlan CC BY 4.0, CubiCasa5k
CC BY-NC 4.0. All three are parsed with our own code; none of their code is imported or run (the
ResPlan pickle is read with a restricted unpickler that allows three globals only).

Each ``*_item`` function returns the files of one split item (``plan.pdf``, ``plan.png``,
``ground_truth.json``) or ``None`` if the plan is unusable, deterministically in its arguments.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from shapely import affinity, wkt
from shapely.geometry import Polygon
from shapely.ops import unary_union

from kelpod.evaluation.vectorplans import (
    DPI,
    VECTOR_VERSION,
    VectorPlan,
    _clean,
    _polys,
    build_graph,
    render_plan,
)

DATA = Path.home() / "kelpod-data"
SWISS_CSV = DATA / "swiss-dwellings" / "swiss-dwellings-v3.0.0" / "geometries.csv"
RESPLAN_DIR = DATA / "resplan"
RESPLAN_SPLIT = DATA / "src" / "ResPlan" / "split.json"
CUBICASA_DIR = DATA / "cubicasa5k" / "cubicasa5k"
PDF_DATE = __import__("time").gmtime(946684800)
CUBICASA_DPI = 200.0

SWISS_NAMES = {
    "BATHROOM": "Bad", "BEDROOM": "Schlafen", "LIVING_ROOM": "Wohnen", "LIVING_DINING": "Wohnen",
    "KITCHEN": "Küche", "KITCHEN_DINING": "Küche", "DINING": "Essen", "CORRIDOR": "Flur",
    "CORRIDORS_AND_HALLS": "Flur", "STOREROOM": "Abst.", "STAIRCASE": "Treppe", "OFFICE": "Büro",
}  # fmt: skip
SWISS_ROOM_NAMES = ("Zimmer", "Kind", "Schlafen", "Arbeiten", "Gast")
SWISS_NOT_ROOMS = {
    "BALCONY", "LOGGIA", "TERRACE", "GARDEN", "PATIO", "WINTERGARTEN", "ARCADE", "SHAFT", "VOID",
    "OUTDOOR_VOID", "LIGHTWELL", "AIR", "ELEVATOR", "TRANSPORT_SHAFT", "ELEVATOR_FACILITIES",
    "VEHICLE_TRAFFIC_AREA", "CARPARK", "NOT_DEFINED",
}  # fmt: skip
RESPLAN_NAMES = {
    "bathroom": "Bad", "bedroom": "Schlafen", "kitchen": "Küche", "living": "Wohnen",
    "storage": "Abst.", "stair": "Treppe",
}  # fmt: skip
CUBICASA_NAMES = {
    "Kitchen": "Küche", "LivingRoom": "Wohnen", "Bedroom": "Schlafen", "Bath": "Bad",
    "Entrance": "Diele", "Hall": "Flur", "Storage": "Abst.", "Closet": "Abst.",
    "Dining": "Essen", "Garage": "Garage",
}  # fmt: skip


def sha_int(*parts: object) -> int:
    return int(hashlib.sha256("/".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def write_item(directory: Path, image: np.ndarray, graph, meta: dict, dpi: float = DPI) -> Path:
    """``plan.pdf`` (image-only, fixed dates), ``plan.png`` and ``ground_truth.json``."""
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(image, mode="L")
    img.save(directory / "plan.png", dpi=(dpi, dpi))
    img.save(
        directory / "plan.pdf", resolution=dpi, quality=95, creationDate=PDF_DATE, modDate=PDF_DATE
    )
    gt = {
        "renderer_version": VECTOR_VERSION,
        "dpi": dpi,
        **meta,
        "graph": json.loads(graph.model_dump_json()),
    }
    (directory / "ground_truth.json").write_text(json.dumps(gt, indent=1), encoding="utf-8")
    return directory


# --------------------------------------------------------------------------------------------
# Swiss Dwellings
# --------------------------------------------------------------------------------------------


def _geoms(rows) -> list:
    out = []
    for g in rows:
        try:
            geom = wkt.loads(g)
        except Exception:  # noqa: BLE001 - malformed rows are skipped
            continue
        geom = _clean(geom)
        if not geom.is_empty and geom.area > 1e-4:
            out.append(geom)
    return out


def swiss_rows(
    floors: set[int], csv_path: Path = SWISS_CSV
) -> dict[int, list[tuple[str, str, str]]]:
    """(entity_type, entity_subtype, geometry) rows of the given floors, duplicates removed."""
    csv.field_size_limit(1 << 30)
    out: dict[int, list[tuple[str, str, str]]] = {f: [] for f in floors}
    seen: set[tuple[int, str, str]] = set()
    with open(csv_path, newline="") as f:
        r = csv.reader(f)
        head = next(r)
        i_f, i_t, i_s, i_g = (
            head.index(c) for c in ("floor_id", "entity_type", "entity_subtype", "geometry")
        )
        for row in r:
            fl = int(row[i_f])
            if fl in out:
                key = (fl, row[i_t], row[i_g])
                if key not in seen:
                    seen.add(key)
                    out[fl].append((row[i_t], row[i_s], row[i_g]))
    return out


def swiss_floor_plan(floor_id: int, rows: list[tuple[str, str, str]]) -> VectorPlan:
    def sel(et: str, subs) -> list:
        return _geoms([g for t, s, g in rows if t == et and s in subs])

    rng = np.random.default_rng([floor_id, 0x73776973])
    walls = [p for g in sel("separator", {"WALL"}) for p in _polys(g)]
    rooms: list[tuple[Polygon, str | None]] = []
    for t, s, g in rows:
        if t != "area" or s in SWISS_NOT_ROOMS:
            continue
        gs = _geoms([g])
        if gs:
            name = SWISS_NAMES.get(s) or (
                str(rng.choice(SWISS_ROOM_NAMES)) if s == "ROOM" else None
            )
            rooms.append((gs[0], name))
    return VectorPlan(
        walls=walls,
        doors=[p for g in sel("opening", {"DOOR", "ENTRANCE_DOOR"}) for p in _polys(g)],
        windows=[p for g in sel("opening", {"WINDOW"}) for p in _polys(g)],
        rooms=rooms,
        meta={"source": "swiss-dwellings", "floor_id": floor_id},
    )


# --------------------------------------------------------------------------------------------
# ResPlan
# --------------------------------------------------------------------------------------------

_ALLOWED = {("shapely.io", "from_wkb"), ("numpy._core.multiarray", "scalar"), ("numpy", "dtype")}


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) not in _ALLOWED:
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")
        return super().find_class(module, name)


def resplan_load(directory: Path = RESPLAN_DIR) -> dict[int, dict]:
    with open(Path(directory) / "ResPlan.pkl", "rb") as f:
        return {int(p["id"]): p for p in _SafeUnpickler(f).load()}


def resplan_plan(p: dict) -> VectorPlan | None:
    """A ResPlan entry in metres: plan units are scaled so that ``wall_depth`` becomes a wall
    thickness drawn uniformly from 0.20-0.30 m (seeded by the plan id; the dataset has no metric
    scale). Rooms: the named ones plus the unnamed rest of ``inner`` as ``Flur``."""
    wd = float(p.get("wall_depth") or 0)
    if wd <= 0 or p.get("wall") is None or p["wall"].is_empty:
        return None
    s = float(np.random.default_rng([int(p["id"]), 0x72657370]).uniform(0.20, 0.30)) / wd

    def get(key):
        g = p.get(key)
        return (
            []
            if g is None or g.is_empty
            else [affinity.scale(x, s, -s, origin=(0, 0)) for x in _polys(_clean(g))]
        )

    named = [(g, n) for k, n in RESPLAN_NAMES.items() for g in get(k)]
    inner = unary_union(get("inner")) if get("inner") else Polygon()
    rest = inner.difference(unary_union([g for g, _ in named]).buffer(0.02)) if named else inner
    rest = rest.buffer(-0.12).buffer(0.12)
    rooms = named + [(g, "Flur") for g in _polys(rest) if g.area >= 2.0]
    return VectorPlan(
        walls=get("wall"), doors=get("door") + get("front_door"), windows=get("window"),
        rooms=rooms, meta={"source": "resplan", "plan_id": int(p["id"]), "union_walls": True,
                           "stats": {"wall_scale_m_per_unit": round(s, 5)}},
    )  # fmt: skip


# --------------------------------------------------------------------------------------------
# CubiCasa5k (real raster pages, scale-free ground truth)
# --------------------------------------------------------------------------------------------

NS = "{http://www.w3.org/2000/svg}"
TARGET_WALL_PX = 20.0
GT_M_PER_PX = 0.2 / TARGET_WALL_PX
"""Pseudo-scale of the CubiCasa ground truth: the page is resized so that the median wall is
20 px and then taken as 0.20 m. It is *not* a measured scale (CubiCasa5k has none); it only fixes
the units of the GT file and is never used for scoring (``scoring = "scale-free"``)."""


def _svg_points(g: ET.Element) -> np.ndarray | None:
    pol = g.find(f"{NS}polygon")
    if pol is None:
        return None
    vals = pol.get("points", "").replace(",", " ").split()
    if len(vals) < 6:
        return None
    pts = np.asarray(vals[: len(vals) // 2 * 2], float).reshape(-1, 2)
    return pts if np.isfinite(pts).all() else None


def cubicasa_parse(svg: Path) -> dict:
    """Walls, openings and spaces of ``model.svg`` as polygons in svg px (y down)."""
    root = ET.parse(svg).getroot()
    out: dict = {"walls": [], "doors": [], "windows": [], "rooms": []}

    def walk(e: ET.Element) -> None:
        gid, cls = e.get("id", ""), e.get("class", "")
        pts = _svg_points(e) if e.tag == f"{NS}g" else None
        if pts is not None:
            poly = _clean(Polygon(pts))
            if not poly.is_empty:
                if gid == "Wall":
                    out["walls"].append(poly)
                elif gid == "Door":
                    out["doors"].append(poly)
                elif gid == "Window":
                    out["windows"].append(poly)
                elif cls.startswith("Space ") and cls.split()[1] not in {"Outdoor", "Background"}:
                    out["rooms"].append((poly, cls.split()[1]))
        for ch in e:
            walk(ch)

    walk(root)
    return out


def cubicasa_item(folder: str, root: Path = CUBICASA_DIR) -> tuple[np.ndarray, object, dict] | None:
    """(page, GT graph, meta) of one CubiCasa5k plan folder, or ``None``. The page is the real
    ``F1_scaled.png`` (grayscale, transparent = white) resized so that the median wall is 20 px."""
    d = root / folder.strip("/")
    raw = cv2.imread(str(d / "F1_scaled.png"), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    if raw.dtype != np.uint8:
        raw = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if raw.ndim == 3 and raw.shape[2] == 4:
        a = raw[..., 3:].astype(np.float32) / 255.0
        raw = np.round(raw[..., :3].astype(np.float32) * a + 255.0 * (1 - a)).astype(np.uint8)
    img = raw if raw.ndim == 2 else cv2.cvtColor(raw[..., :3], cv2.COLOR_BGR2GRAY)
    svg = cubicasa_parse(d / "model.svg")
    if len(svg["walls"]) < 4:
        return None
    thick = float(np.median([2 * w.area / max(w.length, 1e-9) for w in svg["walls"]]))
    if not math.isfinite(thick) or thick <= 0:
        return None
    f = float(np.clip(TARGET_WALL_PX / thick, 0.3, 4.0))
    h0, w0 = img.shape
    size = (max(1, round(w0 * f)), max(1, round(h0 * f)))
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
    h = img.shape[0]

    def to_plan(g):  # svg px -> resized px -> pseudo-metres, y up
        g = affinity.scale(g, f * GT_M_PER_PX, -f * GT_M_PER_PX, origin=(0, 0))
        return affinity.translate(g, 0, h * GT_M_PER_PX)

    plan = VectorPlan(
        walls=[to_plan(g) for w in svg["walls"] for g in _polys(w)],
        doors=[to_plan(g) for g in svg["doors"]],
        windows=[to_plan(g) for g in svg["windows"]],
        rooms=[(to_plan(g), CUBICASA_NAMES.get(n)) for g, n in svg["rooms"]],
        meta={"source": "cubicasa5k", "folder": folder.strip("/")},
    )
    built = build_graph(plan, m_per_px=GT_M_PER_PX, image_height_px=h)
    if len(built.graph.walls) < 4 or len(built.graph.rooms) < 2:
        return None
    meta = {
        "source": "cubicasa5k", "folder": folder.strip("/"), "scoring": "scale-free",
        "metric_scale": False, "resize_factor": round(f, 4), "median_wall_px": round(thick, 2),
        "stats": built.stats,
    }  # fmt: skip
    return img, built.graph, meta


# --------------------------------------------------------------------------------------------
# one item of a dataset split
# --------------------------------------------------------------------------------------------


def vector_item(plan: VectorPlan, *, seed: int, degrade: float, directory: Path) -> dict | None:
    """Render ``plan`` into ``directory``; returns the manifest fields of the item or ``None``."""
    item = render_plan(plan, seed=seed, degrade=degrade)
    if item is None:
        return None
    write_item(directory, item.image, item.graph, {"source_meta": plan.meta, "stats": item.stats})
    return {"stats": item.stats, "n_rooms": len(item.graph.rooms)}
