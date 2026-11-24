"""CubiCasa5k (CC BY-NC 4.0) to Kelpod training pairs.

Each plan folder holds ``F1_scaled.png`` and ``model.svg`` in the same pixel frame. The SVG is
parsed with our own code (the dataset is untrusted; none of its code is imported or run):

* ``<g class="Space <Type>">`` polygons -> room (4), except outdoor/background types;
* ``<g id="Railing">`` polygons -> background (0);
* ``<g id="Wall">`` polygons -> wall (1);
* ``<g id="Window">`` / ``<g id="Door">`` polygons (nested in their wall) -> window (2) / door (3).
  The opening box is stretched across the wall thickness and clipped to its parent wall, so the
  gap always covers the full wall body even where the SVG box is a little thinner.

The page is then rescaled so the median wall thickness is ~20 px (Kelpod's 200-dpi 1:100 frame).

    uv run python -m kelpod_train.datasets.cubicasa --split train --out ~/kelpod-data/train/cubicasa
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from kelpod_train.labels import BG, DOOR, ROOM, WALL, WINDOW, wall_thickness_px

NS = "{http://www.w3.org/2000/svg}"
SUBSETS = {"high_quality_architectural": "hqa", "high_quality": "hq", "colorful": "col"}
NOT_ROOM = {"Outdoor", "Background"}  # Space types that are outside the building
TARGET_WALL_PX = 20.0
SCALE_CLAMP = (0.3, 4.0)
SHIFT = 4  # fillPoly fixed-point bits (sub-pixel vertices)
DEFAULT_ROOT = Path.home() / "kelpod-data" / "cubicasa5k" / "cubicasa5k"


def _points(g: ET.Element) -> np.ndarray | None:
    """First direct ``<polygon>`` child of a group as an (N, 2) float array."""
    pol = g.find(f"{NS}polygon")
    if pol is None:
        return None
    vals = pol.get("points", "").replace(",", " ").split()
    if len(vals) < 6:
        return None
    pts = np.asarray(vals[: len(vals) // 2 * 2], float).reshape(-1, 2)
    return pts if np.isfinite(pts).all() else None


def _fill(canvas: np.ndarray, pts: np.ndarray, value: int) -> None:
    # Pixel i covers [i, i + 1): shift by half a pixel so centres are tested.
    p = np.round((pts - 0.5) * (1 << SHIFT)).astype(np.int32)
    cv2.fillPoly(canvas, [p], int(value), lineType=cv2.LINE_8, shift=SHIFT)


def _across_wall(box: np.ndarray, wall: np.ndarray | None) -> np.ndarray:
    """Opening box stretched along its short axis (the wall normal) so it spans the wall."""
    if box.shape[0] != 4:
        return box
    c = box.mean(0)
    e1, e2 = box[1] - box[0], box[2] - box[1]
    short, long_ = (e1, e2) if np.linalg.norm(e1) < np.linalg.norm(e2) else (e2, e1)
    n = short / max(np.linalg.norm(short), 1e-9)
    reach = np.linalg.norm(short) / 2
    if wall is not None:  # far enough to cross the parent wall, never beyond the box length
        reach = max(reach, min(float(np.abs((wall - c) @ n).max()), np.linalg.norm(long_) / 2))
    off = (box - c) @ n
    sgn = np.where(off >= 0, 1.0, -1.0)
    return box + np.outer(sgn * reach - off, n)


def rasterise(svg: Path, shape: tuple[int, int]) -> np.ndarray:
    """Class map (image frame) of one CubiCasa5k ``model.svg``."""
    root = ET.parse(svg).getroot()
    rooms: list[np.ndarray] = []
    rails: list[np.ndarray] = []
    walls: list[np.ndarray] = []
    openings: list[tuple[int, np.ndarray, np.ndarray | None]] = []

    def walk(e: ET.Element, wall: np.ndarray | None) -> None:
        gid, cls = e.get("id", ""), e.get("class", "")
        pts = _points(e) if e.tag == f"{NS}g" else None
        if pts is not None:
            if gid == "Wall":
                walls.append(pts)
                wall = pts
            elif gid == "Railing":
                rails.append(pts)
            elif gid in ("Window", "Door"):
                openings.append((WINDOW if gid == "Window" else DOOR, pts, wall))
            elif cls.startswith("Space ") and cls.split()[1] not in NOT_ROOM:
                rooms.append(pts)
        for ch in e:
            walk(ch, wall)

    walk(root, None)
    lab = np.full(shape, BG, np.uint8)
    for p in rooms:
        _fill(lab, p, ROOM)
    for p in rails:
        _fill(lab, p, BG)
    wall_mask = np.zeros(shape, np.uint8)
    for p in walls:
        _fill(wall_mask, p, 1)
    lab[wall_mask > 0] = WALL
    for kind, box, wall in openings:
        m = np.zeros(shape, np.uint8)
        if wall is not None:  # clip the stretched box to its own wall polygon
            wm = np.zeros(shape, np.uint8)
            _fill(wm, wall, 1)
            _fill(m, _across_wall(box, wall), 1)
            m &= wm
        _fill(m, box, 1)
        lab[m > 0] = kind
    return lab


def _load_gray(path: Path) -> np.ndarray:
    """Grayscale uint8 page; transparent pixels become white paper."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"cannot read {path}")
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        a = img[..., 3:].astype(np.float32) / 255.0
        rgb = img[..., :3].astype(np.float32) * a + 255.0 * (1 - a)
        img = np.round(rgb).astype(np.uint8)
    return cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)


def sample_name(folder: str) -> str:
    """``/high_quality_architectural/1191/`` -> ``cubicasa_hqa_1191``."""
    subset, fid = folder.strip("/").split("/")[-2:]
    return f"cubicasa_{SUBSETS.get(subset, subset)}_{fid}"


def _job(args: tuple[Path, str, str, Path]) -> str:
    root, folder, split, out = args
    name = sample_name(folder)
    if (out / f"{name}_lab.png").exists():
        return name
    try:
        d = root / folder.strip("/")
        img = _load_gray(d / "F1_scaled.png")
        lab = rasterise(d / "model.svg", img.shape)
        t = wall_thickness_px(lab)
        if not math.isfinite(t) or t <= 0:
            return f"SKIP {name}: no walls"
        f = float(np.clip(TARGET_WALL_PX / t, *SCALE_CLAMP))
        h, w = img.shape
        size = (max(1, round(w * f)), max(1, round(h * f)))
        interp = cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, size, interpolation=interp)
        lab = cv2.resize(lab, size, interpolation=cv2.INTER_NEAREST)
        meta = {
            "source": "cubicasa",
            "folder": folder,
            "split": split,
            "scale": f,
            "wall_px_src": t,
            "wall_px": t * f,
        }
        cv2.imwrite(str(out / f"{name}_img.png"), img)
        cv2.imwrite(str(out / f"{name}_lab.png"), lab)
        (out / f"{name}.json").write_text(json.dumps(meta))
    except Exception:  # noqa: BLE001 - one bad plan must not stop the run
        return f"ERR {name}: {traceback.format_exc(limit=2)}"
    return name


def build_split(
    root: Path, split: str, out: Path, *, workers: int = 8, limit: int | None = None
) -> None:
    """Write every plan of ``<root>/<split>.txt`` as an image/label PNG pair into ``out``."""
    root, out = Path(root).expanduser(), Path(out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    folders = [s.strip() for s in (root / f"{split}.txt").read_text().splitlines() if s.strip()]
    if limit is not None:
        folders = folders[:limit]
    jobs = [(root, f, split, out) for f in folders]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cv2.setNumThreads(1)
    ok = bad = 0
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(_job, jobs, chunksize=4)):
            if r.startswith(("ERR", "SKIP")):
                bad += 1
                if bad <= 20:
                    print(r, flush=True)
            else:
                ok += 1
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(jobs)} ok={ok} bad={bad} {time.time() - t0:.0f}s", flush=True)
    print(f"done {split}: ok={ok} bad={bad} in {time.time() - t0:.0f}s -> {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    build_split(Path(a.root), a.split, Path(a.out), workers=a.workers, limit=a.limit)


if __name__ == "__main__":
    main()
