"""Dev tool: hand-assisted annotation of real plans (rooms, walls, scale) in ORIGINAL pixel space.

Not part of the pipeline and never imported by it. It deliberately does not use the pipeline's
segmenter: the prefill is an independent, crude orthogonal-stroke extractor, so that the labels are
not biased towards what the pipeline already finds. The workflow is: ``prefill`` -> look at the
overlay (``render``) -> correct the walls/spaces in a label JSON -> ``build`` (derives the room
polygons by flood fill from the seed points) -> ``render`` again until the overlay matches.

Label JSON (pixel coordinates of the original image, y down)::

    {"image": "x.jpg", "size": [w, h], "m_per_px": 0.02, "scale_note": "...",
     "walls": [{"p": [x, y], "q": [x, y], "t": px}],
     "cuts": [[x0, y0, x1, y1]],            # thin lines closing doors/arches for flood fill only
     "spaces": [{"name": "Küche", "kind": "room"|"exterior", "seed": [x, y], "polygon": [[x, y], ...]}],
     "openings": [{"kind": "door"|"window", "p": [x, y], "q": [x, y]}],
     "plans": [{"caption": "...", "bbox": [x0, y0, x1, y1], "labelled": true}]}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

_PALETTE = [(255, 120, 0), (0, 160, 0), (200, 0, 200), (0, 140, 255), (160, 160, 0), (0, 0, 220)]


def load_gray(path: str | Path) -> np.ndarray:
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise FileNotFoundError(path)
    return g


def prefill_walls(gray: np.ndarray, k: int = 5, run: int = 25, min_len: int = 20) -> list[dict]:
    """Crude orthogonal poché walls: dark strokes at least ``k`` px thick and ``run`` px long."""
    ink = (gray < 140).astype(np.uint8)
    walls: list[dict] = []
    for horizontal in (True, False):
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (run, k) if horizontal else (k, run))
        m = cv2.morphologyEx(ink, cv2.MORPH_OPEN, ker)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            x, y, w, h, a = st[i]
            length, thick = (w, h) if horizontal else (h, w)
            if length < min_len:
                continue
            t = a / max(length, 1)
            if horizontal:
                walls.append({"p": [x, y + h / 2], "q": [x + w, y + h / 2], "t": round(float(t), 1)})
            else:
                walls.append({"p": [x + w / 2, y], "q": [x + w / 2, y + h], "t": round(float(t), 1)})
    return walls


def wall_mask(labels: dict, shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    for w in labels["walls"]:
        cv2.line(m, tuple(int(round(v)) for v in w["p"]), tuple(int(round(v)) for v in w["q"]),
                 1, max(1, int(round(w["t"]))))
    for c in labels.get("cuts", []):
        cv2.line(m, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), 1, 2)
    return m


def build_spaces(labels: dict) -> dict:
    """Fill ``polygon`` of every space by flood fill from its seed in the free space."""
    w, h = labels["size"]
    free = (wall_mask(labels, (h, w)) == 0).astype(np.uint8)
    n, lab = cv2.connectedComponents(free, connectivity=4)
    for sp in labels["spaces"]:
        sx, sy = int(sp["seed"][0]), int(sp["seed"][1])
        comp = lab[sy, sx]
        if comp == 0:
            sp["polygon"], sp["note"] = [], "seed on a wall"
            continue
        mask = (lab == comp).astype(np.uint8)
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, 1.5, True)[:, 0, :]
        sp["polygon"] = [[int(x), int(y)] for x, y in c]
        sp["area_px"] = int(mask.sum())
        sp["touches_border"] = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    return labels


def render(gray: np.ndarray, labels: dict, crop=None, zoom: float = 1.0, ids: bool = True) -> np.ndarray:
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    ov = img.copy()
    for i, sp in enumerate(labels.get("spaces", [])):
        if sp.get("polygon"):
            col = (200, 200, 255) if sp["kind"] == "exterior" else _PALETTE[i % len(_PALETTE)]
            cv2.fillPoly(ov, [np.array(sp["polygon"], np.int32)], col)
    img = cv2.addWeighted(ov, 0.35, img, 0.65, 0)
    x0, y0, x1, y1 = crop or (0, 0, gray.shape[1], gray.shape[0])
    img = img[y0:y1, x0:x1]
    img = cv2.resize(img, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)

    def tz(p):
        return int(round((p[0] - x0) * zoom)), int(round((p[1] - y0) * zoom))

    for i, w in enumerate(labels.get("walls", [])):
        cv2.line(img, tz(w["p"]), tz(w["q"]), (0, 0, 255), 1)
        if ids:
            cv2.putText(img, str(i), tz(((w["p"][0] + w["q"][0]) / 2, (w["p"][1] + w["q"][1]) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
    for c in labels.get("cuts", []):
        cv2.line(img, tz(c[:2]), tz(c[2:]), (0, 200, 0), 1)
    for o in labels.get("openings", []):
        cv2.line(img, tz(o["p"]), tz(o["q"]), (0, 160, 255) if o["kind"] == "door" else (255, 160, 0), 2)
    for sp in labels.get("spaces", []):
        cv2.putText(img, sp["name"][:14], tz(sp["seed"]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    for p in labels.get("plans", []):
        cv2.rectangle(img, tz(p["bbox"][:2]), tz(p["bbox"][2:]), (0, 120, 0), 2)
    step = 50 if zoom <= 2 else 20
    for gx in range((x0 // step + 1) * step, x1, step):
        cv2.line(img, tz((gx, y0)), tz((gx, y0 + 6)), (0, 0, 0), 1)
        cv2.putText(img, str(gx), tz((gx - 8, y0 + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    for gy in range((y0 // step + 1) * step, y1, step):
        cv2.line(img, tz((x0, gy)), tz((x0 + 6, gy)), (0, 0, 0), 1)
        cv2.putText(img, str(gy), tz((x0 + 8, gy + 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    return img


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prefill")
    p.add_argument("image"), p.add_argument("out")
    p.add_argument("--k", type=int, default=5), p.add_argument("--run", type=int, default=25)
    b = sub.add_parser("build")
    b.add_argument("labels")
    r = sub.add_parser("render")
    r.add_argument("image"), r.add_argument("labels"), r.add_argument("out")
    r.add_argument("--crop", type=int, nargs=4), r.add_argument("--zoom", type=float, default=1.0)
    r.add_argument("--no-ids", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "prefill":
        g = load_gray(a.image)
        lab = {"image": Path(a.image).name, "size": [g.shape[1], g.shape[0]], "m_per_px": None,
               "walls": prefill_walls(g, a.k, a.run), "cuts": [], "spaces": [], "openings": [], "plans": []}
        Path(a.out).write_text(json.dumps(lab, indent=1))
    elif a.cmd == "build":
        lab = build_spaces(json.loads(Path(a.labels).read_text()))
        Path(a.labels).write_text(json.dumps(lab, indent=1, ensure_ascii=False))
    else:
        lab = json.loads(Path(a.labels).read_text())
        cv2.imwrite(a.out, render(load_gray(a.image), lab, a.crop, a.zoom, not a.no_ids))


if __name__ == "__main__":
    main()
