"""Run the pipeline on real web plans (no ground truth) and print closure / room count; optional
overlay PNGs of the segmentation. Inputs are never committed. Usage:
  uv run python scripts/web_plan_check.py IMG... [--out DIR]"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from kelpod import clean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--backend", default="classical")
    a = ap.parse_args()
    from kelpod.contracts import m_to_px
    from kelpod.pipeline import convert

    for f in a.images:
        t0 = time.perf_counter()
        res = convert(f, segment_backend=a.backend)
        g, page = res.graph, clean.run(f)[0]
        rep = g.report
        print(
            f"{Path(f).name}: page {page.width}x{page.height} walls={len(g.walls)} "
            f"rooms={len(g.rooms)} m/px={g.m_per_px:.4f} src={g.scale_source} "
            f"dangling={len(rep.dangling_nodes)} comps={rep.n_components} "
            f"closed={rep.is_closed} t={time.perf_counter() - t0:.1f}s"
        )
        print("   rooms m2:", [round(r.area_m2, 1) for r in g.rooms])
        if a.out:
            im = cv2.cvtColor(page.image, cv2.COLOR_GRAY2BGR)
            im = (im * 0.5 + 127).astype(np.uint8)
            P = {
                n.id: m_to_px(
                    n.xy if hasattr(n, "xy") else (n.x, n.y), g.m_per_px, g.image_height_px
                )
                for n in g.nodes
            }
            for r in g.rooms:
                pts = [m_to_px(q, g.m_per_px, g.image_height_px) for q in r.polygon]
                cv2.fillPoly(im, [np.round(pts).astype(np.int32)], (200, 235, 200))
            for w in g.walls:
                cv2.line(im, tuple(map(int, P[w.start])), tuple(map(int, P[w.end])), (255, 0, 0), 3)
            Path(a.out).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(a.out) / (Path(f).stem + "_ov.png")), im)


if __name__ == "__main__":
    main()
