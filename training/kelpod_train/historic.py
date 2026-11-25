"""Qualitative check on real historic plans (dev only, no ground truth, files never committed):
class-map overlays and pipeline summary per checkpoint.

    uv run python -m kelpod_train.historic --images ~/kelpod-data/real/user-feedback-2026-11-24 \
        --models r34=~/kelpod-models/kelpod-seg/r34-onnx/model.onnx v1-s1=... --out ~/kelpod-runs/historic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

COLOURS = np.array([[255, 255, 255], [0, 0, 255], [255, 0, 0], [0, 200, 0], [230, 200, 120]],
                   np.uint8)  # BGR: bg, wall red, window blue, door green, room light blue


def main() -> None:
    from kelpod import clean
    from kelpod.pipeline import convert
    from kelpod.segment.learned import LearnedSegmenter

    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--models", nargs="+", required=True, help="name=path")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = a.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for img in sorted(a.images.expanduser().glob("*.jpg")):
        page = clean.run(img, dpi=200.0, plan_scale=100.0)[0]
        gray = np.asarray(page.image, np.uint8)
        panels = [cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)]
        for spec in a.models:
            name, path = spec.split("=", 1)
            path = str(Path(path).expanduser())
            lab = LearnedSegmenter(weights=path).predict_labels(page)
            base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
            ov = (0.45 * base + 0.55 * COLOURS[np.clip(lab, 0, 4)]).astype(np.uint8)
            cv2.putText(ov, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            panels.append(ov)
            row = {"image": img.name, "model": name}
            try:
                import os

                os.environ["KELPOD_SEG_CHECKPOINT"] = path
                res = convert(img, None, segment_backend="learned", ocr_backend="none")
                rep = res.graph.report
                row.update(rooms=len(res.graph.rooms), walls=len(res.graph.walls),
                           dangling=len(rep.dangling_nodes) if rep else None,
                           closed=bool(rep and rep.is_closed))
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"[:200]
            rows.append(row)
            print(json.dumps(row), flush=True)
        cv2.imwrite(str(out / f"{img.stem}_overlay.png"), np.hstack(panels))
    (out / "summary.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
