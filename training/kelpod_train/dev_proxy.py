"""Dev proxy for style generalisation (experiment ``learned-v2``): per-class IoU of ONNX models on
the dev-proxy pages (``kelpod_train.build dev_proxy``: options reserved from training), at the
deployment setting of ``LearnedSegmenter`` (whole page, 0.65x).

    uv run python -m kelpod_train.dev_proxy --models v1-s1=~/kelpod-models/kelpod-seg/v1-s1/model.onnx \
        --data ~/kelpod-data/dev/proxy-v2 --out runs/dev_proxy.json

Also reports ``rooms_match``: the share of pages whose number of room regions (class 4, after a
small opening, area >= 0.5 m^2) equals the ground truth's, a cheap stand-in for the room-count
condition of correct-closed. Dev numbers (synthetic), never a locked evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CLASSES = ("background", "wall", "window", "door", "room")
IGNORE = 255
M_PER_PX = 0.0254 / 200 * 100


def _rooms(lab: np.ndarray) -> int:
    m = cv2.morphologyEx((lab == 4).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    return int(sum(st[i, cv2.CC_STAT_AREA] * M_PER_PX**2 >= 0.5 for i in range(1, n)))


def score(weights: Path, pages: list[Path], threads: int) -> dict:
    from kelpod.segment.learned import LearnedSegmenter

    seg = LearnedSegmenter(weights, threads=threads)
    inter, union = np.zeros(5), np.zeros(5)
    match = 0
    for ip in pages:
        img = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        lab = cv2.imread(str(ip.with_name(ip.name.replace("_img", "_lab"))), cv2.IMREAD_GRAYSCALE)
        pr = seg._predict(img)
        v = lab != IGNORE
        for c in range(5):
            a, b = (pr == c) & v, (lab == c) & v
            inter[c] += (a & b).sum()
            union[c] += (a | b).sum()
        match += _rooms(pr) == _rooms(lab)
    iou = inter / np.maximum(union, 1)
    out = {n: round(float(i), 4) for n, i in zip(CLASSES, iou, strict=True)}
    out["mean_wwd"] = round(float(np.mean([out["wall"], out["window"], out["door"]])), 4)
    out["rooms_match"] = round(match / len(pages), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=path.onnx")
    ap.add_argument("--data", default=str(Path.home() / "kelpod-data/dev/proxy-v2"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pages = sorted(Path(a.data).expanduser().glob("*_img.png"))[: a.limit]
    res = {}
    for spec in a.models:
        name, path = spec.split("=", 1)
        res[name] = score(Path(path).expanduser(), pages, a.threads)
        print(name, len(pages), json.dumps(res[name]), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
