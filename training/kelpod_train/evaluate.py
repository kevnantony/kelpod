"""Compare segmentation backends on the same held-out test split.

Synthetic split (``synth``): seeds 100000-100049 (never used for training, training seeds are
>= 1,000,000) at degrade 0.5 and 1.0. Per plan and backend: segmentation metrics in page px
(``tests/segment/synth_eval.py``: wall recall/precision, door/window recall, room recall/IoU),
then the deterministic geometry step on the same cached OCR and the plan-space metrics
(``kelpod.evaluation.metrics``: wall F1, room IoU, opening recall, closure).

Dataset splits (``swiss``, ``resplan``, ``cubicasa``: held-out pages written by
``kelpod_train.build`` into ``~/kelpod-data/test/<name>``): mask-level metrics only (no ground
truth graph): wall pixel IoU/F1, opening recall/precision (centre within 0.3 m, same kind), room
recall/IoU against the rooms the deterministic post-processor extracts from the ground-truth
class map.

    uv run python -m kelpod_train.evaluate --backends classical zeroshot learned \
        --checkpoint ~/kelpod-models/kelpod-seg/model.pt --splits synth swiss --out runs/eval
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[2]
DATA = Path.home() / "kelpod-data"
SYNTH_SEEDS = range(100000, 100050)
DEV_SEEDS = range(90000, 90010)
_BACKENDS: dict = {}
_CFG: dict = {}


def _synth_eval_module():
    spec = importlib.util.spec_from_file_location(
        "synth_eval", REPO / "tests" / "segment" / "synth_eval.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _backend(name: str):
    if name in _BACKENDS:
        return _BACKENDS[name]
    from kelpod.segment.classical import ClassicalSegmenter
    from kelpod.segment.learned import LearnedSegmenter

    if name == "classical":
        b = ClassicalSegmenter()
    elif name.startswith("zeroshot"):
        from kelpod_train.zeroshot import CubiCasaPredictor

        scale = float(name.split("@")[1]) if "@" in name else 1.0
        b = LearnedSegmenter(predictor=CubiCasaPredictor(scale=scale, device="cpu"))
        b.name = name
    elif name.startswith("learned"):
        ck = _CFG["checkpoints"][name]
        b = LearnedSegmenter(weights=ck, threads=_CFG.get("threads", 2))
    else:
        raise ValueError(name)
    _BACKENDS[name] = b
    return b


def _ocr_cached(key: str, page):
    from kelpod import ocr
    from kelpod.contracts import OcrAnnotations

    path = DATA / "cache" / "ocr" / f"{key}.json"
    if path.exists():
        return OcrAnnotations.model_validate_json(path.read_text())
    ann = ocr.run(page, backend="rapidocr")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ann.model_dump_json())
    return ann


def _synth_case(args) -> list[dict]:
    seed, degrade, backends = args
    from kelpod import geometry
    from kelpod.evaluation import synth
    from kelpod.evaluation.metrics import closure_metrics, compare_to_ground_truth

    se = _synth_eval_module()
    sp = synth.generate(seed, degrade=degrade)
    page, to_page = se.make_page(sp)
    ann = _ocr_cached(f"synth_{seed}_{degrade}", page) if _CFG.get("ocr", True) else None
    rows = []
    for name in backends:
        row = {"split": f"synth_d{degrade}", "case": seed, "backend": name}
        try:
            b = _backend(name)
            t = time.perf_counter()
            seg = b.segment(page)
            row["segment_s"] = time.perf_counter() - t
            r = se.evaluate(sp, seg, to_page, page)
            row.update({f"seg_{k}": v for k, v in r.items() if not isinstance(v, tuple)})
            t = time.perf_counter()
            graph = geometry.run(page, seg, ann)
            row["geometry_s"] = time.perf_counter() - t
            rep = closure_metrics(graph)
            cmp = compare_to_ground_truth(graph, sp.graph)
            row.update(
                closed=rep.is_closed,
                node_closure=rep.node_closure,
                room_closure=rep.room_closure,
                wall_f1=cmp.wall_f1,
                room_iou=cmp.room_iou_mean,
                room_recall=cmp.room_recall,
                door_recall=cmp.opening_recall.get("door"),
                window_recall=cmp.opening_recall.get("window"),
                opening_recall=cmp.opening_recall.get("any"),
            )
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
        rows.append(row)
    return rows


def _components(mask: np.ndarray, min_px: float):
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [tuple(cen[i]) for i in range(1, n) if max(st[i, 2], st[i, 3]) >= min_px]


def _room_masks(rooms, shape):
    out = []
    for r in rooms:
        m = np.zeros(shape, np.uint8)
        cv2.fillPoly(m, [np.round(np.array(r.polygon)).astype(np.int32)], 1)
        out.append(m.astype(bool))
    return out


def mask_metrics(seg, label: np.ndarray, page) -> dict:
    """Mask-level comparison with a ground-truth class map (dataset pages)."""
    from kelpod.segment.learned import DOOR, WALL, WINDOW, masks_to_segmentation

    valid = label != 255
    gt_w = (label == WALL) & valid
    pr_w = np.asarray(seg.wall_mask, bool) & valid
    inter = (gt_w & pr_w).sum()
    union = (gt_w | pr_w).sum()
    p = inter / max(pr_w.sum(), 1)
    r = inter / max(gt_w.sum(), 1)
    mpp = page.m_per_px_from_scale
    out = {
        "wall_iou": inter / max(union, 1),
        "wall_px_f1": 2 * p * r / max(p + r, 1e-9),
    }
    tol = 0.3 / mpp
    n_tp = n_pred = 0
    for kind, cls in (("door", DOOR), ("window", WINDOW)):
        gt = _components(label == cls, 0.3 / mpp)
        pr = [((o.p0[0] + o.p1[0]) / 2, (o.p0[1] + o.p1[1]) / 2) for o in seg.openings
              if str(o.kind) == kind]
        n_pred += len(pr)
        tp = 0
        if gt and pr:
            c = np.linalg.norm(np.array(gt)[:, None] - np.array(pr)[None], axis=-1)
            ri, ci = linear_sum_assignment(np.where(c <= tol, c, 1e6))
            tp = int(sum(c[a, b] <= tol for a, b in zip(ri, ci, strict=True)))
        n_tp += tp
        out[f"{kind}_recall"] = tp / len(gt) if gt else None
    out["opening_precision"] = n_tp / n_pred if n_pred else None
    gt_rooms = masks_to_segmentation(label, page).rooms
    g, q = _room_masks(gt_rooms, label.shape), _room_masks(seg.rooms, label.shape)
    if g and q:
        iou = np.zeros((len(g), len(q)))
        for i, a in enumerate(g):
            for j, b in enumerate(q):
                u = (a | b).sum()
                iou[i, j] = (a & b).sum() / u if u else 0
        ri, ci = linear_sum_assignment(-iou)
        out["room_iou"] = float(iou[ri, ci].sum() / len(g))
        out["room_recall"] = float((iou[ri, ci] >= 0.5).sum() / len(g))
        out["room_precision"] = float((iou[ri, ci] >= 0.5).sum() / len(q))
    else:
        out["room_iou"] = 0.0 if g else None
        out["room_recall"] = 0.0 if g else None
        out["room_precision"] = None
    return out


def _dataset_case(args) -> list[dict]:
    split, img_path, backends = args
    from kelpod.clean.core import clean_image
    from kelpod.contracts import RasterPage

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(str(img_path).replace("_img.png", "_lab.png"), cv2.IMREAD_GRAYSCALE)
    ink = clean_image(img, 200.0, deskew=False).ink
    page = RasterPage(image=img, ink=ink, dpi=200.0, source=str(img_path), plan_scale=100.0)
    rows = []
    for name in backends:
        row = {"split": split, "case": img_path.name.replace("_img.png", ""), "backend": name}
        try:
            t = time.perf_counter()
            seg = _backend(name).segment(page)
            row["segment_s"] = time.perf_counter() - t
            row.update(mask_metrics(seg, label, page))
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
        rows.append(row)
    return rows


def _init(cfg: dict) -> None:
    _CFG.update(cfg)
    os.environ["OMP_NUM_THREADS"] = str(cfg.get("threads", 2))
    cv2.setNumThreads(1)
    try:
        import torch

        torch.set_num_threads(cfg.get("threads", 2))
    except ImportError:
        pass


def summarise(rows: list[dict]) -> str:
    keys = [
        "wall_f1", "room_iou", "room_recall", "door_recall", "window_recall", "opening_recall",
        "closed", "seg_wall_recall", "seg_wall_precision", "seg_door_recall", "seg_window_recall",
        "seg_room_recall", "seg_room_iou", "wall_iou", "wall_px_f1", "opening_precision",
        "segment_s",
    ]
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["split"], r["backend"]), []).append(r)
    lines = []
    for split in sorted({s for s, _ in groups}):
        present = [k for k in keys if any(k in r for (s, _), rs in groups.items() if s == split
                                           for r in rs)]
        lines += [f"### {split}", "", "| backend | n | errors | " + " | ".join(present) + " |",
                  "|" + "---|" * (3 + len(present))]
        for (s, b), rs in sorted(groups.items()):
            if s != split:
                continue
            cells = []
            for k in present:
                vals = [float(r[k]) for r in rs if r.get(k) is not None and "error" not in r]
                cells.append(f"{statistics.mean(vals):.3f}" if vals else "–")
            n_err = sum("error" in r for r in rs)
            lines.append(f"| {b} | {len(rs)} | {n_err} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=["classical", "zeroshot"])
    ap.add_argument("--checkpoint", nargs="*", default=[],
                    help="learned checkpoints; backend names learned, learned2, ... in order")
    ap.add_argument("--splits", nargs="+", default=["synth"])
    ap.add_argument("--degrade", nargs="+", type=float, default=[0.5, 1.0])
    ap.add_argument("--seeds", default="test", choices=["test", "dev"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ckpts = {("learned" if i == 0 else f"learned{i + 1}"): c for i, c in enumerate(a.checkpoint)}
    cfg = {"checkpoints": ckpts, "threads": a.threads, "ocr": not a.no_ocr}
    jobs, fns = [], []
    for split in a.splits:
        if split == "synth":
            seeds = list(SYNTH_SEEDS if a.seeds == "test" else DEV_SEEDS)[: a.limit]
            jobs += [(s, d, a.backends) for d in a.degrade for s in seeds]
            fns += [_synth_case] * (len(seeds) * len(a.degrade))
        else:
            files = sorted((DATA / "test" / split).glob("*_img.png"))[: a.limit]
            jobs += [(split, f, a.backends) for f in files]
            fns += [_dataset_case] * len(files)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(cfg,)) as ex:
        futs = [ex.submit(f, j) for f, j in zip(fns, jobs, strict=True)]
        for i, fu in enumerate(futs):
            rows += fu.result()
            if (i + 1) % 20 == 0:
                print(f"{i + 1}/{len(futs)} cases, {time.time() - t0:.0f} s", flush=True)
    (out / "rows.json").write_text(json.dumps(rows, indent=1, default=float))
    md = summarise(rows)
    (out / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
