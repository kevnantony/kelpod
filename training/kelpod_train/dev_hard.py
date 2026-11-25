"""Hard dev proxy, end to end (experiment ``learned-v2``).

Plans from the synthetic generator's layout and renderer (``kelpod.evaluation.synth``, train
geometry, seeds 950,000+; never a held-out option) drawn with options that appear in no train list
and no held-out list: a font from ``style_v2.DEV_FONTS``, 45-degree hatching at 3.4-4.5 mm (train
range 1.3-3.0 mm), line scale 1.0, a damage kind from ``style_v2.DEV_DAMAGE``, scan degradation
0.6-1.2, and with some probability extra low resolution (150-220 dpi), JPEG quality 25-60 and blur.
Written like the generator's own plans (``plan.pdf`` image-only, ``ground_truth.json``), then
scored with the eval branch's panel through the real pipeline (``kelpod_train.val_synth._one``):
correct-closed and the rest of the panel, per model checkpoint.

    uv run python -m kelpod_train.dev_hard build --out ~/kelpod-data/dev/hard-e2e --n 100
    uv run python -m kelpod_train.dev_hard score --data ~/kelpod-data/dev/hard-e2e \
        --models v1-s1=<onnx> v1-s2=<onnx> --out runs/dev_hard.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import types
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

START = 950_000


def make_plan(seed: int):
    from kelpod.evaluation.synth import SyntheticPlan, save  # noqa: F401
    from kelpod.evaluation.synth.render import Style, render
    from kelpod.evaluation.synth.layout import make_layout
    from kelpod_train import style_v2 as S2
    from kelpod_train.degrade_v1 import damage

    dpi, scale = 300.0, 100.0
    m_per_px = 0.0254 / dpi * scale
    rng = np.random.default_rng([seed, 0x6B656C70])
    from kelpod.evaluation.synth import MARGIN_M

    layout, size = make_layout(rng, margin_m=MARGIN_M, m_per_px=m_per_px)
    r = np.random.default_rng([seed, 0x68617264])
    font = str(r.choice(sorted(S2.DEV_FONTS)))
    spacing = float(r.uniform(3.4, 4.5))
    style = Style(length=str(rng.choice(["cm", "sup", "m"])), sill=str(rng.choice(["m", "cm"])),
                    font=str(S2.FONT_DIR / S2.DEV_FONTS[font]), hatch="diag45",
                    hatch_spacing_mm=spacing, line_scale=1.0)
    clean, wall_mask, drawn = render(layout, size, dpi, style, m_per_px)
    deg = float(r.uniform(0.6, 1.2))
    from kelpod.evaluation.synth.degrade import degrade as scan_degrade

    img, info = scan_degrade(clean, deg, np.random.default_rng([seed, 0x64656772]))
    kind = str(r.choice(S2.DEV_DAMAGE))
    ts = types.SimpleNamespace(damage=kind, damage_strength=float(r.uniform(0.6, 1.0)))
    img = damage(img, ts, np.random.default_rng([seed, 0x646D6767]), seed)
    extra = {"font": font, "hatch_spacing_mm": spacing, "damage": kind, "degrade": deg}
    if r.random() < 0.5:  # low-resolution scan
        d = float(r.uniform(150, 220))
        h, w = img.shape
        small = cv2.resize(img, (round(w * d / dpi), round(h * d / dpi)), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        extra["scan_dpi"] = d
    if r.random() < 0.4:
        q = int(r.integers(25, 61))
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1],
                           cv2.IMREAD_GRAYSCALE)
        extra["jpeg"] = q
    if r.random() < 0.4:
        sg = float(r.uniform(0.8, 1.4))
        img = cv2.GaussianBlur(img, (0, 0), sg)
        extra["blur"] = sg
    from kelpod.evaluation.metrics import closure_metrics
    from kelpod.evaluation.synth import TextAnnotation

    ann = [TextAnnotation(text=d.text, kind=d.kind, quad=d.quad, angle_deg=d.angle_deg,
                          value=d.value, line=d.line, room_id=d.room_id, opening_id=d.opening_id)
           for d in drawn]
    g = layout.graph
    g.report = closure_metrics(g)
    plan = SyntheticPlan(seed=seed, image=img.astype(np.uint8), clean_image=clean, wall_mask=wall_mask,
                         dpi=dpi, plan_scale=scale, graph=g, annotations=ann, degradation=info,
                         length_style=style.length, sill_style=style.sill, style=None)
    return plan, extra


def _build_one(a):
    seed, out = a
    from kelpod.evaluation.synth import save

    d = out / f"h{seed}"
    if (d / "ground_truth.json").exists():
        return seed
    plan, extra = make_plan(seed)
    save(plan, d)
    (d / "hard_style.json").write_text(json.dumps(extra))
    return seed


def _score_one(a):
    item, model, name = a
    os.environ["KELPOD_SEG_CHECKPOINT"] = str(Path(model).expanduser())
    from kelpod_train.val_synth import _one

    return _one((item, item.name, "learned")) | {"model": name}


def _init():
    os.environ["OMP_NUM_THREADS"] = "1"
    cv2.setNumThreads(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "score"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--data", default=str(Path.home() / "kelpod-data/dev/hard-e2e"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "build":
        out = Path(a.out or a.data).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(a.workers, initializer=_init) as ex:
            list(ex.map(_build_one, [(s, out) for s in range(START, START + a.n)]))
        print("built", a.n, out)
        return
    items = sorted(Path(a.data).expanduser().glob("h*"))
    jobs = []
    for spec in a.models:
        name, path = spec.split("=", 1)
        jobs += [(it, path, name) for it in items]
    rows = []
    with ProcessPoolExecutor(a.workers, initializer=_init) as ex:
        rows = list(ex.map(_score_one, jobs))
    res = {}
    for spec in a.models:
        name = spec.split("=", 1)[0]
        rs = [r for r in rows if r["model"] == name]
        keys = ("correct_closed", "closed", "room_iou_mean", "wall_f1_10cm", "edit_cost")
        res[name] = {"n": len(rs), "errors": sum("error" in r for r in rs)}
        for k in sorted({k for r in rs for k, v in r["panel"].items() if isinstance(v, (int, float))}):
            vals = [float(r["panel"][k]) for r in rs if r["panel"].get(k) is not None]
            if vals and (k in keys or "correct" in k or "iou" in k):
                res[name][k] = round(statistics.mean(vals), 4)
        print(name, json.dumps(res[name]), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"summary": res, "rows": rows}, default=float))


if __name__ == "__main__":
    main()
