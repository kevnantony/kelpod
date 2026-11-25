"""Write training samples to disk as PNG pairs: ``<name>_img.png`` (cleaned page, 200 dpi) and
``<name>_lab.png`` (class map, see :mod:`kelpod_train.labels`).

    uv run python -m kelpod_train.build synth --out ~/kelpod-data/train/synth --start 1000000 --n 4000
    uv run python -m kelpod_train.build swiss --out ~/kelpod-data/train/swiss --n 3000
    uv run python -m kelpod_train.build resplan --out ~/kelpod-data/train/resplan --n 3000
    uv run python -m kelpod_train.build cubicasa --out ~/kelpod-data/train/cubicasa --split train

Style-randomised sources of experiment learned-v1 (``render_v1`` / ``degrade_v1``, train-style
options only, see ``style_v1``)::

    uv run python -m kelpod_train.build synth_v1 --out ~/kelpod-data/train/synth_v1 --start 1100000 --n 6000
    uv run python -m kelpod_train.build swiss_v1 --out ~/kelpod-data/train/swiss_v1 --start 5000000 --n 3000
    uv run python -m kelpod_train.build resplan_v1 --out ~/kelpod-data/train/resplan_v1 --start 6000000 --n 3000

Seeds >= 1,000,000 are training plans; the held-out synthetic test seeds are 100000-100049
(see ``kelpod_train.evaluate``), so they never overlap.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def _write(out: Path, name: str, image: np.ndarray, label: np.ndarray, meta: dict) -> None:
    cv2.imwrite(str(out / f"{name}_img.png"), image)
    cv2.imwrite(str(out / f"{name}_lab.png"), label)
    (out / f"{name}.json").write_text(json.dumps(meta))


def _synth_job(args) -> str:
    seed, out = args
    from kelpod_train.labels import synth_sample

    name = f"synth_{seed}"
    if (out / f"{name}_lab.png").exists():
        return name
    rng = np.random.default_rng([seed, 7])
    degrade = float(rng.choice([0.0, rng.uniform(0.2, 1.0), 1.0], p=[0.1, 0.7, 0.2]))
    try:
        s = synth_sample(seed, degrade)
    except Exception:  # noqa: BLE001
        return f"ERR {seed}: {traceback.format_exc(limit=1)}"
    _write(out, name, s.image, s.label, s.meta)
    return name


def _synth_v1_job(args) -> str:
    seed, out = args
    from kelpod_train.render_v1 import styled_sample, synth_vector_plan

    name = f"synth_{seed}"
    if (out / f"{name}_lab.png").exists():
        return name
    try:
        s = styled_sample(synth_vector_plan(seed), seed=seed)
    except Exception:  # noqa: BLE001
        return f"ERR {seed}: {traceback.format_exc(limit=2)}"
    if s is None:
        return f"SKIP {name}"
    s.meta.update(source="synth_v1")
    _write(out, name, s.image, s.label, s.meta)
    return name


def _vector_v1_job(args) -> str:
    kind, idx, plan, out, seed = args
    from kelpod_train.render_v1 import styled_sample

    name = f"{kind}_{idx}"
    if (out / f"{name}_lab.png").exists():
        return name
    try:
        s = styled_sample(plan, seed=seed)
    except Exception:  # noqa: BLE001
        return f"ERR {name}: {traceback.format_exc(limit=2)}"
    if s is None:
        return f"SKIP {name}"
    s.meta.update(source=f"{kind}_v1", index=idx)
    _write(out, name, s.image, s.label, s.meta)
    return name


def _vector_job(args) -> str:
    kind, idx, plan, out, seed = args
    from kelpod_train.render import render_sample

    name = f"{kind}_{idx}"
    if (out / f"{name}_lab.png").exists():
        return name
    try:
        s = render_sample(plan, seed=seed)
    except Exception:  # noqa: BLE001
        return f"ERR {name}: {traceback.format_exc(limit=2)}"
    if s is None:
        return f"SKIP {name}"
    s.meta.update(source=kind, index=idx)
    _write(out, name, s.image, s.label, s.meta)
    return name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["synth", "swiss", "resplan", "cubicasa",
                                       "synth_v1", "swiss_v1", "resplan_v1"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=1_000_000)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--split", default="train")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--data", default=str(Path.home() / "kelpod-data"))
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cv2.setNumThreads(1)

    if a.source == "synth":
        jobs = [(s, out) for s in range(a.start, a.start + a.n)]
        fn = _synth_job
    elif a.source == "synth_v1":  # style-randomised (learned-v1); train seeds from 1,100,000
        jobs = [(s, out) for s in range(a.start, a.start + a.n)]
        fn = _synth_v1_job
    elif a.source == "cubicasa":
        from kelpod_train.datasets.cubicasa import build_split

        build_split(Path(a.data) / "cubicasa5k" / "cubicasa5k", a.split, out, workers=a.workers)
        return
    else:
        base = a.source.removesuffix("_v1")
        if base == "swiss":
            from kelpod_train.datasets.swiss import iter_plans
        else:
            from kelpod_train.datasets.resplan import iter_plans
        plans = list(iter_plans(Path(a.data), split=a.split, limit=a.n))
        jobs = [(base, pid, plan, out, a.start + i) for i, (pid, plan) in enumerate(plans)]
        fn = _vector_v1_job if a.source.endswith("_v1") else _vector_job
    ok = bad = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(fn, jobs, chunksize=4)):
            if r.startswith(("ERR", "SKIP")):
                bad += 1
                if bad <= 20:
                    print(r, flush=True)
            else:
                ok += 1
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(jobs)} ok={ok} bad={bad}", flush=True)
    print(f"done: ok={ok} bad={bad} -> {out}")


if __name__ == "__main__":
    main()
