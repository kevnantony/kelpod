"""Training manifests in the eval branch's format (``eval/README.md``, "Training manifests"), so
``scripts/locked_eval.py --verify-manifests`` can check overlap with the frozen test splits.

    uv run python -m kelpod_train.manifest            # all sources under ~/kelpod-data/train

Writes ``training/manifests/<source>.json``::

    {"source": "synth-train", "files": [{"path": "synth_1000000_img.png", "sha256": "...",
      "phash": "<16 hex>", "seed": 1000000, "style": "train"}, ...]}

plus per source ``site_id`` / ``building_id`` / ``floor_id`` (Swiss Dwellings),
``cubicasa_id`` (e.g. ``high_quality/1234``) or ``resplan_id``. Every image and label file is
listed with its SHA-256; the pHash is computed on the image.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

DATA = Path.home() / "kelpod-data"
OUT = Path(__file__).resolve().parents[1] / "manifests"


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _entries(args) -> list[dict]:
    source, img, ids = args
    from kelpod.evaluation.splits import phash

    lab = img.with_name(img.name.replace("_img.png", "_lab.png"))
    meta_p = img.with_name(img.name.replace("_img.png", ".json"))
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    key = img.name.replace("_img.png", "")
    extra: dict = {}
    if source == "synth":
        extra = {"seed": int(meta.get("seed", key.split("_")[1])), "style": "train",
                 "degrade": meta.get("degrade")}
    elif source == "swiss":
        extra = dict(ids.get(int(key.split("_")[1]), {}))
    elif source == "cubicasa":
        folder = meta.get("folder", "").strip("/")
        extra = {"cubicasa_id": folder}
    elif source == "resplan":
        extra = {"resplan_id": int(key.split("_")[1])}
    out = [{"path": img.name, "sha256": _sha(img), "phash": phash(img), **extra}]
    if lab.exists():
        out.append({"path": lab.name, "sha256": _sha(lab), **extra})
    return out


def _swiss_ids(floors: set[int]) -> dict[int, dict]:
    import pandas as pd

    from kelpod_train.datasets.swiss import CSV

    df = pd.read_csv(DATA / CSV, usecols=["site_id", "building_id", "floor_id"]).drop_duplicates()
    df = df[df.floor_id.isin(floors)]
    return {int(r.floor_id): {"floor_id": int(r.floor_id), "building_id": int(r.building_id),
                              "site_id": int(r.site_id)} for r in df.itertuples()}


def write(source: str) -> Path:
    imgs = sorted((DATA / "train" / source).glob("*_img.png"))
    ids = _swiss_ids({int(p.name.split("_")[1]) for p in imgs}) if source == "swiss" else {}
    files: list[dict] = []
    with ProcessPoolExecutor(12) as ex:
        for e in ex.map(_entries, [(source, p, ids) for p in imgs], chunksize=32):
            files += e
    man = {"source": f"{source}-train", "data_dir": str(DATA / "train" / source),
           "n_pages": len(imgs), "files": files}
    if source == "swiss":
        man["holdout_rule"] = "crc32(str(site_id)) % 10: 0 = test, 1 = val, else train"
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{source}.json"
    path.write_text(json.dumps(man))
    return path


if __name__ == "__main__":
    import sys

    for s in sys.argv[1:] or [p.name for p in sorted((DATA / "train").iterdir())]:
        print(write(s), flush=True)
