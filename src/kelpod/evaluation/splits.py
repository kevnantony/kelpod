"""Frozen evaluation splits, SHA-256 manifests, leak checks (``docs/EVAL_PROTOCOL.md`` §2, §6).

Manifests live in git (``eval/manifests/<split>.json``); data live outside git, by default in
``~/kelpod-data/eval/<split>/`` (``KELPOD_EVAL_DATA`` overrides the root). See ``eval/README.md``
for the manifest format and the status of every split.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import zlib
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

MANIFEST_VERSION = 1
FRESH_SEED_BASE = 1 << 40
"""Fresh-split seeds are ``FRESH_SEED_BASE + k``; seeds >= 2**40 are reserved for fresh splits."""
DEGRADE_RANGE = (0.0, 1.2)

# Synthetic splits: seed ranges chosen apart from training (>= 1,000,000), the finetune agent's
# own synthetic test seeds (100000-100049) and the dev benchmark seeds (0-9999).
SYNTH_SPLITS: dict[str, dict] = {
    "val-synth": {"first_seed": 200_000, "n": 200, "style": "train", "family": "val"},
    "test-synth": {"first_seed": 300_000, "n": 300, "style": "heldout", "family": "test-synth"},
}
FRESH = {"n": 200, "style": "heldout", "family": "fresh"}
# Splits that need data or renderers that are not frozen yet (see eval/README.md).
PENDING_SPLITS: dict[str, dict] = {
    "real-dev": {"family": "val", "kind": "proxy"},
    "test-real": {"family": "test-real", "kind": "sealed"},
}
# Dataset splits (frozen by ``build_dataset_split``): sizes follow from the runtime budget
# (docs/EVAL_PROTOCOL.md; ~25 s per plan and run, 8 workers, 4 runs per plan).
DATASET_SPLITS: dict[str, dict] = {
    "val-swiss": {"family": "val", "dataset": "swiss", "n": 100, "ids": "val"},
    "val-resplan": {"family": "val", "dataset": "resplan", "n": 100, "ids": "val"},
    "val-cubicasa": {"family": "val", "dataset": "cubicasa", "n": 100, "ids": "val"},
    "test-ood-swiss": {"family": "test-ood", "dataset": "swiss", "n": 150, "ids": "test"},
    "test-ood-resplan": {"family": "test-ood", "dataset": "resplan", "n": 150, "ids": "test"},
    "test-ood-cubicasa": {"family": "test-ood", "dataset": "cubicasa", "n": 200, "ids": "test"},
}
LOCKED_FAMILIES = ("test-synth", "test-ood", "test-real", "fresh")
"""Families whose items are aggregates-only (every run is logged)."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def manifest_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / "eval" / "manifests"


def data_root() -> Path:
    return Path(os.environ.get("KELPOD_EVAL_DATA", Path.home() / "kelpod-data" / "eval"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------
# Perceptual hash (near-duplicate check, §2 leak hygiene)
# --------------------------------------------------------------------------------------------


def phash(image: np.ndarray | str | Path, size: int = 32, keep: int = 8) -> str:
    """64-bit DCT perceptual hash (pHash) of a grayscale image or image file, as 16 hex chars.

    Invariant to scale, mild blur, noise and JPEG; two renderings of the same plan land within a
    few bits, different plans ~32 bits apart.
    """
    import cv2

    if not isinstance(image, np.ndarray):
        from PIL import Image

        image = np.asarray(Image.open(image).convert("L"))
    img = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(img.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    d = cv2.dct(small)[:keep, :keep].flatten()
    bits = d[1:] > np.median(d[1:])
    bits = np.concatenate([[d[0] > 0], bits])
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def hamming(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def near_duplicates(
    candidates: dict[str, str], references: dict[str, str], max_distance: int = 8
) -> list[tuple[str, str, int]]:
    """Pairs (candidate id, reference id, distance) with pHash distance <= ``max_distance``."""
    if not candidates or not references:
        return []
    ref_ids = list(references)
    ref = np.array([int(references[k], 16) for k in ref_ids], dtype=np.uint64)
    out = []
    for cid, h in candidates.items():
        x = np.bitwise_xor(ref, np.uint64(int(h, 16)))
        dist = np.array([int(v).bit_count() for v in x])
        for j in np.nonzero(dist <= max_distance)[0]:
            out.append((cid, ref_ids[int(j)], int(dist[j])))
    return out


# --------------------------------------------------------------------------------------------
# Synthetic splits
# --------------------------------------------------------------------------------------------


def split_degrades(name: str, n: int) -> list[float]:
    """Per-plan degradation strengths, uniform in ``DEGRADE_RANGE``, fixed by the split name."""
    seed = int.from_bytes(hashlib.sha256(f"kelpod-eval/{name}".encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    return [round(float(v), 3) for v in rng.uniform(*DEGRADE_RANGE, n)]


def fresh_params(candidate_commit: str, main_commit: str, n: int = FRESH["n"]) -> dict:
    """Seeds and degrades of the fresh split: seed ``sha256(candidate commit + main commit)``."""
    digest = hashlib.sha256((candidate_commit + main_commit).encode()).hexdigest()
    rng = np.random.default_rng(int(digest[:16], 16))
    offset = int(digest[16:26], 16) * 1000  # < 2**40 * 1000; keeps fresh sets apart
    seeds = [FRESH_SEED_BASE + offset + k for k in range(n)]
    degrades = [round(float(v), 3) for v in rng.uniform(*DEGRADE_RANGE, n)]
    return {"digest": digest, "seeds": seeds, "degrades": degrades}


def _item_files(d: Path) -> dict[str, str]:
    return {p.name: sha256_file(p) for p in sorted(d.iterdir()) if p.is_file()}


def _make_synth_item(args: tuple[int, float, str, str]) -> dict:
    seed, degrade, style, out = args
    from kelpod.evaluation.synth import generate, save

    plan = generate(seed, degrade, style=style)  # type: ignore[arg-type]
    d = save(plan, Path(out) / f"s{seed}")
    return {
        "id": f"s{seed}",
        "seed": seed,
        "degrade": degrade,
        "style": plan.style.to_dict() if plan.style else None,
        "n_rooms": len(plan.graph.rooms),
        "phash": phash(plan.image),
        "files": _item_files(d),
    }


def generate_synth(
    seeds: list[int], degrades: list[float], style: str, out: Path, workers: int = 8
) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(s, g, style, str(out)) for s, g in zip(seeds, degrades, strict=True)]
    if workers <= 1:
        return [_make_synth_item(j) for j in jobs]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(_make_synth_item, jobs, chunksize=4))


def git_commit(path: Path) -> str:
    import subprocess

    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def git_dirty(path: Path, sub: str) -> bool:
    import subprocess

    out = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--", sub],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return bool(out.strip())


def build_synth_split(name: str, *, workers: int = 8, data: Path | None = None) -> dict:
    """Generate a frozen synthetic split and return its manifest (not yet written)."""
    from kelpod.evaluation.synth import GENERATOR_VERSION

    cfg = SYNTH_SPLITS[name]
    root = repo_root()
    if git_dirty(root, "src/kelpod/evaluation"):
        raise RuntimeError("commit the generator first: the manifest records its commit")
    seeds = list(range(cfg["first_seed"], cfg["first_seed"] + cfg["n"]))
    degrades = split_degrades(name, cfg["n"])
    out = (data or data_root()) / name
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty: a frozen split is never regenerated")
    items = generate_synth(seeds, degrades, cfg["style"], out, workers)
    return {
        "manifest_version": MANIFEST_VERSION,
        "split": name,
        "status": "frozen",
        "kind": "synth",
        "family": cfg["family"],
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
        "data_dir": f"~/kelpod-data/eval/{name}",
        "input_file": "plan.pdf",
        "gt_file": "ground_truth.json",
        "generator": {
            "module": "kelpod.evaluation.synth",
            "commit": git_commit(root),
            "version": GENERATOR_VERSION,
            "style": cfg["style"],
            "dpi": 300.0,
        },
        "params": {
            "seeds": f"{seeds[0]}-{seeds[-1]}",
            "degrade": f"uniform {DEGRADE_RANGE}, per item, from sha256('kelpod-eval/{name}')",
        },
        "n_items": len(items),
        "items": items,
    }


# --------------------------------------------------------------------------------------------
# Dataset splits: held-out ID lists (data pending)
# --------------------------------------------------------------------------------------------


def swiss_split_of(site_id: int) -> str:
    """Site-level split, the same rule the training converter uses
    (``crc32(str(site_id)) % 10``: 0 test, 1 val, else train)."""
    h = zlib.crc32(str(int(site_id)).encode()) % 10
    return "test" if h == 0 else "val" if h == 1 else "train"


def swiss_heldout_ids(csv_path: Path) -> dict[str, dict[str, list[int]]]:
    """Site and building IDs of the Swiss Dwellings test and val sites (streams the CSV)."""
    sites: dict[str, set[int]] = {"test": set(), "val": set()}
    buildings: dict[str, set[int]] = {"test": set(), "val": set()}
    floors: dict[str, set[int]] = {"test": set(), "val": set()}
    seen: set[tuple[int, int, int]] = set()
    csv.field_size_limit(1 << 30)
    with open(csv_path, newline="") as f:
        r = csv.reader(f)
        head = next(r)
        i_site, i_bld, i_floor = (head.index(c) for c in ("site_id", "building_id", "floor_id"))
        for row in r:
            key = (int(row[i_site]), int(row[i_bld]), int(row[i_floor]))
            if key in seen:
                continue
            seen.add(key)
            split = swiss_split_of(key[0])
            if split in sites:
                sites[split].add(key[0])
                buildings[split].add(key[1])
                floors[split].add(key[2])
    return {
        s: {
            "site_ids": sorted(sites[s]),
            "building_ids": sorted(buildings[s]),
            "floor_ids": sorted(floors[s]),
        }
        for s in sites
    }


def _dataset_job(args: tuple) -> dict | None:
    """Render one candidate of a dataset split (runs in a pool worker)."""
    from kelpod.evaluation import datasets as D

    dataset, cid, seed, degrade, out, payload = args
    d = Path(out) / cid
    if dataset == "cubicasa":
        r = D.cubicasa_item(payload)
        if r is None:
            return None
        img, graph, meta = r
        D.write_item(d, img, graph, meta, D.CUBICASA_DPI)
        info = {"n_rooms": len(graph.rooms), "stats": meta["stats"]}
        png = img
    else:
        plan = (
            D.swiss_floor_plan(payload[0], payload[1])
            if dataset == "swiss"
            else D.resplan_plan(payload)
        )
        if plan is None:
            return None
        info = D.vector_item(plan, seed=seed, degrade=degrade, directory=d)
        if info is None:
            return None
        png = str(d / "plan.png")
    return {"id": cid, "seed": seed, "degrade": degrade if dataset != "cubicasa" else None,
            "n_rooms": info["n_rooms"], "stats": info["stats"], "phash": phash(png),
            "files": _item_files(d)}  # fmt: skip


def build_dataset_split(name: str, *, workers: int = 8, data: Path | None = None) -> dict:
    """Render a frozen dataset split from the held-out ID lists of its pending manifest.
    Candidates are taken in the order of ``sha256('kelpod-eval/<split>/<id>')``; the first ``n``
    that render (extent 3-60 m, enough walls and rooms, little unsupported wall area) are kept."""
    from kelpod.evaluation import datasets as D
    from kelpod.evaluation.vectorplans import VECTOR_VERSION

    cfg = DATASET_SPLITS[name]
    root = repo_root()
    if git_dirty(root, "src/kelpod/evaluation"):
        raise RuntimeError("commit the renderer first: the manifest records its commit")
    old = json.loads((manifest_dir(root) / f"{name}.json").read_text())
    if old["status"] == "frozen":
        raise FileExistsError(f"{name} is frozen")
    out = (data or data_root()) / name
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty: a frozen split is never regenerated")
    ds, n = cfg["dataset"], cfg["n"]
    held = old["heldout_ids"]
    key = lambda i: hashlib.sha256(f"kelpod-eval/{name}/{i}".encode()).hexdigest()  # noqa: E731
    pool = int(n * 1.7) + 10
    degrades = split_degrades(name, pool)
    jobs = []
    if ds == "swiss":
        floors = sorted(held["swiss_floor_id"], key=key)[:pool]
        rows = D.swiss_rows(set(floors))
        payloads = [(f"swiss_{f}", (f, rows[f])) for f in floors]
    elif ds == "resplan":
        plans = D.resplan_load()
        ids = sorted(held["resplan_id"], key=key)
        payloads = [(f"resplan_{i}", plans[int(i)]) for i in ids if int(i) in plans][:pool]
    else:
        folders = sorted(held["cubicasa_id"], key=key)[:pool]
        payloads = [("cubicasa_" + f.strip("/").replace("/", "_"), f) for f in folders]
    for k, (cid, payload) in enumerate(payloads):
        seed = int(key(cid)[:8], 16)
        jobs.append((ds, cid, seed, degrades[k], str(out), payload))
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_dataset_job, jobs, chunksize=2))
    ok = [r for r in res if r]
    # a candidate that is a near duplicate (pHash <= 4) of a training image is not used: some
    # CubiCasa5k plans occur in both its train and test lists
    train_ph = {}
    for f in sorted((root / "training" / "manifests").glob("*.json")):
        for e in json.loads(f.read_text()).get("files", []):
            if e.get("phash"):
                train_ph[f"{f.stem}/{e['path']}"] = e["phash"]
    dup = {a for a, _, _ in near_duplicates({r["id"]: r["phash"] for r in ok}, train_ph, 4)}
    res = [None if (r and r["id"] in dup) else r for r in res]
    items = [r for r in res if r][:n]
    keep = {i["id"] for i in items}
    import shutil

    for r in ok:  # candidates beyond the first n successes are not part of the split
        if r["id"] not in keep:
            shutil.rmtree(out / r["id"], ignore_errors=True)
    if len(items) < n:
        raise RuntimeError(f"{name}: only {len(items)} of {n} candidates rendered")
    m = {
        "manifest_version": MANIFEST_VERSION, "split": name, "status": "frozen", "kind": "dataset",
        "family": cfg["family"], "created_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
        "data_dir": f"~/kelpod-data/eval/{name}", "input_file": "plan.pdf",
        "gt_file": "ground_truth.json",
        "scoring": "scale-free" if ds == "cubicasa" else "metric",
        "generator": {"module": "kelpod.evaluation.vectorplans", "commit": git_commit(root),
                      "version": VECTOR_VERSION, "style": "train",
                      "dpi": D.CUBICASA_DPI if ds == "cubicasa" else D.DPI},
        "source": old.get("source"),
        "params": {"candidates": f"sorted by sha256('kelpod-eval/{name}/<id>'), first {n} that "
                   f"render and are no near duplicate of a training image ({len(res)} tried, "
                   f"{len(dup)} dropped as near duplicates)",
                   "degrade": f"uniform {DEGRADE_RANGE}, per candidate index, from "
                   f"sha256('kelpod-eval/{name}') (not for CubiCasa: real scans, no degradation)"},
        "heldout_ids": held, "n_items": len(items), "items": items,
    }  # fmt: skip
    return m


def pending_manifest(name: str, *, note: str, extra: dict | None = None) -> dict:
    cfg = PENDING_SPLITS.get(name) or {**DATASET_SPLITS[name], "kind": "dataset"}
    return {
        "manifest_version": MANIFEST_VERSION,
        "split": name,
        "status": "pending",
        "kind": cfg["kind"],
        "family": cfg["family"],
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
        "note": note,
        **(extra or {}),
        "n_items": 0,
        "items": [],
    }


def hash_only_manifest(name: str, directory: Path, *, note: str) -> dict:
    """Manifest of a sealed / proxy directory: file hashes only, files are never decoded."""
    cfg = PENDING_SPLITS[name]
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    items = [{"id": str(p.relative_to(directory)), "sha256": sha256_file(p)} for p in files]
    return {
        "manifest_version": MANIFEST_VERSION,
        "split": name,
        "status": "frozen",
        "kind": cfg["kind"],
        "family": cfg["family"],
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
        "data_dir": str(directory).replace(str(Path.home()), "~"),
        "note": note,
        "n_items": len(items),
        "items": items,
    }


# --------------------------------------------------------------------------------------------
# Loading, verification, leak checks
# --------------------------------------------------------------------------------------------


def load_manifests(root: Path | None = None) -> dict[str, dict]:
    d = manifest_dir(root)
    return (
        {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json"))} if d.is_dir() else {}
    )


def split_data_dir(manifest: dict) -> Path:
    if os.environ.get("KELPOD_EVAL_DATA") and manifest.get("kind") in ("synth", "dataset"):
        return data_root() / manifest["split"]
    return Path(os.path.expanduser(manifest["data_dir"]))


def verify_manifest(manifest: dict) -> list[str]:
    """Problems with a frozen split's files (empty list = all hashes match)."""
    if manifest.get("status") != "frozen":
        return []
    d = split_data_dir(manifest)
    if not d.is_dir():
        return [f"{manifest['split']}: data directory {d} is missing"]
    problems = []
    for it in manifest["items"]:
        files = it.get("files") or {"": it["sha256"]}
        for fname, sha in files.items():
            p = d / it["id"] / fname if fname else d / it["id"]
            if not p.is_file():
                problems.append(f"{manifest['split']}: missing {p}")
            elif sha256_file(p) != sha:
                problems.append(f"{manifest['split']}: hash mismatch {p}")
    return problems


def locked_file_hashes(manifests: dict[str, dict]) -> dict[str, str]:
    """sha256 -> 'split/item' for every file of a locked (test) split."""
    out = {}
    for name, m in manifests.items():
        if m.get("family") not in LOCKED_FAMILIES:
            continue
        for it in m.get("items", []):
            for fname, sha in (it.get("files") or {"": it.get("sha256")}).items():
                if sha:
                    out[sha] = f"{name}/{it['id']}/{fname}".rstrip("/")
    return out


def heldout_dataset_ids(manifests: dict[str, dict]) -> dict[str, set[str]]:
    """Held-out IDs that training must exclude, by key (``swiss_site_id`` etc.), as strings."""
    out: dict[str, set[str]] = {}
    for m in manifests.values():
        if m.get("family") not in LOCKED_FAMILIES:
            continue
        for key, ids in (m.get("heldout_ids") or {}).items():
            out.setdefault(key, set()).update(str(i).strip("/") for i in ids)
    return out


ID_KEYS = {
    "swiss_site_id": ("swiss_site_id", "site_id"),
    "swiss_building_id": ("swiss_building_id", "building_id"),
    "cubicasa_id": ("cubicasa_id", "cubicasa_folder"),
    "resplan_id": ("resplan_id",),
}


def check_training_overlap(
    training_manifests: Iterable[Path], manifests: dict[str, dict]
) -> list[str]:
    """Leak check: a training manifest must not contain a test file hash, a held-out dataset
    ID, a held-out-style synthetic plan or a test-split image (by pHash, if it lists pHashes)."""
    hashes = locked_file_hashes(manifests)
    held = heldout_dataset_ids(manifests)
    test_ph = {
        f"{n}/{it['id']}": it["phash"]
        for n, m in manifests.items()
        if m.get("family") in LOCKED_FAMILIES
        for it in m.get("items", [])
        if it.get("phash")
    }
    problems = []
    for path in training_manifests:
        tm = json.loads(Path(path).read_text())
        entries = tm.get("files") or tm.get("items") or []
        train_ph = {}
        for e in entries:
            label = e.get("path") or e.get("id") or "?"
            for sha in [e.get("sha256"), *(e.get("files") or {}).values()]:
                if sha and sha in hashes:
                    problems.append(f"{path}: {label} has the hash of test file {hashes[sha]}")
            for key, fields in ID_KEYS.items():
                for fld in fields:
                    if fld in e and key in held and str(e[fld]).strip("/") in held[key]:
                        problems.append(f"{path}: {label} has held-out {key} {e[fld]}")
            style = e.get("style")
            if style == "heldout" or (isinstance(style, dict) and style.get("split") == "heldout"):
                problems.append(f"{path}: {label} is drawn in the held-out test style")
            if e.get("phash"):
                train_ph[label] = e["phash"]
        for a, b, dist in near_duplicates(train_ph, test_ph, max_distance=4):
            problems.append(f"{path}: {a} is a near duplicate of {b} (pHash distance {dist})")
    return problems


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------

DATA = Path.home() / "kelpod-data"


def _pending_all() -> dict[str, dict]:
    """Manifests of the splits that cannot be frozen yet, with their held-out ID lists."""
    out = {}
    renderer = (
        "needs a committed DIN-style renderer for vector plans that also writes a PlanGraph "
        "ground truth (the finetune agent's training/kelpod_train/render.py is uncommitted "
        "and writes class maps only)"
    )
    sd = DATA / "swiss-dwellings" / "swiss-dwellings-v3.0.0" / "geometries.csv"
    if sd.is_file():
        ids = swiss_heldout_ids(sd)
        rule = "crc32(str(site_id)) % 10: 0 = test, 1 = val, else train (as the training converter)"
        for split, key in (("test", "test-ood-swiss"), ("val", "val-swiss")):
            out[key] = pending_manifest(
                key,
                note=f"{renderer}. Held out by site (all buildings of a site). Rule: {rule}.",
                extra={
                    "source": {
                        "dataset": "Swiss Dwellings v3.0.0",
                        "file": str(sd).replace(str(Path.home()), "~"),
                        "sha256": sha256_file(sd),
                    },
                    "heldout_ids": {
                        "swiss_site_id": ids[split]["site_ids"],
                        "swiss_building_id": ids[split]["building_ids"],
                        "swiss_floor_id": ids[split]["floor_ids"],
                    },
                },
            )
    cc = DATA / "cubicasa5k" / "cubicasa5k"
    for split, key in (("test", "test-ood-cubicasa"), ("val", "val-cubicasa")):
        f = cc / f"{split}.txt"
        if f.is_file():
            folders = [ln.strip().strip("/") for ln in f.read_text().splitlines() if ln.strip()]
            out[key] = pending_manifest(
                key,
                note=f"official CubiCasa5k {split} split ({len(folders)} plans); needs a loader "
                "from the SVG annotations to a PlanGraph ground truth",
                extra={"source": {"dataset": "CubiCasa5k", "file": f"{split}.txt",
                                  "sha256": sha256_file(f)},
                       "heldout_ids": {"cubicasa_id": folders}},
            )  # fmt: skip
    rp = DATA / "src" / "ResPlan" / "split.json"
    if rp.is_file():
        sp = json.loads(rp.read_text())
        for split, key in (("test", "test-ood-resplan"), ("val", "val-resplan")):
            out[key] = pending_manifest(
                key,
                note=f"{renderer}. ResPlan split.json '{split}' ({len(sp[split])} plans).",
                extra={"source": {"dataset": "ResPlan", "file": "split.json",
                                  "sha256": sha256_file(rp)},
                       "heldout_ids": {"resplan_id": sp[split]}},
            )  # fmt: skip
    rd = DATA / "real" / "dev"
    if rd.is_dir() and any(rd.iterdir()):
        out["real-dev"] = hash_only_manifest(
            "real-dev", rd, note="real dev pages, proxy panel only (no annotations)"
        )
    else:
        out["real-dev"] = pending_manifest(
            "real-dev", note=f"{str(rd).replace(str(Path.home()), '~')} does not exist yet"
        )
    st = DATA / "sealed" / "real-test"
    if st.is_dir() and any(st.iterdir()):
        out["test-real"] = hash_only_manifest(
            "test-real", st, note="sealed; hashes only, items are never opened one by one"
        )
        out["test-real"]["status"] = "sealed-no-annotations"
    else:
        out["test-real"] = pending_manifest(
            "test-real", note="~/kelpod-data/sealed/real-test does not exist yet"
        )
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Build frozen eval splits and their manifests.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="generate a frozen synthetic split")
    b.add_argument("split", choices=sorted(SYNTH_SPLITS))
    b.add_argument("--workers", type=int, default=8)
    bd = sub.add_parser("build-dataset", help="render a frozen dataset split")
    bd.add_argument("split", choices=sorted(DATASET_SPLITS))
    bd.add_argument("--workers", type=int, default=8)
    pd = sub.add_parser("pending", help="(re)write manifests of splits that are not frozen")
    pd.add_argument("--only", nargs="*")
    args = ap.parse_args(argv)
    mdir = manifest_dir()
    mdir.mkdir(parents=True, exist_ok=True)
    if args.cmd == "build":
        path = mdir / f"{args.split}.json"
        if path.exists():
            raise SystemExit(f"{path} exists: frozen splits are never rebuilt")
        m = build_synth_split(args.split, workers=args.workers)
        path.write_text(json.dumps(m, indent=1) + "\n")
        print(f"{path}: {m['n_items']} items in {m['data_dir']}")
    elif args.cmd == "build-dataset":
        path = mdir / f"{args.split}.json"
        m = build_dataset_split(args.split, workers=args.workers)
        path.write_text(json.dumps(m, indent=1) + "\n")
        print(f"{path}: {m['n_items']} items in {m['data_dir']}")
    else:
        for name, m in _pending_all().items():
            if args.only and name not in args.only:
                continue
            path = mdir / f"{name}.json"
            if path.exists() and json.loads(path.read_text()).get("status") == "frozen":
                print(f"{path}: frozen, left alone")
                continue
            path.write_text(json.dumps(m, indent=1) + "\n")
            print(f"{path}: {m['status']}, {m['n_items']} items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
