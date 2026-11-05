#!/usr/bin/env python3
"""Download the models named in hardware profiles and write ``models/manifest.json``.

This is the ONLY code in the repository that may touch the network (DEVELOPMENT_PLAN §4.2,
P0.5.1). It runs on the internet-connected *build host*, never on the appliance.

Usage (build host)::

    ZLLF_ALLOW_ONLINE=1 uv run --with huggingface_hub python scripts/download_models.py \
        --hardware toy mac

    # offline re-hash of everything in the manifest (appliance / after import):
    python scripts/download_models.py --verify

Behaviour:

* Reads ``models.{student,teacher,nli,embedder}`` from each ``configs/hardware/<name>.yaml``;
  ids starting with ``fake-`` are skipped.
* Each model is pinned to a commit hash. If ``models/manifest.json`` already records a revision
  for the model it is reused (reproducible re-download); ``--update`` re-resolves ``main``.
* Files land in the normal Hugging Face cache (``HF_HUB_CACHE``), so ``zllf`` finds them with
  ``HF_HUB_OFFLINE=1``. The manifest stores per-file sha256, total size and the licence from the
  model card metadata.
* ``--verify`` needs no network and no ``huggingface_hub``: it re-hashes the cached snapshot of
  every manifest entry and exits non-zero on any mismatch or missing file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "models" / "manifest.json"
HARDWARE_DIR = REPO_ROOT / "configs" / "hardware"
ROLES = ("student", "teacher", "nli", "embedder")
MANIFEST_SCHEMA = 1

# Weight formats we never need (we load safetensors via transformers / sentence-transformers /
# mlx-lm). Keeps e.g. BAAI/bge-m3 from pulling ONNX + PyTorch duplicates.
IGNORE_PATTERNS = [
    "*.onnx",
    "onnx/*",
    "openvino/*",
    "coreml/*",
    "*.h5",
    "*.msgpack",
    "*.ot",
    "*.tflite",
    "tf_model*",
    "flax_model*",
    "rust_model*",
    "*.gguf",
]


# --------------------------------------------------------------------------- helpers


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Return the hex sha256 of ``path`` (follows symlinks into the HF blob store)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def hf_cache_dir() -> Path:
    """Resolve the Hugging Face hub cache like ``huggingface_hub.constants`` does."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "huggingface" / "hub"


def snapshot_path(repo_id: str, revision: str, cache: Path | None = None) -> Path:
    """Path of a cached snapshot: ``<cache>/models--org--name/snapshots/<sha>``."""
    cache = cache or hf_cache_dir()
    return cache / ("models--" + repo_id.replace("/", "--")) / "snapshots" / revision


def hash_snapshot(root: Path) -> list[dict[str, Any]]:
    """List every file under a snapshot dir with size and sha256, sorted by path."""
    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        files.append({"path": rel, "size": path.stat().st_size, "sha256": sha256_file(path)})
    return files


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml  # pyyaml is a project dependency

    with path.open() as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected a mapping")
    return data


def models_for_hardware(names: list[str]) -> dict[str, dict[str, set[str]]]:
    """Map model id -> {"roles": {...}, "profiles": {...}} for the given hardware profiles."""
    wanted: dict[str, dict[str, set[str]]] = {}
    for name in names:
        path = HARDWARE_DIR / f"{name}.yaml"
        if not path.is_file():
            raise SystemExit(f"unknown hardware profile {name!r}: {path} not found")
        models = load_yaml(path).get("models", {}) or {}
        for role in ROLES:
            model_id = models.get(role)
            if not model_id or str(model_id).startswith("fake-"):
                continue
            entry = wanted.setdefault(str(model_id), {"roles": set(), "profiles": set()})
            entry["roles"].add(role)
            entry["profiles"].add(name)
    return wanted


def read_manifest(path: Path) -> dict[str, Any]:
    if path.is_file():
        data: dict[str, Any] = json.loads(path.read_text())
        return data
    return {"schema_version": MANIFEST_SCHEMA, "models": []}


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["models"] = sorted(manifest["models"], key=lambda m: m["name"])
    manifest["generated_at"] = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    tmp.replace(path)


def licence_of(info: Any) -> str:
    """Licence from model-card metadata, falling back to the ``license:`` tag."""
    card = getattr(info, "card_data", None) or getattr(info, "cardData", None)
    lic = None
    if card is not None:
        lic = card.get("license") if hasattr(card, "get") else getattr(card, "license", None)
        if lic == "other":
            name = card.get("license_name") if hasattr(card, "get") else None
            lic = f"other:{name}" if name else "other"
    if not lic:
        for tag in getattr(info, "tags", None) or []:
            if tag.startswith("license:"):
                lic = tag.split(":", 1)[1]
                break
    if isinstance(lic, list):
        lic = ",".join(map(str, lic))
    return str(lic) if lic else "UNKNOWN"


# --------------------------------------------------------------------------- download


def download(args: argparse.Namespace) -> int:
    if os.environ.get("ZLLF_ALLOW_ONLINE") != "1":
        print(
            "refusing to touch the network: set ZLLF_ALLOW_ONLINE=1 (build host only)",
            file=sys.stderr,
        )
        return 2
    # Undo offline flags a parent shell / `import zllf` may have set.
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.pop(var, None)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        print("needs huggingface_hub: uv run --with huggingface_hub ...", file=sys.stderr)
        return 2

    api = HfApi()
    manifest = read_manifest(args.manifest)
    by_name = {m["name"]: m for m in manifest["models"]}
    wanted = models_for_hardware(args.hardware)
    if not wanted:
        print("no non-fake models in the selected profiles; nothing to do")
        return 0

    for repo_id, meta in sorted(wanted.items()):
        previous = by_name.get(repo_id)
        pinned = None if args.update or previous is None else previous["revision"]
        info = api.model_info(repo_id, revision=pinned or "main")
        revision = str(info.sha)
        siblings = [s.rfilename for s in (info.siblings or [])]
        ignore = list(IGNORE_PATTERNS)
        if any(f.endswith(".safetensors") for f in siblings):
            ignore += ["*.bin", "*.pt", "*.pth"]
        print(f"[download] {repo_id}@{revision[:12]} ({'pinned' if pinned else 'resolved'})")
        local = Path(snapshot_download(repo_id, revision=revision, ignore_patterns=ignore))
        files = hash_snapshot(local)
        entry = {
            "name": repo_id,
            "revision": revision,
            "license": licence_of(info),
            "roles": sorted(meta["roles"] | set(previous.get("roles", []) if previous else [])),
            "profiles": sorted(
                meta["profiles"] | set(previous.get("profiles", []) if previous else [])
            ),
            "ignore_patterns": ignore,
            "total_size": sum(f["size"] for f in files),
            "files": files,
        }
        by_name[repo_id] = entry
        mib = entry["total_size"] / 2**20
        print(f"           {len(files)} files, {mib:.1f} MiB, licence={entry['license']}")

    manifest["schema_version"] = MANIFEST_SCHEMA
    manifest["models"] = list(by_name.values())
    write_manifest(args.manifest, manifest)
    print(
        f"[manifest] wrote {args.manifest.relative_to(REPO_ROOT)} "
        f"({len(manifest['models'])} models)"
    )
    return 0


# --------------------------------------------------------------------------- verify


def verify(args: argparse.Namespace) -> int:
    """Offline: re-hash each cached snapshot against the manifest."""
    if not args.manifest.is_file():
        print(f"no manifest at {args.manifest}", file=sys.stderr)
        return 1
    manifest = read_manifest(args.manifest)
    wanted = set(models_for_hardware(args.hardware)) if args.hardware_given else None
    cache = hf_cache_dir()
    failures = 0
    for entry in manifest["models"]:
        if wanted is not None and entry["name"] not in wanted:
            continue
        root = snapshot_path(entry["name"], entry["revision"], cache)
        bad: list[str] = []
        if not root.is_dir():
            bad.append(f"snapshot missing: {root}")
        else:
            for f in entry["files"]:
                path = root / f["path"]
                if not path.is_file():
                    bad.append(f"missing {f['path']}")
                elif path.stat().st_size != f["size"]:
                    bad.append(f"size mismatch {f['path']}")
                elif sha256_file(path) != f["sha256"]:
                    bad.append(f"sha256 mismatch {f['path']}")
        status = "OK " if not bad else "BAD"
        print(f"[{status}] {entry['name']}@{entry['revision'][:12]} ({len(entry['files'])} files)")
        for line in bad:
            print(f"       {line}")
        failures += bool(bad)
    if wanted is not None:
        missing = wanted - {m["name"] for m in manifest["models"]}
        for name in sorted(missing):
            print(f"[BAD] {name}: not in manifest")
            failures += 1
    return 1 if failures else 0


# --------------------------------------------------------------------------- cli


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--hardware",
        nargs="+",
        default=None,
        metavar="NAME",
        help="hardware profiles under configs/hardware (default: toy)",
    )
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--verify", action="store_true", help="offline re-hash against the manifest")
    p.add_argument(
        "--update", action="store_true", help="re-resolve 'main' instead of the pinned revision"
    )
    args = p.parse_args(argv)
    args.hardware_given = args.hardware is not None
    if args.hardware is None:
        args.hardware = ["toy"]
    args.manifest = args.manifest.resolve()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return verify(args) if args.verify else download(args)


if __name__ == "__main__":
    raise SystemExit(main())
