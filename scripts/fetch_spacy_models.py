#!/usr/bin/env python3
"""Download the small spaCy NER models used by the sanitize stage (build host only).

The appliance never downloads anything (CLAUDE.md rule 1). On the internet-connected build
host run::

    ZLLF_ALLOW_ONLINE=1 python scripts/fetch_spacy_models.py            # en + de
    ZLLF_ALLOW_ONLINE=1 python scripts/fetch_spacy_models.py --models en_core_web_sm

Each model wheel is fetched from the official ``explosion/spacy-models`` GitHub release,
its sha256 is printed and recorded, and only the model *data* directory (the folder holding
``config.cfg`` + ``meta.json``) is unpacked to::

    $ZLLF_MODELS_DIR/spacy/<name>/        (default ZLLF_MODELS_DIR = ~/.cache/zllf/models)

``zllf.stages.sanitize.nlp`` loads models from there with ``spacy.load(<path>)``; nothing is
pip-installed. Copy that directory to the appliance together with the other models.

Stdlib only, so it runs without the project environment.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

# spaCy 3.8.x is pinned by uv.lock; model minor version must match the library minor version.
DEFAULT_MODELS: dict[str, str] = {
    "en_core_web_sm": "3.8.0",
    "de_core_news_sm": "3.8.0",
}
URL = (
    "https://github.com/explosion/spacy-models/releases/download/"
    "{name}-{version}/{name}-{version}-py3-none-any.whl"
)


def models_dir() -> Path:
    """Root directory for locally stored models (same rule as ``zllf.stages.sanitize.nlp``)."""
    root = os.environ.get("ZLLF_MODELS_DIR")
    return Path(root).expanduser() if root else Path.home() / ".cache" / "zllf" / "models"


def _download(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "zllf-fetch-spacy-models"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        buf = io.BytesIO()
        done = 0
        while chunk := resp.read(256 * 1024):
            buf.write(chunk)
            done += len(chunk)
            if total:
                print(f"  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="\r", flush=True)
        print()
        return buf.getvalue()


def _extract_model_data(wheel: bytes, name: str, version: str, dest: Path) -> None:
    """Unpack ``<name>/<name>-<version>/**`` from the wheel into ``dest`` (no path traversal)."""
    prefix = PurePosixPath(name) / f"{name}-{version}"
    with zipfile.ZipFile(io.BytesIO(wheel)) as zf:
        members = [m for m in zf.namelist() if PurePosixPath(m).is_relative_to(prefix)]
        if not any(PurePosixPath(m).name == "config.cfg" for m in members):
            raise RuntimeError(f"{name}: config.cfg not found in wheel")
        tmp = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=dest.parent))
        try:
            for member in members:
                rel = PurePosixPath(member).relative_to(prefix)
                if member.endswith("/") or ".." in rel.parts or rel.is_absolute():
                    continue
                target = tmp.joinpath(*rel.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, target.open("wb") as out:
                    shutil.copyfileobj(src, out)
            if dest.exists():
                shutil.rmtree(dest)
            os.replace(tmp, dest)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise


def fetch(name: str, version: str, root: Path, timeout: float) -> dict[str, str]:
    dest = root / "spacy" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = URL.format(name=name, version=version)
    print(f"[fetch] {name}-{version} <- {url}")
    wheel = _download(url, timeout)
    digest = hashlib.sha256(wheel).hexdigest()
    print(f"  sha256 {digest}  ({len(wheel) / 1e6:.1f} MB)")
    _extract_model_data(wheel, name, version, dest)
    record = {"name": name, "version": version, "url": url, "wheel_sha256": digest}
    (dest / "zllf_source.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"  -> {dest}")
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS),
                    help="model names (default: en_core_web_sm de_core_news_sm)")
    ap.add_argument("--version", default=None, help="override model version for all models")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args(argv)

    if os.environ.get("ZLLF_ALLOW_ONLINE") != "1":
        print("refusing to touch the network: set ZLLF_ALLOW_ONLINE=1 (build host only)",
              file=sys.stderr)
        return 2
    root = models_dir()
    failures = 0
    for name in args.models:
        version = args.version or DEFAULT_MODELS.get(name, "3.8.0")
        if (root / "spacy" / name / "config.cfg").exists() and not args.force:
            print(f"[skip] {name} already present in {root / 'spacy' / name}")
            continue
        try:
            fetch(name, version, root, args.timeout)
        except Exception as exc:  # report and continue with the next model
            failures += 1
            print(f"[error] {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
