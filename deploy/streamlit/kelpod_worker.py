"""Work that runs in a short-lived child process, so its memory goes back to the system after
every conversion (Python keeps freed memory otherwise, and the free tier has about 1-2.7 GB).

Called as ``python kelpod_worker.py <function> <args.pkl> <result.pkl>`` by the Streamlit page.
"""

from __future__ import annotations

import os
import pickle
import sys
import tempfile
from pathlib import Path

# Import kelpod from the repo checkout (see streamlit_app.py), also when run without PYTHONPATH.
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# The viewer page shows plan and model side by side; Streamlit shows the plan itself, so the
# embedded page keeps only its 3D pane, filling the frame.
ONLY_3D = ("<style>header,main>section:first-child,.pane h2,.stats{display:none!important}"
           "main{grid-template-columns:1fr!important;height:100vh!important;padding:0!important}"
           ".pane{height:100vh!important;border-radius:0!important}</style>")


def sample_pdf(seed: int) -> bytes:
    """A synthetic German-style plan from Kelpod's generator, as an image-only PDF."""
    from kelpod.evaluation import synth

    with tempfile.TemporaryDirectory() as d:
        synth.save(synth.generate(seed, degrade=0.5), d)
        return (Path(d) / "plan.pdf").read_bytes()


# 160 dpi instead of the pipeline default 200: peak memory 0.93 GB vs 1.25 GB (8-room plan); on
# 8 synthetic plans 8/8 closed (7/8 at 200 dpi), room IoU about 0.98 vs 0.99. Measured 2026-11-24.
SANDBOX_DPI = 160.0


# The learned segmenter's weights are not in git. They are published (CC BY-NC-SA 4.0) on Hugging
# Face; the sandbox downloads the default checkpoint once and checks its sha256 against
# kelpod-run.toml, so the hosted app runs the same model as main.
WEIGHTS_REPO = "https://huggingface.co/kevinantony/kelpod-seg/resolve/main"
WEIGHTS_CACHE = Path(os.environ.get("KELPOD_CACHE", Path.home() / ".cache" / "kelpod"))


def _fetch_published_weights() -> None:
    """Download the run configuration's first checkpoint (seed 1) when it is not on this machine,
    verify it, and point KELPOD_SEG_CHECKPOINT at it. Any failure leaves the variable unset."""
    import hashlib
    import urllib.request

    from kelpod.runconfig import load_config

    entries = load_config().get("checkpoints") or []
    if not entries or os.environ.get("KELPOD_SEG_CHECKPOINT"):
        return
    entry = entries[0]
    want = str(entry.get("sha256", ""))
    name = Path(str(entry["path"])).parent.name + "/model.onnx"  # e.g. v1-s1/model.onnx
    dest = WEIGHTS_CACHE / (want[:16] or "unverified") / "model.onnx"

    def sha(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()

    if not dest.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        req = urllib.request.Request(f"{WEIGHTS_REPO}/{name}",
                                     headers={"User-Agent": "kelpod-sandbox"})
        with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as f:
            for block in iter(lambda: r.read(1 << 20), b""):
                f.write(block)
        if want and sha(tmp) != want:
            tmp.unlink(missing_ok=True)
            return
        tmp.replace(dest)
    os.environ["KELPOD_SEG_CHECKPOINT"] = str(dest)


def _segment_backend() -> str:
    """The learned segmenter (main's default) needs its checkpoint. Use the local one from the run
    configuration, else the published one; fall back to the classical segmenter if neither works."""
    from kelpod.runconfig import CheckpointError, default_segment_backend, resolve_checkpoint

    backend = default_segment_backend()
    if os.environ.get("KELPOD_SANDBOX_CLASSICAL"):  # set by the page to retry after a failed run
        return "classical"
    if backend == "learned":
        try:
            resolve_checkpoint()
        except (CheckpointError, OSError):
            try:
                _fetch_published_weights()
                resolve_checkpoint()
            except Exception:  # noqa: BLE001 - any download or verify problem: use classical
                return "classical"
    return backend


def convert_bytes(data: bytes, suffix: str, schema: str, plan_scale: float, height: float) -> dict:
    """Run the Kelpod pipeline on an uploaded file and return everything the page shows."""
    import cv2

    from kelpod.clean import load
    from kelpod.pipeline import convert
    from kelpod.viz import write_viewer

    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / f"plan{suffix}"
        src.write_bytes(data)
        ifc = Path(d) / "model.ifc"
        backend = _segment_backend()
        res = convert(src, ifc, schema=schema, dpi=SANDBOX_DPI, plan_scale=plan_scale,
                      storey_height_m=height, segment_backend=backend)
        rep = res.graph.report
        counts = res.export.entity_counts if res.export else {}
        html = write_viewer(src, ifc, Path(d) / "viewer.html", title="", note="", stats={})
        _, png = cv2.imencode(".png", load(src, dpi=110)[0])
        return {
            "segmenter": backend,
            "closed": bool(rep and rep.is_closed),
            "node_closure": rep.node_closure if rep else None,
            "dangling": len(rep.dangling_nodes) if rep else None,
            "issues": list(rep.issues) if rep else [],
            "rooms": [r.name or f"Room {r.id}" for r in res.graph.rooms],
            "areas": [round(r.area_m2, 2) for r in res.graph.rooms],
            "scale_source": res.graph.scale_source,
            "reopened": bool(res.export and res.export.reopened_ok),
            "timings": {t.step: round(t.seconds, 1) for t in res.timings},
            "total": round(res.total_seconds, 1),
            "stats": {
                "walls": counts.get("IfcWall", 0) + counts.get("IfcWallStandardCase", 0),
                "doors": counts.get("IfcDoor", 0),
                "windows": counts.get("IfcWindow", 0),
                "rooms": counts.get("IfcSpace", 0),
            },
            "ifc": ifc.read_bytes(),
            "viewer": html.read_text(encoding="utf-8").replace("</head>", ONLY_3D + "</head>", 1),
            "plan_png": png.tobytes(),
        }


if __name__ == "__main__":
    fn, args_path, out_path = sys.argv[1:4]
    args = pickle.loads(Path(args_path).read_bytes())
    result = {"sample_pdf": sample_pdf, "convert_bytes": convert_bytes}[fn](*args)
    Path(out_path).write_bytes(pickle.dumps(result))
