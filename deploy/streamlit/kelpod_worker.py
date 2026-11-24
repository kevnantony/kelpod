"""Work that runs in a short-lived child process, so its memory goes back to the system after
every conversion (Python keeps freed memory otherwise, and the free tier has about 1-2.7 GB).

Called as ``python kelpod_worker.py <function> <args.pkl> <result.pkl>`` by the Streamlit page.
"""

from __future__ import annotations

import pickle
import sys
import tempfile
from pathlib import Path

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
        res = convert(src, ifc, schema=schema, dpi=SANDBOX_DPI, plan_scale=plan_scale,
                      storey_height_m=height)
        rep = res.graph.report
        counts = res.export.entity_counts if res.export else {}
        html = write_viewer(src, ifc, Path(d) / "viewer.html", title="", note="", stats={})
        _, png = cv2.imencode(".png", load(src, dpi=110)[0])
        return {
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
