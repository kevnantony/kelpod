# ruff: noqa: E501  (embedded HTML/JS template and CLI help text)
"""Show a floor-plan PDF turning into a 3D IFC model.

    uv run python scripts/demo.py --seed 4                   # generate a synthetic plan, then convert
    uv run python scripts/demo.py --pdf plan.pdf             # convert your own scan
    uv run python scripts/demo.py --seed 4 --ground-truth    # skip recognition: export the generator's truth

Writes ``out/demo/<name>/{plan.pdf, model.ifc, viewer.html}``; open ``viewer.html`` in a browser.
``--ground-truth`` only exercises the export and the viewer; it is labelled as such on the page.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from kelpod.viz import write_viewer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pdf", type=Path, help="input scan (PDF/PNG/JPG/TIFF)")
    ap.add_argument("--seed", type=int, help="generate a synthetic plan with this seed")
    ap.add_argument("--degrade", type=float, default=0.5)
    ap.add_argument("--ground-truth", action="store_true",
                    help="export the synthetic ground truth instead of running recognition")
    ap.add_argument("--schema", default="IFC4", choices=["IFC4", "IFC2X3"])
    ap.add_argument("--out", type=Path, default=Path("out/demo"))
    args = ap.parse_args()

    if args.seed is None and args.pdf is None:
        ap.error("give --pdf or --seed")
    if args.seed is not None:
        from kelpod.evaluation import synth

        name = f"seed_{args.seed:04d}"
        directory = args.out / name
        synth.save(synth.generate(args.seed, degrade=args.degrade), directory)
        pdf = directory / "plan.pdf"
    else:
        pdf = args.pdf
        name = pdf.stem
        directory = args.out / name
        directory.mkdir(parents=True, exist_ok=True)
    ifc = directory / "model.ifc"

    t0 = time.perf_counter()
    if args.ground_truth:
        if args.seed is None:
            ap.error("--ground-truth needs --seed")
        from kelpod.evaluation import synth
        from kelpod.export import build_model, write_ifc

        graph = synth.load(directory).graph
        res = write_ifc(build_model(graph, name=name), ifc, schema=args.schema)
        note = ("Ground-truth mode: the 3D model comes from the generator's known geometry, not from "
                "reading the PDF. It demonstrates the IFC export and this viewer only.")
        stats = {"mode": "ground truth (no recognition)"}
    else:
        from kelpod.pipeline import convert

        result = convert(pdf, ifc, schema=args.schema)
        res = result.export
        rep = result.graph.report
        note = "Recognised from the PDF by the Kelpod pipeline (clean → segment → OCR → fit → IFC)."
        stats = {"mode": "pipeline"}
        if rep is not None:
            stats |= {"node closure": f"{rep.node_closure:.0%}", "room closure": f"{rep.room_closure:.0%}",
                      "dangling nodes": len(rep.dangling_nodes)}
        stats |= {t.step: f"{t.seconds:.1f} s" for t in result.timings}
    stats["total time"] = f"{time.perf_counter() - t0:.1f} s"
    counts = res.entity_counts
    stats |= {"walls": counts.get("IfcWall", 0) + counts.get("IfcWallStandardCase", 0),
              "doors": counts.get("IfcDoor", 0), "windows": counts.get("IfcWindow", 0),
              "rooms": counts.get("IfcSpace", 0), "IFC re-opened": "yes" if res.reopened_ok else "NO"}

    html = write_viewer(pdf, ifc, directory / "viewer.html", title=f"Kelpod: {name} → {args.schema}",
                        note=note, stats=stats)
    print(html.resolve())


if __name__ == "__main__":
    main()
