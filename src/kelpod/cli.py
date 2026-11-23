"""Command line: ``kelpod convert plan.pdf -o model.ifc``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Floor-plan scans to IFC.")


@app.callback()
def main() -> None:
    """Kelpod command line."""


@app.command()
def convert(
    source: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="PDF/PNG/JPG/TIFF")],
    output: Annotated[Path, typer.Option("-o", "--output", help="IFC file to write")] = Path(
        "model.ifc"
    ),
    schema: Annotated[str, typer.Option(help="IFC4 or IFC2X3")] = "IFC4",
    dpi: Annotated[float, typer.Option(help="Working resolution")] = 200.0,
    scale: Annotated[float, typer.Option(help="Plan scale denominator (100 = 1:100)")] = 100.0,
    storey_height: Annotated[float, typer.Option(help="Clear room height in metres")] = 2.60,
    page: Annotated[int, typer.Option(help="Page of a multi-page PDF (0-based)")] = 0,
    report: Annotated[Path | None, typer.Option(help="Write a JSON report here")] = None,
) -> None:
    """Convert a floor-plan scan to a closed IFC model."""
    from kelpod.pipeline import convert as run

    schema = schema.upper()
    if schema not in ("IFC4", "IFC2X3"):
        raise typer.BadParameter("schema must be IFC4 or IFC2X3")
    res = run(
        source,
        output,
        schema=schema,
        dpi=dpi,
        plan_scale=scale,
        storey_height_m=storey_height,
        page_index=page,
    )
    rep = res.graph.report
    typer.echo(
        f"wrote {output} ({schema}), re-opened: {res.export.reopened_ok if res.export else '-'}"
    )
    if rep is not None:
        typer.echo(
            f"walls {rep.n_walls}, rooms {rep.n_rooms}, components {rep.n_components}, "
            f"dangling nodes {len(rep.dangling_nodes)}, node closure {rep.node_closure:.1%}, "
            f"room closure {rep.room_closure:.1%}"
        )
        for issue in rep.issues:
            typer.echo(f"  issue: {issue}")
    typer.echo(
        "time: "
        + ", ".join(f"{t.step} {t.seconds:.2f}s" for t in res.timings)
        + f", total {res.total_seconds:.2f}s"
    )
    if report is not None:
        report.write_text(
            json.dumps(
                {
                    "source": res.source,
                    "export": res.export.model_dump(by_alias=True) if res.export else None,
                    "report": rep.model_dump() if rep else None,
                    "timings": [t.model_dump() for t in res.timings],
                    "total_seconds": res.total_seconds,
                },
                indent=2,
            )
        )
