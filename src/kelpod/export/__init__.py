"""Step 6: extrude to 3D and write IFC4 / IFC2x3."""

from __future__ import annotations

from pathlib import Path

from kelpod.contracts import BuildingModel, ExportResult, IfcSchema, PlanGraph


def build_model(
    graph: PlanGraph, *, storey_height_m: float = 2.60, name: str = "Kelpod building"
) -> BuildingModel:
    """Extrude the plan graph to a one-storey 3D model with default opening heights."""
    raise NotImplementedError


def write_ifc(
    model: BuildingModel, path: str | Path, *, schema: IfcSchema = "IFC4"
) -> ExportResult:
    """Write the model with IfcOpenShell, then re-open the file to validate it."""
    raise NotImplementedError
