"""Shared data contracts between the pipeline steps.

Every step is a function ``step(input contract(s), **options) -> output contract``.
Modules only talk
to each other through these types, so each one can be built and tested on its own.

Coordinate systems
------------------
* **Image space** (``Px`` suffix / "px"): pixels of the cleaned raster, x to the right, y down,
  origin at the top-left corner of the image.
* **Plan space** ("m"): metres, x to the right, y **up**, origin at the bottom-left corner of the
  image. Use :func:`px_to_m` / :func:`m_to_px` to convert. IFC uses plan space directly
  (z up, storey floor at z = 0).

Units: lengths in image space are pixels; everything else is metres, square metres and degrees.
Angles are counter-clockwise in plan space.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

Point = tuple[float, float]
"""A 2D point ``(x, y)``: pixels in image space or metres in plan space, as the field name says."""

CONTRACT_VERSION = "0.1"


class _Model(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


# --------------------------------------------------------------------------------------------
# Step 1: clean the scan
# --------------------------------------------------------------------------------------------


class RasterPage(_Model):
    """One cleaned page of the input, ready for segmentation and OCR."""

    image: np.ndarray = Field(exclude=True, repr=False)
    """Cleaned grayscale image, shape (H, W), dtype uint8, white paper = 255, ink = 0."""
    ink: np.ndarray = Field(exclude=True, repr=False)
    """Binarised ink mask, shape (H, W), dtype bool, True where there is ink."""
    dpi: float
    """Resolution of ``image`` after normalisation (dots per inch of the paper)."""
    source: str
    """Path of the input file this page came from."""
    page_index: int = 0
    deskew_deg: float = 0.0
    """Rotation that was removed from the scan (degrees, counter-clockwise positive)."""
    plan_scale: float | None = 100.0
    """Drawing scale denominator, e.g. 100 for 1:100. ``None`` if unknown."""

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def m_per_px_from_scale(self) -> float | None:
        """Metres per pixel implied by ``dpi`` and ``plan_scale`` (a guess; dimensions override)."""
        if self.plan_scale is None:
            return None
        return 0.0254 / self.dpi * self.plan_scale


# --------------------------------------------------------------------------------------------
# Step 2: segmentation
# --------------------------------------------------------------------------------------------


class WallKind(StrEnum):
    EXTERIOR = "exterior"  # load-bearing / solid masonry, usually hatched (DIN 1356-1)
    INTERIOR = "interior"  # interior masonry
    DRYWALL = "drywall"  # light partition, usually unhatched outline
    UNKNOWN = "unknown"


class OpeningKind(StrEnum):
    DOOR = "door"
    WINDOW = "window"
    UNKNOWN = "unknown"


class WallSegment(_Model):
    """A straight wall piece in image space, described by its centreline."""

    id: int
    p0: Point
    p1: Point
    thickness_px: float
    kind: WallKind = WallKind.UNKNOWN
    confidence: float = 1.0


class OpeningPx(_Model):
    """A door or window in image space. ``p0``/``p1`` are the ends of the gap on the wall
    centreline (so the width is ``|p1 - p0|``)."""

    id: int
    kind: OpeningKind
    p0: Point
    p1: Point
    wall_id: int | None = None
    """Wall the opening sits in (a ``WallSegment.id``); ``None`` if not yet assigned."""
    hinge: Point | None = None
    """Door hinge point if a swing arc was found."""
    swing_into: Point | None = None
    """A point on the side the door opens into (inside the arc), if known."""
    confidence: float = 1.0


class RoomRegion(_Model):
    """A rough room region in image space (free space enclosed by walls)."""

    id: int
    polygon: list[Point]
    """Outer boundary, image space, not repeated at the end."""
    seed: Point
    """A point well inside the region."""
    area_px: float


class Segmentation(_Model):
    """Output of the segmentation step for one page."""

    wall_mask: np.ndarray = Field(exclude=True, repr=False)
    """Shape (H, W) bool, True on wall pixels (wall bodies including hatching)."""
    walls: list[WallSegment] = Field(default_factory=list)
    openings: list[OpeningPx] = Field(default_factory=list)
    rooms: list[RoomRegion] = Field(default_factory=list)
    backend: str = "classical"
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------------------
# Step 3: OCR of annotations
# --------------------------------------------------------------------------------------------


class TextItem(_Model):
    """One piece of recognised text."""

    text: str
    quad: list[Point]
    """Four corners in image space (clockwise from top-left of the text as read)."""
    angle_deg: float = 0.0
    """Reading direction in image space: 0 = left-to-right, 90 = bottom-to-top (rotated text)."""
    confidence: float = 1.0
    alternatives: list[tuple[str, float]] = Field(default_factory=list)
    """n-best readings ``(text, probability)``, most probable first, if the engine provides them
    (empty otherwise). ``text`` is the engine's own best guess and need not be the first entry."""

    @property
    def center(self) -> Point:
        xs, ys = zip(*self.quad, strict=True)
        return (sum(xs) / len(xs), sum(ys) / len(ys))


class DimensionLabel(_Model):
    """A parsed length from a dimension chain (Maßkette)."""

    value_m: float
    text: str
    anchor: Point
    """Centre of the text, image space."""
    angle_deg: float = 0.0
    """Direction of the dimension line in image space (0 = horizontal, 90 = vertical)."""
    line: tuple[Point, Point] | None = None
    """Endpoints (ticks) of the measured span on the dimension line, image space, if found."""
    confidence: float = 1.0
    origin: Literal["read", "corrected", "inferred"] = "read"
    """``read``: as recognised; ``corrected``: another n-best reading or the chain arithmetic
    replaced the recognised value; ``inferred``: no label was read, the value comes from the
    other labels of its chains (``text`` is then empty)."""


class SillHeight(_Model):
    """A sill height (Brüstungshöhe), e.g. ``BRH 87,5`` -> 0.875 m."""

    value_m: float
    text: str
    anchor: Point
    confidence: float = 1.0


class RoomStamp(_Model):
    """A room stamp, e.g. ``Wohnen 14,52 m²``."""

    name: str
    area_m2: float | None = None
    text: str
    anchor: Point
    confidence: float = 1.0


class OcrAnnotations(_Model):
    items: list[TextItem] = Field(default_factory=list)
    dimensions: list[DimensionLabel] = Field(default_factory=list)
    sill_heights: list[SillHeight] = Field(default_factory=list)
    room_stamps: list[RoomStamp] = Field(default_factory=list)
    backend: str = "none"
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------------------
# Steps 4-5: fitted vector plan as a graph, plus consistency checks
# --------------------------------------------------------------------------------------------


class Node(_Model):
    id: int
    x: float
    y: float


class Wall(_Model):
    """A straight wall between two graph nodes (centreline), plan space."""

    id: int
    start: int
    end: int
    thickness_m: float
    kind: WallKind = WallKind.UNKNOWN


class Opening(_Model):
    """A door or window placed on a wall, plan space."""

    id: int
    kind: OpeningKind
    wall_id: int
    offset_m: float
    """Distance along the wall centreline from the wall's ``start`` node to the near edge."""
    width_m: float
    sill_m: float | None = None
    """Sill height above the finished floor (doors: 0)."""
    height_m: float | None = None
    """Clear opening height (doors typically 2.01 m, windows from sill to lintel)."""


class Room(_Model):
    id: int
    polygon: list[Point]
    """Closed boundary (inner faces of walls), plan space, counter-clockwise, not repeated."""
    name: str | None = None
    area_m2: float
    stamp_area_m2: float | None = None
    """Area written on the room stamp, if any (for cross-checking)."""


class ConsistencyReport(_Model):
    """Result of the graph checks (step 5)."""

    n_nodes: int
    n_walls: int
    n_rooms: int
    n_components: int
    """Connected components of the wall graph (1 for a fully connected storey)."""
    dangling_nodes: list[int] = Field(default_factory=list)
    """Nodes with degree 1 (wall ends that do not meet another wall)."""
    node_closure: float
    """Share of nodes with degree >= 2, in [0, 1]."""
    room_closure: float
    """Share of detected room regions that became valid closed polygons, in [0, 1]."""
    dimension_residual_m: float | None = None
    """RMS difference between fitted wall lengths and the OCR dimensions they were matched to."""
    issues: list[str] = Field(default_factory=list)

    @property
    def is_closed(self) -> bool:
        return self.n_components == 1 and not self.dangling_nodes and self.room_closure == 1.0


class PlanGraph(_Model):
    """The fitted, closed vector plan of one storey, plan space (metres)."""

    nodes: list[Node] = Field(default_factory=list)
    walls: list[Wall] = Field(default_factory=list)
    openings: list[Opening] = Field(default_factory=list)
    rooms: list[Room] = Field(default_factory=list)
    m_per_px: float
    scale_source: Literal["dimensions", "plan_scale", "default", "ground_truth"]
    image_height_px: int
    """Needed to convert back to image space (see :func:`m_to_px`)."""
    report: ConsistencyReport | None = None

    def node(self, node_id: int) -> Node:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)


# --------------------------------------------------------------------------------------------
# Step 6: 3D model and IFC export
# --------------------------------------------------------------------------------------------


class Wall3D(_Model):
    id: int
    start: Point
    end: Point
    thickness_m: float
    height_m: float
    kind: WallKind = WallKind.UNKNOWN


class Opening3D(_Model):
    id: int
    kind: OpeningKind
    wall_id: int
    offset_m: float
    width_m: float
    sill_m: float
    height_m: float


class Space(_Model):
    id: int
    name: str
    polygon: list[Point]
    height_m: float
    area_m2: float


class Storey(_Model):
    name: str = "EG"
    elevation_m: float = 0.0
    height_m: float = 2.60
    """Clear room height used for wall extrusion."""
    walls: list[Wall3D] = Field(default_factory=list)
    openings: list[Opening3D] = Field(default_factory=list)
    spaces: list[Space] = Field(default_factory=list)


class BuildingModel(_Model):
    name: str = "Kelpod building"
    storeys: list[Storey] = Field(default_factory=list)


IfcSchema = Literal["IFC4", "IFC2X3"]


class ExportResult(_Model):
    path: str
    schema_: IfcSchema = Field(alias="schema")
    entity_counts: dict[str, int] = Field(default_factory=dict)
    """Counts by IFC class after re-opening, e.g. ``{"IfcWall": 12, "IfcSpace": 5}``."""
    reopened_ok: bool
    warnings: list[str] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


# --------------------------------------------------------------------------------------------
# Whole-pipeline result
# --------------------------------------------------------------------------------------------


class StepTiming(_Model):
    step: str
    seconds: float


class PipelineResult(_Model):
    source: str
    graph: PlanGraph
    model: BuildingModel
    export: ExportResult | None = None
    timings: list[StepTiming] = Field(default_factory=list)

    @property
    def total_seconds(self) -> float:
        return sum(t.seconds for t in self.timings)


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def px_to_m(p: Point, m_per_px: float, image_height_px: int) -> Point:
    """Image space (px, y down) -> plan space (m, y up)."""
    return (p[0] * m_per_px, (image_height_px - p[1]) * m_per_px)


def m_to_px(p: Point, m_per_px: float, image_height_px: int) -> Point:
    """Plan space (m, y up) -> image space (px, y down)."""
    return (p[0] / m_per_px, image_height_px - p[1] / m_per_px)
