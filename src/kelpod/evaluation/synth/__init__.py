"""Deterministic synthetic German floor plans (DIN 1356-1 style) with exact ground truth.

>>> from kelpod.evaluation.synth import generate, save
>>> plan = generate(seed=1, degrade=0.5)
>>> plan.graph.rooms[0].name, plan.image.shape  # doctest: +SKIP

What is drawn (see ``docs/ARCHITECTURE.md``, "Drawing conventions"): one storey, rectangular
footprint, 3-8 rectangular rooms from a guillotine partition with octametric sizes; hatched 36.5 cm
exterior walls, hatched 24 / 17.5 cm interior walls, unhatched 11.5 / 10 cm partitions; doors
(gap + leaf + quarter arc) and windows (gap + 2 or 3 lines); three dimension chains on every side
(openings, rooms + walls, overall) with 45° ticks, extension lines and German numbers centred above
the line (vertical chains rotated 90°, read bottom-to-top); ``BRH 0,90`` next to windows; room
stamps (``Wohnen`` / ``14,52 m²``); a title ``M 1:100``. 1:100, 300 dpi by default.

Ground truth (``SyntheticPlan.graph``) is a :class:`~kelpod.contracts.PlanGraph` in plan space:
metres, y up, origin at the bottom-left corner of the *undegraded* image, wall centrelines split
at every junction, room polygons on the inner wall faces (CCW), ``scale_source="ground_truth"``.
Text annotations and the wall mask are in undegraded image px; ``degradation.affine_px`` maps them
into the degraded image.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from kelpod.contracts import PlanGraph, Point
from kelpod.evaluation.synth.degrade import Degradation
from kelpod.evaluation.synth.degrade import degrade as _degrade
from kelpod.evaluation.synth.layout import make_layout
from kelpod.evaluation.synth.render import Style, render
from kelpod.evaluation.synth.text import format_area, format_length, format_sill, render_text

__all__ = [
    "Degradation",
    "SyntheticPlan",
    "TextAnnotation",
    "format_area",
    "format_length",
    "format_sill",
    "generate",
    "load",
    "render_text",
    "save",
]

GENERATOR_VERSION = "1"
MARGIN_M = 4.0
"""Paper margin around the outer wall faces (plan metres); holds the dimension chains and title."""

TextKind = Literal["dimension", "sill", "room_name", "room_area", "title"]


class TextAnnotation(BaseModel):
    """A text drawn on the plan, in undegraded image px."""

    model_config = ConfigDict(extra="forbid")

    text: str
    kind: TextKind
    quad: list[Point]
    """Four corners clockwise from the top-left of the text as read (as ``TextItem.quad``)."""
    angle_deg: float = 0.0
    """0 = left-to-right, 90 = bottom-to-top."""
    value: float | None = None
    """Parsed value: metres for dimensions and sills, m² for room areas."""
    line: tuple[Point, Point] | None = None
    """Dimensions: the two ticks of the measured span on the dimension line (px)."""
    room_id: int | None = None
    opening_id: int | None = None
    """Sills: id of the window in ``graph.openings``."""

    @property
    def center(self) -> Point:
        xs, ys = zip(*self.quad, strict=True)
        return (sum(xs) / 4, sum(ys) / 4)


class SyntheticPlan(BaseModel):
    """One generated plan: images, ground-truth graph and drawn annotations."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    seed: int
    image: np.ndarray = Field(exclude=True, repr=False)
    """Degraded grayscale image (H, W) uint8, white paper = 255."""
    clean_image: np.ndarray = Field(exclude=True, repr=False)
    """The same plan before degradation."""
    wall_mask: np.ndarray = Field(exclude=True, repr=False)
    """(H, W) bool, wall bodies (with hatching, without opening gaps), undegraded px."""
    dpi: float
    plan_scale: float = 100.0
    graph: PlanGraph
    annotations: list[TextAnnotation] = Field(default_factory=list)
    degradation: Degradation = Field(default_factory=Degradation)
    length_style: str = "cm"
    sill_style: str = "m"
    generator_version: str = GENERATOR_VERSION

    @property
    def m_per_px(self) -> float:
        return 0.0254 / self.dpi * self.plan_scale

    def to_degraded_px(self, p: Point) -> Point:
        """Map an undegraded image point into the degraded image."""
        m = self.degradation.affine_px
        x, y = p
        return (m[0][0] * x + m[0][1] * y + m[0][2], m[1][0] * x + m[1][1] * y + m[1][2])


def generate(
    seed: int,
    degrade: float = 0.5,
    *,
    dpi: float = 300.0,
    n_rooms: int | None = None,
) -> SyntheticPlan:
    """Generate one plan. Same ``seed`` (and arguments) -> identical output.

    ``degrade`` in [0, 1] scales the scan degradation (0 = clean drawing); ``n_rooms`` defaults to a
    random 3-8.
    """
    plan_scale = 100.0
    m_per_px = 0.0254 / dpi * plan_scale
    rng = np.random.default_rng([int(seed), 0x6B656C70])  # layout stream
    layout, size = make_layout(rng, margin_m=MARGIN_M, m_per_px=m_per_px, n_rooms=n_rooms)
    style = Style(length=str(rng.choice(["cm", "sup", "m"])), sill=str(rng.choice(["m", "cm"])))
    clean, wall_mask, drawn = render(layout, size, dpi, style, m_per_px)
    # a separate stream for the degradation, so the layout does not change with ``degrade``
    image, info = _degrade(clean, float(degrade), np.random.default_rng([int(seed), 0x64656772]))
    annotations = [
        TextAnnotation(
            text=d.text,
            kind=d.kind,
            quad=d.quad,
            angle_deg=d.angle_deg,
            value=d.value,
            line=d.line,
            room_id=d.room_id,
            opening_id=d.opening_id,
        )
        for d in drawn
    ]
    from kelpod.evaluation.metrics import closure_metrics

    graph = layout.graph
    graph.report = closure_metrics(graph)
    return SyntheticPlan(
        seed=int(seed),
        image=image,
        clean_image=clean,
        wall_mask=wall_mask,
        dpi=dpi,
        plan_scale=plan_scale,
        graph=graph,
        annotations=annotations,
        degradation=info,
        length_style=style.length,
        sill_style=style.sill,
    )


def save(plan: SyntheticPlan, directory: str | Path, *, clean: bool = False) -> Path:
    """Write ``plan.png``, an image-only ``plan.pdf`` (scanner-like JPEG inside, page size from the
    dpi) and ``ground_truth.json``; with ``clean=True`` also ``plan_clean.png`` and
    ``wall_mask.png``. Returns the directory."""
    from PIL import Image

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(plan.image, mode="L")
    img.save(out / "plan.png", dpi=(plan.dpi, plan.dpi))
    img.save(out / "plan.pdf", resolution=plan.dpi, quality=95)
    if clean:
        Image.fromarray(plan.clean_image, mode="L").save(
            out / "plan_clean.png", dpi=(plan.dpi, plan.dpi)
        )
        Image.fromarray((plan.wall_mask * 255).astype(np.uint8), mode="L").save(
            out / "wall_mask.png"
        )
    (out / "ground_truth.json").write_text(plan.model_dump_json(indent=1), encoding="utf-8")
    return out


class GroundTruth(BaseModel):
    """What :func:`load` returns: everything from ``ground_truth.json`` plus the image."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    seed: int
    dpi: float
    plan_scale: float
    graph: PlanGraph
    annotations: list[TextAnnotation]
    degradation: Degradation
    image: np.ndarray | None = Field(default=None, exclude=True, repr=False)


def load(directory: str | Path, *, image: bool = False) -> GroundTruth:
    """Read a directory written by :func:`save` (optionally with the degraded image)."""
    d = Path(directory)
    data = json.loads((d / "ground_truth.json").read_text(encoding="utf-8"))
    gt = GroundTruth(
        seed=data["seed"],
        dpi=data["dpi"],
        plan_scale=data["plan_scale"],
        graph=PlanGraph.model_validate(data["graph"]),
        annotations=[TextAnnotation.model_validate(a) for a in data["annotations"]],
        degradation=Degradation.model_validate(data["degradation"]),
    )
    if image:
        from PIL import Image

        gt.image = np.asarray(Image.open(d / "plan.png").convert("L"))
    return gt
