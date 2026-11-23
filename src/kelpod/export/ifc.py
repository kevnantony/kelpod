"""3D building model -> IFC4 / IFC2X3 file, written with ``ifcopenshell.api``.

Structure: IfcProject -> IfcSite -> IfcBuilding -> IfcBuildingStorey (aggregation). Walls, doors,
windows and the optional floor slab are contained in the storey; spaces are aggregated into it.

Geometry conventions
--------------------
* Units: metre, square metre, cubic metre, radian. Plan space maps to IFC x/y, z up, storey floor at
  the storey elevation.
* Each wall gets a local placement at its (extended) start point with the local x axis along the
  wall, an ``Axis`` representation (the centreline) and a ``Body`` representation (rectangle profile
  ``length x thickness`` centred on the axis, extruded by the wall height). IFC2X3 uses
  ``IfcWallStandardCase`` with an ``IfcMaterialLayerSetUsage`` (one layer, offset ``-t/2``); IFC4
  uses ``IfcWall`` with the same layer set usage.
* Corners are closed by a butt joint: at a node where no wall continues straight through, the
  longest wall is extended by half the largest thickness of the other walls there (so the outer
  corner is filled once, not twice). Walls ending on a straight-through wall (T-junction) already
  reach its axis and overlap its body, so they are not extended. Junctions are also recorded as
  ``IfcRelConnectsPathElements`` (ATSTART / ATEND / ATPATH).
* Openings: an ``IfcOpeningElement`` box (opening width x wall thickness + 2 cm x opening height)
  voids the wall (``IfcRelVoidsElement``) and is filled by an ``IfcDoor`` / ``IfcWindow``
  (``IfcRelFillsElement``) with a parametric lining/panel body from ``ifcopenshell.api``.
* Spaces: polygon extruded by the storey height, ``Name`` = running number, ``LongName`` = room
  name, base quantities (net/gross floor area, perimeter, height).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import ifcopenshell
import ifcopenshell.api.aggregate
import ifcopenshell.api.attribute
import ifcopenshell.api.context
import ifcopenshell.api.feature
import ifcopenshell.api.geometry
import ifcopenshell.api.material
import ifcopenshell.api.owner
import ifcopenshell.api.project
import ifcopenshell.api.pset
import ifcopenshell.api.root
import ifcopenshell.api.spatial
import ifcopenshell.api.unit
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from kelpod.contracts import (
    BuildingModel,
    ExportResult,
    IfcSchema,
    OpeningKind,
    Storey,
    Wall3D,
    WallKind,
)

OPENING_MARGIN_M = 0.01
"""The opening box sticks out of each wall face by this much so the boolean cut is clean."""
SLAB_THICKNESS_M = 0.20
WINDOW_FRAME_DEPTH_M = 0.10
COLLINEAR_TOL_DEG = 5.0
NODE_TOL_M = 1e-3

COUNTED_CLASSES = (
    "IfcProject",
    "IfcSite",
    "IfcBuilding",
    "IfcBuildingStorey",
    "IfcWall",
    "IfcWallStandardCase",
    "IfcOpeningElement",
    "IfcDoor",
    "IfcWindow",
    "IfcSpace",
    "IfcSlab",
    "IfcRelVoidsElement",
    "IfcRelFillsElement",
    "IfcRelConnectsPathElements",
    "IfcRelAggregates",
    "IfcRelContainedInSpatialStructure",
    "IfcRelAssociatesMaterial",
    "IfcPropertySet",
    "IfcElementQuantity",
)

_WALL_LABEL = {
    WallKind.EXTERIOR: "Außenwand",
    WallKind.INTERIOR: "Innenwand",
    WallKind.DRYWALL: "Trennwand",
    WallKind.UNKNOWN: "Wand",
}


# --------------------------------------------------------------------------------------------
# Corner handling (pure geometry, tested separately)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WallEnd:
    wall_id: int
    at_start: bool


def _node_key(p: tuple[float, float]) -> tuple[int, int]:
    return (round(p[0] / NODE_TOL_M), round(p[1] / NODE_TOL_M))


def _unit(p0, p1) -> np.ndarray:
    d = np.subtract(p1, p0, dtype=float)
    return d / np.linalg.norm(d)


def _cross(a, b) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _outward(w: Wall3D, at_start: bool) -> np.ndarray:
    """Unit vector pointing from the wall body out through the given end."""
    d = _unit(w.start, w.end)
    return -d if at_start else d


def _junctions(walls: list[Wall3D]) -> dict[tuple[int, int], list[WallEnd]]:
    ends: dict[tuple[int, int], list[WallEnd]] = defaultdict(list)
    for w in walls:
        ends[_node_key(w.start)].append(WallEnd(w.id, True))
        ends[_node_key(w.end)].append(WallEnd(w.id, False))
    return ends


def _straight_pairs(ends: list[WallEnd], by_id: dict[int, Wall3D]) -> list[tuple[WallEnd, WallEnd]]:
    """Pairs of wall ends at one node that continue in a straight line."""
    cos_tol = math.cos(math.radians(COLLINEAR_TOL_DEG))
    pairs = []
    for i, a in enumerate(ends):
        for b in ends[i + 1 :]:
            da, db = _outward(by_id[a.wall_id], a.at_start), _outward(by_id[b.wall_id], b.at_start)
            if float(np.dot(da, db)) < -cos_tol:
                pairs.append((a, b))
    return pairs


def corner_extensions(walls: list[Wall3D]) -> dict[int, tuple[float, float]]:
    """How far to extend each wall body beyond its start and end node (metres).

    Rule: at a node where at least one pair of walls continues straight through, nothing is
    extended (straight joints and T-junctions are already closed). Otherwise (L corner, odd angle,
    free end with several walls) the longest wall at the node is extended by half the largest
    thickness of the other walls there (butt joint). Perpendicular other walls are trimmed back
    by half the main wall's thickness, so the corner is filled exactly once and wall volumes are
    not double-counted. Skewed walls are not trimmed (small overlap instead of a gap).
    Negative values mean "trim".
    """
    by_id = {w.id: w for w in walls}
    ext = {w.id: [0.0, 0.0] for w in walls}
    for ends in _junctions(walls).values():
        if len(ends) < 2 or _straight_pairs(ends, by_id):
            continue
        lengths = {e: math.dist(by_id[e.wall_id].start, by_id[e.wall_id].end) for e in ends}
        main = max(ends, key=lambda e: (lengths[e], -e.wall_id))
        main_wall = by_id[main.wall_id]
        d_main = _outward(main_wall, main.at_start)
        sines = {
            e: abs(_cross(d_main, _outward(by_id[e.wall_id], e.at_start)))
            for e in ends
            if e != main
        }
        # Perpendicular corners need exactly t/2; for skewed corners scale so the outer face
        # still reaches the other wall's outer face (capped to avoid spikes at sharp angles).
        amount = max(by_id[e.wall_id].thickness_m for e in sines) / 2.0
        amount /= max(min(sines.values()), 0.5)
        i = 0 if main.at_start else 1
        ext[main.wall_id][i] = max(ext[main.wall_id][i], amount)
        for e, s in sines.items():
            if s > 0.999:
                j = 0 if e.at_start else 1
                ext[e.wall_id][j] = min(ext[e.wall_id][j], -main_wall.thickness_m / 2.0)
    out = {}
    for k, (e0, e1) in ext.items():
        w = by_id[k]
        if math.dist(w.start, w.end) + e0 + e1 < 0.01:  # very short wall: do not trim it away
            e0, e1 = max(e0, 0.0), max(e1, 0.0)
        out[k] = (e0, e1)
    return out


def path_connections(walls: list[Wall3D]) -> list[tuple[int, str, int, str]]:
    """Wall junctions as ``(relating_id, relating_type, related_id, related_type)`` tuples.

    Types follow ``IfcConnectionTypeEnum``: ATSTART / ATEND for wall ends; a wall that runs
    straight through a node where another wall ends is given as ATPATH. The graph usually splits
    walls at T-junctions, so ATPATH only appears for unsplit through walls.
    """
    conns: list[tuple[int, str, int, str]] = []
    seen: set[tuple[int, int]] = set()

    def tag(e: WallEnd) -> str:
        return "ATSTART" if e.at_start else "ATEND"

    for ends in _junctions(walls).values():
        for i, a in enumerate(ends):
            for b in ends[i + 1 :]:
                key = (min(a.wall_id, b.wall_id), max(a.wall_id, b.wall_id))
                if a.wall_id == b.wall_id or key in seen:
                    continue
                seen.add(key)
                conns.append((a.wall_id, tag(a), b.wall_id, tag(b)))
    # Unsplit walls passing through another wall's end point.
    for w in walls:
        d = _unit(w.start, w.end)
        length = math.dist(w.start, w.end)
        for other in walls:
            if other.id == w.id:
                continue
            for at_start, p in ((True, other.start), (False, other.end)):
                v = np.subtract(p, w.start)
                along = float(np.dot(v, d))
                across = abs(_cross(d, v))
                if NODE_TOL_M < along < length - NODE_TOL_M and across < NODE_TOL_M:
                    key = (min(w.id, other.id), max(w.id, other.id))
                    if key not in seen:
                        seen.add(key)
                        conns.append((w.id, "ATPATH", other.id, tag(WallEnd(other.id, at_start))))
    return conns


# --------------------------------------------------------------------------------------------
# Writer
# --------------------------------------------------------------------------------------------


def _kelpod_version() -> str:
    try:
        from importlib.metadata import version

        return version("kelpod")
    except Exception:
        return "0.0"


def _closed(pts) -> list[tuple[float, float]]:
    """Polyline for ``add_slab_representation``, which expects the first point repeated."""
    pts = [(float(x), float(y)) for x, y in pts]
    if pts and pts[0] != pts[-1]:
        pts.append(pts[0])
    return pts


def _matrix(origin: tuple[float, float, float], x_dir: tuple[float, float]) -> np.ndarray:
    m = np.eye(4)
    x = np.array([x_dir[0], x_dir[1], 0.0])
    y = np.array([-x_dir[1], x_dir[0], 0.0])
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, (0.0, 0.0, 1.0), origin
    return m


class _Writer:
    def __init__(self, model: BuildingModel, schema: IfcSchema, floor_slab: bool):
        self.model = model
        self.schema = schema
        self.floor_slab = floor_slab
        self.f = ifcopenshell.api.project.create_file(version=schema)
        self.warnings: list[str] = []
        self._layer_sets: dict[tuple[WallKind, float], object] = {}

    # -- setup --------------------------------------------------------------------------------

    def _setup(self):
        f = self.f
        # Owner history is mandatory in IFC2X3; the api picks these up from the file. Generic
        # placeholders only: no personal data in exported files.
        org = ifcopenshell.api.owner.add_organisation(f, identification="Kelpod", name="Kelpod")
        person = ifcopenshell.api.owner.add_person(
            f, identification="kelpod", family_name="Kelpod", given_name="Export"
        )
        ifcopenshell.api.owner.add_person_and_organisation(f, person=person, organisation=org)
        ifcopenshell.api.owner.add_application(
            f,
            application_developer=org,
            version=_kelpod_version(),
            application_full_name="Kelpod",
            application_identifier="Kelpod",
        )
        self.project = ifcopenshell.api.root.create_entity(f, "IfcProject", name=self.model.name)
        units = [
            ifcopenshell.api.unit.add_si_unit(f, unit_type=t)
            for t in ("LENGTHUNIT", "AREAUNIT", "VOLUMEUNIT", "PLANEANGLEUNIT")
        ]
        ifcopenshell.api.unit.assign_unit(f, units=units)
        model_ctx = ifcopenshell.api.context.add_context(f, context_type="Model")
        self.body = ifcopenshell.api.context.add_context(
            f, "Model", "Body", "MODEL_VIEW", parent=model_ctx
        )
        self.axis = ifcopenshell.api.context.add_context(
            f, "Model", "Axis", "GRAPH_VIEW", parent=model_ctx
        )
        self.site = ifcopenshell.api.root.create_entity(f, "IfcSite", name="Grundstück")
        self.building = ifcopenshell.api.root.create_entity(f, "IfcBuilding", name=self.model.name)
        ifcopenshell.api.aggregate.assign_object(f, [self.site], self.project)
        ifcopenshell.api.aggregate.assign_object(f, [self.building], self.site)
        for product in (self.site, self.building):
            ifcopenshell.api.geometry.edit_object_placement(f, product=product)
        if self.schema == "IFC2X3":
            for product in (self.site, self.building):
                product.CompositionType = "ELEMENT"

    def _layer_set(self, kind: WallKind, thickness: float):
        key = (kind, round(thickness, 4))
        if key not in self._layer_sets:
            label = f"{_WALL_LABEL[kind]} {thickness * 100:.1f} cm".replace(".", ",")
            mat = ifcopenshell.api.material.add_material(self.f, name=_WALL_LABEL[kind])
            ls = ifcopenshell.api.material.add_material_set(
                self.f, name=label, set_type="IfcMaterialLayerSet"
            )
            layer = ifcopenshell.api.material.add_layer(self.f, layer_set=ls, material=mat)
            ifcopenshell.api.material.edit_layer(
                self.f, layer=layer, attributes={"LayerThickness": thickness}
            )
            self._layer_sets[key] = ls
        return self._layer_sets[key]

    # -- storey content -----------------------------------------------------------------------

    def _storey(self, storey: Storey):
        f = self.f
        st = ifcopenshell.api.root.create_entity(f, "IfcBuildingStorey", name=storey.name)
        ifcopenshell.api.aggregate.assign_object(f, [st], self.building)
        ifcopenshell.api.geometry.edit_object_placement(
            f, product=st, matrix=_matrix((0.0, 0.0, storey.elevation_m), (1.0, 0.0))
        )
        st.Elevation = storey.elevation_m
        if self.schema == "IFC2X3":
            st.CompositionType = "ELEMENT"

        ext = corner_extensions(storey.walls)
        walls: dict[int, tuple[object, Wall3D, float]] = {}
        contained = []
        for w in storey.walls:
            ifc_wall, e0 = self._wall(w, ext[w.id], storey)
            walls[w.id] = (ifc_wall, w, e0)
            contained.append(ifc_wall)

        for rid, rtype, oid, otype in path_connections(storey.walls):
            rel = ifcopenshell.api.root.create_entity(f, "IfcRelConnectsPathElements")
            ifcopenshell.api.attribute.edit_attributes(
                f,
                product=rel,
                attributes=dict(
                    RelatingElement=walls[rid][0],
                    RelatedElement=walls[oid][0],
                    RelatingPriorities=[],
                    RelatedPriorities=[],
                    RelatingConnectionType=rtype,
                    RelatedConnectionType=otype,
                ),
            )

        for o in storey.openings:
            if o.wall_id not in walls:
                self.warnings.append(f"opening {o.id}: wall {o.wall_id} not found, skipped")
                continue
            contained.append(self._opening(o, *walls[o.wall_id], storey))

        if self.floor_slab:
            slab = self._slab(storey, ext)
            if slab is not None:
                contained.append(slab)

        if contained:
            ifcopenshell.api.spatial.assign_container(f, contained, st)

        spaces = [self._space(i, s, storey) for i, s in enumerate(storey.spaces, start=1)]
        if spaces:
            ifcopenshell.api.aggregate.assign_object(f, spaces, st)

    def _wall(self, w: Wall3D, ext: tuple[float, float], storey: Storey):
        f = self.f
        d = _unit(w.start, w.end)
        length = math.dist(w.start, w.end) + ext[0] + ext[1]
        origin = np.asarray(w.start) - d * ext[0]
        cls = "IfcWallStandardCase" if self.schema == "IFC2X3" else "IfcWall"
        if self.schema == "IFC2X3":
            ptype = None
        else:
            ptype = {
                WallKind.DRYWALL: "PARTITIONING",
                WallKind.EXTERIOR: "SOLIDWALL",
                WallKind.INTERIOR: "SOLIDWALL",
            }.get(w.kind, "NOTDEFINED")
        name = f"{_WALL_LABEL[w.kind]} {w.id}"
        wall = ifcopenshell.api.root.create_entity(f, cls, name=name, predefined_type=ptype)
        wall.Tag = str(w.id)
        ifcopenshell.api.geometry.edit_object_placement(
            f,
            product=wall,
            matrix=_matrix((origin[0], origin[1], storey.elevation_m), (d[0], d[1])),
        )
        body = ifcopenshell.api.geometry.add_wall_representation(
            f,
            context=self.body,
            length=length,
            height=w.height_m,
            thickness=w.thickness_m,
            offset=-w.thickness_m / 2.0,
        )
        axis = ifcopenshell.api.geometry.add_axis_representation(
            f, context=self.axis, axis=((0.0, 0.0), (length, 0.0))
        )
        ifcopenshell.api.geometry.assign_representation(f, product=wall, representation=axis)
        ifcopenshell.api.geometry.assign_representation(f, product=wall, representation=body)

        usage = ifcopenshell.api.material.assign_material(
            f,
            products=[wall],
            type="IfcMaterialLayerSetUsage",
            material=self._layer_set(w.kind, w.thickness_m),
        )
        if usage is not None:
            usage = usage[0] if isinstance(usage, list) else usage
            if usage.is_a("IfcRelAssociatesMaterial"):
                usage = usage.RelatingMaterial
            usage.LayerSetDirection = "AXIS2"
            usage.DirectionSense = "POSITIVE"
            usage.OffsetFromReferenceLine = -w.thickness_m / 2.0

        props: dict[str, object] = {"Reference": name}
        if w.kind != WallKind.UNKNOWN:
            props["IsExternal"] = w.kind == WallKind.EXTERIOR
            props["LoadBearing"] = w.kind != WallKind.DRYWALL
        pset = ifcopenshell.api.pset.add_pset(f, product=wall, name="Pset_WallCommon")
        ifcopenshell.api.pset.edit_pset(f, pset=pset, properties=props)

        qto_name = "BaseQuantities" if self.schema == "IFC2X3" else "Qto_WallBaseQuantities"
        qto = ifcopenshell.api.pset.add_qto(f, product=wall, name=qto_name)
        ifcopenshell.api.pset.edit_qto(
            f,
            qto=qto,
            properties={
                "Length": length,
                "Width": w.thickness_m,
                "Height": w.height_m,
                "GrossFootprintArea": length * w.thickness_m,
                "GrossVolume": length * w.thickness_m * w.height_m,
            },
        )
        return wall, ext[0]

    def _opening(self, o, wall, w: Wall3D, e0: float, storey: Storey):
        f = self.f
        d = _unit(w.start, w.end)
        n = np.array([-d[1], d[0]])
        t = w.thickness_m
        along = np.asarray(w.start) + d * o.offset_m
        z = storey.elevation_m + o.sill_m

        # Opening box: local x along the wall, y across it, from the face at -t/2 - margin.
        depth = t + 2 * OPENING_MARGIN_M
        base = along - n * (t / 2 + OPENING_MARGIN_M)
        opening = ifcopenshell.api.root.create_entity(
            f, "IfcOpeningElement", name=f"Öffnung {o.id}", predefined_type="OPENING"
        )
        ifcopenshell.api.geometry.edit_object_placement(
            f, product=opening, matrix=_matrix((base[0], base[1], z), (d[0], d[1]))
        )
        rep = ifcopenshell.api.geometry.add_wall_representation(
            f, context=self.body, length=o.width_m, height=o.height_m, thickness=depth
        )
        ifcopenshell.api.geometry.assign_representation(f, product=opening, representation=rep)
        ifcopenshell.api.feature.add_feature(f, feature=opening, element=wall)

        # Filling: placed on the wall face at -t/2, lining as deep as the wall.
        is_door = o.kind == OpeningKind.DOOR
        cls = "IfcDoor" if is_door else "IfcWindow"
        label = "Tür" if is_door else "Fenster"
        fill = ifcopenshell.api.root.create_entity(f, cls, name=f"{label} {o.id}")
        fill.Tag = str(o.id)
        fill.OverallWidth = o.width_m
        fill.OverallHeight = o.height_m
        if self.schema != "IFC2X3":
            fill.PredefinedType = "DOOR" if is_door else "WINDOW"
        face = along - n * (t / 2)
        ifcopenshell.api.geometry.edit_object_placement(
            f, product=fill, matrix=_matrix((face[0], face[1], z), (d[0], d[1]))
        )
        lining_t = min(0.05, o.width_m / 4)
        if is_door:
            # Frame (Zarge) as deep as the wall, no casings so nothing sticks out sideways.
            lining = {
                "LiningDepth": t,
                "LiningOffset": 0.0,
                "LiningThickness": lining_t,
                "CasingThickness": 0.0,
                "CasingDepth": 0.0,
            }
        else:
            # Window frame ~10 cm deep, centred in the wall (the api adds ~2.5 cm for the sash).
            depth = min(WINDOW_FRAME_DEPTH_M, t / 2)
            lining = {
                "LiningDepth": depth,
                "LiningOffset": max((t - depth - 0.025) / 2, 0.0),
                "LiningThickness": lining_t,
            }
        try:
            if is_door:
                body = ifcopenshell.api.geometry.add_door_representation(
                    f,
                    context=self.body,
                    overall_height=o.height_m,
                    overall_width=o.width_m,
                    lining_properties=lining,
                )
            else:
                body = ifcopenshell.api.geometry.add_window_representation(
                    f,
                    context=self.body,
                    overall_height=o.height_m,
                    overall_width=o.width_m,
                    lining_properties=lining,
                )
        except Exception as exc:  # parametric generator failed: fall back to a plain panel
            self.warnings.append(f"{label} {o.id}: parametric body failed ({exc}); used a box")
            body = None
        if body is None:
            body = ifcopenshell.api.geometry.add_wall_representation(
                f, context=self.body, length=o.width_m, height=o.height_m, thickness=min(t, 0.07)
            )
        ifcopenshell.api.geometry.assign_representation(f, product=fill, representation=body)
        ifcopenshell.api.feature.add_filling(f, opening=opening, element=fill)

        is_ext = w.kind == WallKind.EXTERIOR
        pset = ifcopenshell.api.pset.add_pset(
            f, product=fill, name="Pset_DoorCommon" if is_door else "Pset_WindowCommon"
        )
        ifcopenshell.api.pset.edit_pset(
            f, pset=pset, properties={"Reference": f"{label} {o.id}", "IsExternal": is_ext}
        )
        return fill

    def _space(self, i: int, s, storey: Storey):
        f = self.f
        space = ifcopenshell.api.root.create_entity(f, "IfcSpace", name=f"{i:02d}")
        space.LongName = s.name
        if self.schema == "IFC2X3":
            space.CompositionType = "ELEMENT"
            space.InteriorOrExteriorSpace = "INTERNAL"
        else:
            space.PredefinedType = "INTERNAL"
        ifcopenshell.api.geometry.edit_object_placement(
            f, product=space, matrix=_matrix((0.0, 0.0, storey.elevation_m), (1.0, 0.0))
        )
        rep = ifcopenshell.api.geometry.add_slab_representation(
            f, context=self.body, depth=s.height_m, polyline=_closed(s.polygon)
        )
        ifcopenshell.api.geometry.assign_representation(f, product=space, representation=rep)
        poly = Polygon(s.polygon)
        qto_name = "BaseQuantities" if self.schema == "IFC2X3" else "Qto_SpaceBaseQuantities"
        qto = ifcopenshell.api.pset.add_qto(f, product=space, name=qto_name)
        ifcopenshell.api.pset.edit_qto(
            f,
            qto=qto,
            properties={
                "NetFloorArea": s.area_m2,
                "GrossFloorArea": s.area_m2,
                "NetPerimeter": poly.length,
                "Height": s.height_m,
                "NetVolume": s.area_m2 * s.height_m,
            },
        )
        return space

    def _slab(self, storey: Storey, ext: dict[int, tuple[float, float]]):
        """Floor slab under the outer outline of walls and spaces (optional, a guess)."""
        parts = [Polygon(s.polygon) for s in storey.spaces]
        for w in storey.walls:
            d = _unit(w.start, w.end)
            length = math.dist(w.start, w.end) + sum(ext[w.id])
            o = np.asarray(w.start) - d * ext[w.id][0]
            rect = box(0.0, -w.thickness_m / 2, length, w.thickness_m / 2)
            coords = [
                tuple(o + d * x + np.array([-d[1], d[0]]) * y) for x, y in rect.exterior.coords
            ]
            parts.append(Polygon(coords))
        outline = unary_union([p.buffer(0) for p in parts if p.is_valid or p.buffer(0).is_valid])
        if outline.is_empty:
            return None
        if outline.geom_type == "MultiPolygon":
            outline = max(outline.geoms, key=lambda g: g.area)
            self.warnings.append("floor slab: outline has several parts, used the largest")
        # Merge the near-duplicate vertices left by the union (they break the extrusion).
        outline = Polygon(outline.exterior).simplify(1e-3, preserve_topology=True)
        pts = [(round(x, 4), round(y, 4)) for x, y in outline.exterior.coords][:-1]
        f = self.f
        ptype = "BASESLAB"
        slab = ifcopenshell.api.root.create_entity(
            f, "IfcSlab", name="Bodenplatte", predefined_type=ptype
        )
        ifcopenshell.api.geometry.edit_object_placement(
            f,
            product=slab,
            matrix=_matrix((0.0, 0.0, storey.elevation_m - SLAB_THICKNESS_M), (1.0, 0.0)),
        )
        rep = ifcopenshell.api.geometry.add_slab_representation(
            f, context=self.body, depth=SLAB_THICKNESS_M, polyline=_closed(pts)
        )
        ifcopenshell.api.geometry.assign_representation(f, product=slab, representation=rep)
        pset = ifcopenshell.api.pset.add_pset(f, product=slab, name="Pset_SlabCommon")
        ifcopenshell.api.pset.edit_pset(f, pset=pset, properties={"IsExternal": True})
        return slab

    def build(self) -> ifcopenshell.file:
        self._setup()
        if not self.model.storeys:
            self.warnings.append("model has no storeys")
        for storey in self.model.storeys:
            self._storey(storey)
        if self.schema == "IFC2X3":
            # A few api calls (e.g. feature.add_filling) leave OwnerHistory empty, but IFC2X3
            # requires it on every rooted entity.
            missing = [e for e in self.f.by_type("IfcRoot") if e.OwnerHistory is None]
            if missing:
                history = ifcopenshell.api.owner.create_owner_history(self.f)
                for e in missing:
                    e.OwnerHistory = history
        return self.f


# --------------------------------------------------------------------------------------------
# Checks after writing
# --------------------------------------------------------------------------------------------


def count_entities(f: ifcopenshell.file) -> dict[str, int]:
    """Exact counts (no subtypes) for the classes Kelpod writes."""
    counts = {}
    for cls in COUNTED_CLASSES:
        try:
            n = len(f.by_type(cls, include_subtypes=False))
        except RuntimeError:  # class not in this schema
            continue
        if n:
            counts[cls] = n
    return counts


def schema_problems(f: ifcopenshell.file, *, express_rules: bool = False) -> list[str]:
    """Run ``ifcopenshell.validate`` and return its findings as plain strings."""
    import ifcopenshell.validate

    logger = ifcopenshell.validate.json_logger()
    ifcopenshell.validate.validate(f, logger, express_rules=express_rules)
    out = []
    for s in logger.statements:
        inst = s.get("instance")
        out.append(f"schema: {s.get('message', s)}" + (f" ({inst})" if inst is not None else ""))
    return out


def geometry_problems(f: ifcopenshell.file) -> list[str]:
    """Build every product's body with ``ifcopenshell.geom``; report products that fail."""
    import ifcopenshell.geom

    settings = ifcopenshell.geom.settings()
    problems = []
    for product in f.by_type("IfcProduct"):
        if product.Representation is None or product.is_a("IfcOpeningElement"):
            continue
        try:
            shape = ifcopenshell.geom.create_shape(settings, product)
            if not len(shape.geometry.verts):
                problems.append(f"geometry: {product.is_a()} {product.Name!r} is empty")
        except Exception as exc:
            problems.append(f"geometry: {product.is_a()} {product.Name!r} failed ({exc})")
    return problems


def write_ifc(
    model: BuildingModel,
    path: str | Path,
    *,
    schema: IfcSchema = "IFC4",
    floor_slab: bool = False,
    express_rules: bool = False,
) -> ExportResult:
    """Write the model with IfcOpenShell, then re-open the file to validate it.

    After writing, the file is re-opened, entities are counted, ``ifcopenshell.validate`` runs
    (schema types, cardinalities, inverse attributes; ``express_rules=True`` also checks WHERE
    rules, which is slower) and every body is built with ``ifcopenshell.geom``. Any problem is
    returned as a warning instead of raising.
    """
    if schema not in ("IFC4", "IFC2X3"):
        raise ValueError(f"unsupported schema {schema!r}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = _Writer(model, schema, floor_slab)
    writer.build().write(str(path))

    warnings = list(writer.warnings)
    try:
        f = ifcopenshell.open(str(path))
    except Exception as exc:
        warnings.append(f"re-open failed: {exc}")
        return ExportResult(path=str(path), schema=schema, reopened_ok=False, warnings=warnings)

    counts = count_entities(f)
    warnings += schema_problems(f, express_rules=express_rules)
    warnings += geometry_problems(f)
    n_walls = sum(len(s.walls) for s in model.storeys)
    n_written = counts.get("IfcWall", 0) + counts.get("IfcWallStandardCase", 0)
    if n_written != n_walls:
        warnings.append(f"wrote {n_written} walls, model has {n_walls}")
    return ExportResult(
        path=str(path), schema=schema, entity_counts=counts, reopened_ok=True, warnings=warnings
    )
