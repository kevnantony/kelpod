"""Tunable tolerances of the geometry step (metres and degrees unless stated)."""

from __future__ import annotations

from dataclasses import dataclass

STANDARD_THICKNESSES_M = (0.10, 0.115, 0.175, 0.24, 0.30, 0.365, 0.49)
"""Common German wall thicknesses (DIN 4172 brick sizes plus mortar); fitted thicknesses snap to
these when they are within ``thickness_snap_m``."""


@dataclass(frozen=True)
class GeometryParams:
    # angles
    snap_deg: float = 6.0
    """Walls within this angle of 0/90 degrees become axis-parallel."""
    snap_abs_m: float = 0.06
    """...or whose ends deviate across the wall by less than max(this, half the thickness)."""
    snap_short_deg: float = 15.0
    short_piece_m: float = 1.0
    """Pieces shorter than this have noisy angles and snap to the axes within snap_short_deg."""
    min_skew_deg: float = 0.25
    """Residual skew below this is left alone (clean already deskews to ~0.1 degrees)."""
    diag_deg: float = 3.0
    """Walls within this angle of 45/135 degrees are snapped to exactly 45/135."""
    max_skew_deg: float = 6.0
    """Residual page skew estimated from the walls and removed if below this."""

    # clustering and merging along a wall line
    line_tol_m: float = 0.06
    line_tol_rel: float = 0.4
    """Parallel wall pieces whose centrelines differ by less than max(line_tol_m, line_tol_rel *
    thinner thickness) share one coordinate."""
    merge_gap_m: float = 0.06
    """Collinear pieces closer than this are merged without further questions."""
    small_gap_m: float = 0.30
    """Collinear gaps up to this are bridged silently (segmentation dropouts)."""
    max_bridge_m: float = 2.60
    """Collinear gaps up to this are bridged if an opening sits there or both gap ends are free."""
    thickness_compat_m: float = 0.06
    """Pieces whose thickness differs by more than this are not merged into one wall."""
    thickness_snap_m: float = 0.02
    thickness_snap_rel: float = 0.10
    """Measured thickness snaps to the nearest standard one within max(2 cm, 10 %)."""
    min_thickness_m: float = 0.04

    # junctions
    end_tol_min_m: float = 0.25
    end_tol_factor: float = 0.75
    """A wall end snaps (extends or trims) to a perpendicular wall within
    ``max(end_tol_min_m, end_tol_factor * max(thicknesses))``."""
    free_snap_m: float = 0.30
    """Non-axis walls snap their ends to nodes or walls within this distance."""
    node_tol_m: float = 0.01
    spur_len_m: float = 0.35
    """Dangling stubs shorter than this hanging off a junction are removed."""
    min_component_share: float = 0.25
    """Wall groups without a loop and shorter than this share of the main group are dropped."""

    # dimension fitting
    dim_match_tol_m: float = 0.12
    dim_max_change_m: float = 0.30
    dim_max_change_rel: float = 0.10
    dim_consistent_m: float = 0.01
    """After the fit every used label must agree to within this; otherwise the label whose removal
    helps most is dropped (greedy leave-one-out)."""
    dim_max_reject_share: float = 0.34
    scale_inlier_rel: float = 0.03
    scale_inlier_rel_loose: float = 0.12
    """Looser agreement accepted for the scale if at least three dimension spans fit it."""
    prior_weight: float = 0.02

    # rooms and openings
    min_room_area_m2: float = 0.5
    min_opening_width_m: float = 0.15
    brh_radius_m: float = 2.0
    stamp_mismatch_rel: float = 0.10

    # defaults (German Rohbau-Richtmaße)
    window_sill_m: float = 0.90
    window_height_m: float = 1.385
    door_height_m: float = 2.01
