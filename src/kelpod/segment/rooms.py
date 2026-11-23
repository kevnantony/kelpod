"""Rooms as enclosed free-space components once walls and openings form a barrier."""

from __future__ import annotations

import cv2
import numpy as np

from kelpod.contracts import RoomRegion


def draw_rect(mask: np.ndarray, orient: str, c: float, a0: float, a1: float, th: float) -> None:
    """Fill an axis-aligned band (in place)."""
    n0, n1 = int(np.floor(c - th / 2)), int(np.ceil(c + th / 2))
    b0, b1 = int(np.floor(a0)), int(np.ceil(a1))
    if orient == "h":
        mask[max(0, n0) : max(0, n1), max(0, b0) : max(0, b1)] = True
    else:
        mask[max(0, b0) : max(0, b1), max(0, n0) : max(0, n1)] = True


def free_space(barrier: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """Label the free space (4-connected). Returns (n, labels, is_outside per label)."""
    free = (~barrier).astype(np.uint8)
    n, lab = cv2.connectedComponents(free, connectivity=4)
    outside = np.zeros(n, bool)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    outside[border] = True
    outside[0] = False
    return n, lab, outside


def extract_rooms(
    lab: np.ndarray, outside: np.ndarray, *, min_area_px: float, eps_px: float = 1.5
) -> list[RoomRegion]:
    rooms: list[RoomRegion] = []
    n = len(outside)
    areas = np.bincount(lab.ravel(), minlength=n)
    for i in range(1, n):
        if outside[i] or areas[i] < min_area_px:
            continue
        comp = (lab == i).astype(np.uint8)
        x0, y0, w, h = cv2.boundingRect(comp)
        sub = comp[y0 : y0 + h, x0 : x0 + w]
        contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(cnt, eps_px, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        poly = [(float(x + x0), float(y + y0)) for x, y in approx]
        dt = cv2.distanceTransform(np.pad(sub, 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
        sy, sx = np.unravel_index(int(np.argmax(dt)), dt.shape)
        rooms.append(
            RoomRegion(
                id=len(rooms),
                polygon=poly,
                seed=(float(sx + x0), float(sy + y0)),
                area_px=float(areas[i]),
            )
        )
    return rooms
