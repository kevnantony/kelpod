"""Learned segmentation backend (``backend="learned"``).

A fine-tuned semantic segmentation network predicts a per-pixel class map on the cleaned page
(0 background, 1 wall, 2 window, 3 door, 4 room interior); a deterministic post-processor
(:func:`masks_to_segmentation`, NumPy/OpenCV only) turns the map into wall centreline segments
with thickness, openings in wall gaps and room polygons, i.e. the same :class:`Segmentation` the
classical baseline returns, so the geometry step stays deterministic.

The checkpoint is a TorchScript file (``.pt``, needs ``torch``) or an ONNX file (``.onnx``, needs
``onnxruntime``, which the main environment already has through RapidOCR). Its path comes from
the ``weights`` argument or the ``KELPOD_SEG_CHECKPOINT`` environment variable. Inference runs on
the CPU. Training code and results: ``training/README.md``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from kelpod.contracts import (
    OpeningKind,
    OpeningPx,
    RasterPage,
    RoomRegion,
    Segmentation,
    WallKind,
    WallSegment,
)

ENV_CHECKPOINT = "KELPOD_SEG_CHECKPOINT"
CLASSES = ("background", "wall", "window", "door", "room")
BG, WALL, WINDOW, DOOR, ROOM = range(5)
TRAIN_M_PER_PX = 0.0254 / 200.0 * 100.0
"""Scale the networks are trained at: 200 dpi at 1:100 (1 px = 1.27 cm)."""

LabelPredictor = Callable[[np.ndarray], np.ndarray]
"""Grayscale page (H, W) uint8 at ``TRAIN_M_PER_PX`` -> class map (H, W) uint8 (``CLASSES``)."""


# --------------------------------------------------------------------------------------------
# Model loading and inference
# --------------------------------------------------------------------------------------------


class _TorchScriptModel:
    def __init__(self, path: Path, threads: int | None) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "the learned segmentation backend needs PyTorch for a .pt checkpoint; install it "
                "(see training/README.md, `cd training && uv sync`) or use an .onnx checkpoint"
            ) from exc
        self.torch = torch
        if threads:
            torch.set_num_threads(threads)
        extra = {"meta.json": ""}
        self.model = torch.jit.load(str(path), map_location="cpu", _extra_files=extra)
        self.model.eval()
        self.meta = json.loads(extra["meta.json"] or "{}")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with self.torch.inference_mode():
            return self.model(self.torch.from_numpy(x)).numpy()


class _OnnxModel:
    def __init__(self, path: Path, threads: int | None) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise ImportError("an .onnx checkpoint needs onnxruntime") from exc
        so = ort.SessionOptions()
        # explicit thread count (usable CPUs, capped): ORT's default pins threads to CPUs that
        # may lie outside a Slurm/cgroup allocation
        so.intra_op_num_threads = threads or min(8, len(os.sched_getaffinity(0)))
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.meta = json.loads(meta.get("kelpod", "{}"))
        self.input = self.sess.get_inputs()[0].name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.input: x})[0]


def load_model(path: str | Path, *, threads: int | None = None):
    """Load a TorchScript (``.pt``) or ONNX (``.onnx``) segmentation network for CPU inference.
    The network takes (1, 1, H, W) float32 in [0, 1] (white paper = 1) and returns class logits
    (1, C, H, W)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"segmentation checkpoint not found: {p}")
    if p.suffix == ".onnx":
        return _OnnxModel(p, threads)
    return _TorchScriptModel(p, threads)


_CACHE: dict[tuple, object] = {}


def load_model_cached(path: str | Path, *, threads: int | None = None):
    """:func:`load_model`, kept per process (keyed by path, size and mtime), so a server or a
    batch run loads the network once."""
    p = Path(path)
    st = p.stat()
    key = (str(p.resolve()), st.st_size, st.st_mtime_ns, threads)
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = load_model(p, threads=threads)
    return _CACHE[key]


def predict_tiled(
    model: Callable[[np.ndarray], np.ndarray],
    gray: np.ndarray,
    *,
    tile: int = 1024,
    overlap: int = 128,
    multiple: int = 32,
) -> np.ndarray:
    """Class logits for a whole page, tile by tile (padded to ``multiple``), averaged in the
    overlaps. Returns (C, H, W) float32."""
    h, w = gray.shape
    x = gray.astype(np.float32) / 255.0
    step = tile - overlap
    out = None
    weight = np.zeros((h, w), np.float32)
    ys = list(range(0, max(h - overlap, 1), step)) if h > tile else [0]
    xs = list(range(0, max(w - overlap, 1), step)) if w > tile else [0]
    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            y0_, x0_ = max(0, y1 - tile), max(0, x1 - tile)
            crop = x[y0_:y1, x0_:x1]
            ch, cw = crop.shape
            ph, pw = -ch % multiple, -cw % multiple
            crop = np.pad(crop, ((0, ph), (0, pw)), constant_values=1.0)
            logits = model(crop[None, None])[0][:, :ch, :cw]
            if out is None:
                out = np.zeros((logits.shape[0], h, w), np.float32)
            out[:, y0_:y1, x0_:x1] += logits
            weight[y0_:y1, x0_:x1] += 1.0
    return out / np.maximum(weight, 1.0)[None]


# --------------------------------------------------------------------------------------------
# Class map -> vector segmentation (deterministic)
# --------------------------------------------------------------------------------------------

H, V = "h", "v"


@dataclass
class _Run:
    orient: str
    c: float  # centreline coordinate across (y for H, x for V)
    a0: float  # extent along
    a1: float
    th: float
    snap0: bool = False
    snap1: bool = False

    @property
    def length(self) -> float:
        return self.a1 - self.a0

    def xy(self, a: float, c: float | None = None) -> tuple[float, float]:
        c = self.c if c is None else c
        return (float(a), float(c)) if self.orient == H else (float(c), float(a))


@dataclass
class PostParams:
    min_wall_m: float = 0.05
    max_wall_m: float = 0.80
    min_wall_len_m: float = 0.10
    min_component_m: float = 0.30
    kernel_m: float = 0.55
    """Line-opening length that separates horizontal from vertical wall bands (> thickest wall)."""
    min_opening_m: float = 0.30
    max_opening_m: float = 4.0
    min_room_m2: float = 0.6
    room_frac: float = 0.5
    """A free-space region is a room if at least this share of it is predicted as room interior."""
    exterior_min_th_m: float = 0.20
    drywall_max_th_m: float = 0.15


def _profile_runs(comp: np.ndarray, orient: str, tol: float, min_len: int):
    """Split one band component into pieces of constant cross-section (top/bottom faces)."""
    m = comp if orient == H else comp.T  # along axis = columns
    cols = np.nonzero(m.any(axis=0))[0]
    if cols.size == 0:
        return []
    top = np.argmax(m[:, cols], axis=0).astype(float)
    bot = (m.shape[0] - 1 - np.argmax(m[::-1, cols], axis=0)).astype(float)
    pieces: list[list[int]] = []
    start = 0
    for i in range(1, len(cols) + 1):
        brk = i == len(cols) or cols[i] != cols[i - 1] + 1
        if not brk:
            ref_t = np.median(top[start:i])
            ref_b = np.median(bot[start:i])
            if i - start >= min_len and (abs(top[i] - ref_t) > tol or abs(bot[i] - ref_b) > tol):
                # a lasting change of either face starts a new piece
                j = min(len(cols), i + min_len)
                if (
                    np.all(np.abs(top[i:j] - ref_t) > tol) or np.all(np.abs(bot[i:j] - ref_b) > tol)
                ) and cols[j - 1] - cols[i] == j - 1 - i:
                    brk = True
        if brk:
            pieces.append([start, i])
            start = i
    out = []
    for s, e in pieces:
        if e - s < 2:
            continue
        t, b = np.median(top[s:e]), np.median(bot[s:e])
        out.append((float(cols[s]) - 0.5, float(cols[e - 1]) + 0.5, (t + b) / 2, b - t + 1))
    return out


def _bands(wall: np.ndarray, orient: str, k: int) -> np.ndarray:
    kern = np.ones((1, k), np.uint8) if orient == H else np.ones((k, 1), np.uint8)
    return cv2.morphologyEx(wall.astype(np.uint8), cv2.MORPH_OPEN, kern).astype(bool)


def _runs_from_mask(mask: np.ndarray, orient: str, px, p: PostParams) -> list[_Run]:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    runs = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 4:
            continue
        comp = lab[y : y + h, x : x + w] == i
        for a0, a1, c, th in _profile_runs(comp, orient, tol=max(2.0, px(0.03)), min_len=4):
            if th < px(p.min_wall_m) or th > px(p.max_wall_m):
                continue
            off_a, off_c = (x, y) if orient == H else (y, x)
            runs.append(_Run(orient, c + off_c, a0 + off_a, a1 + off_a, th))
    return runs


def _merge_collinear(runs: list[_Run], gap: float, tol: float) -> list[_Run]:
    out: list[_Run] = []
    for r in sorted(runs, key=lambda r: (r.orient, round(r.c), r.a0)):
        for o in out:
            if (
                o.orient == r.orient
                and abs(o.c - r.c) <= tol
                and abs(o.th - r.th) <= max(tol, 0.25 * o.th)
                and r.a0 <= o.a1 + gap
                and r.a1 >= o.a0 - gap
            ):
                lo, hi = min(o.a0, r.a0), max(o.a1, r.a1)
                wo, wr = o.length, r.length
                o.c = (o.c * wo + r.c * wr) / max(wo + wr, 1e-6)
                o.th = (o.th * wo + r.th * wr) / max(wo + wr, 1e-6)
                o.a0, o.a1 = lo, hi
                break
        else:
            out.append(_Run(r.orient, r.c, r.a0, r.a1, r.th))
    return out


def _snap_ends(runs: list[_Run], tol: float) -> None:
    """Move wall ends that lie inside (or just short of) a perpendicular wall onto that wall's
    centreline, so corners and T-junctions share one point."""
    for r in runs:
        for end in (0, 1):
            a = r.a0 if end == 0 else r.a1
            best = None
            for o in runs:
                if o.orient == r.orient:
                    continue
                if not (o.a0 - tol <= r.c <= o.a1 + tol):
                    continue
                d = abs(a - o.c)
                if d <= o.th / 2 + tol and (best is None or d < best[0]):
                    best = (d, o)
            if best is not None:
                o = best[1]
                if end == 0:
                    r.a0, r.snap0 = o.c, True
                else:
                    r.a1, r.snap1 = o.c, True
                # the perpendicular wall must reach this centreline too
                o.a0, o.a1 = min(o.a0, r.c), max(o.a1, r.c)


def _components(mask: np.ndarray, min_len: float):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if max(w, h) >= min_len:
            yield x, y, w, h


def masks_to_segmentation(
    label: np.ndarray,
    page: RasterPage,
    *,
    params: PostParams | None = None,
    backend: str = "learned",
) -> Segmentation:
    """Turn a class map (``CLASSES``, page px) into a :class:`Segmentation`.

    Assumes a deskewed plan with axis-parallel walls (as ``kelpod.clean`` delivers and the
    geometry step expects); diagonal walls are not vectorised yet.
    """
    p = params or PostParams()
    m_per_px = page.m_per_px_from_scale or TRAIN_M_PER_PX

    def px(m: float) -> float:
        return m / m_per_px

    label = np.asarray(label, np.uint8)
    wall = label == WALL
    wall = cv2.morphologyEx(wall.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(wall, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = np.maximum(stats[1:, 2], stats[1:, 3]) >= px(p.min_component_m)
    wall = keep[lab]
    windows = label == WINDOW
    doors = label == DOOR

    # 1. walls: horizontal and vertical bands, split by cross-section, merged, junctions snapped
    k = max(5, int(round(px(p.kernel_m))))
    hb, vb = _bands(wall, H, k), _bands(wall, V, k)
    runs = _runs_from_mask(hb, H, px, p) + _runs_from_mask(vb, V, px, p)
    # short pieces in neither band (wall stubs next to an opening): orientation from the
    # neighbouring opening or the stub's long side
    rest = wall & ~cv2.dilate((hb | vb).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    opening_any = cv2.dilate((windows | doors).astype(np.uint8), np.ones((5, 5), np.uint8))
    for x, y, w, h in _components(rest, px(p.min_wall_len_m)):
        sub = rest[y : y + h, x : x + w]
        ys, xs = np.nonzero(opening_any[max(0, y - 3) : y + h + 3, max(0, x - 3) : x + w + 3])
        if ys.size and max(w, h) < px(0.30):
            continue  # jamb sliver between an opening and a wall face: the opening absorbs it
        orient = (H if w >= h else V) if ys.size == 0 else (H if np.ptp(xs) >= np.ptp(ys) else V)
        if orient == H:
            runs.append(_Run(H, y + (h - 1) / 2, x - 0.5, x + w - 0.5, float(sub.sum() / w)))
        else:
            runs.append(_Run(V, x + (w - 1) / 2, y - 0.5, y + h - 0.5, float(sub.sum() / h)))
    runs = _merge_collinear(runs, gap=2.0, tol=max(2.0, px(0.03)))
    _snap_ends(runs, tol=max(3.0, px(0.05)))
    runs = [r for r in runs if r.length >= px(p.min_wall_len_m) and r.th >= px(p.min_wall_m)]

    # 2. openings: components of the door / window classes, centred on the nearest wall line
    openings: list[OpeningPx] = []
    for kind, mask in ((OpeningKind.WINDOW, windows), (OpeningKind.DOOR, doors)):
        for x, y, w, h in _components(mask, px(p.min_opening_m)):
            orient = H if w >= h else V
            a0, a1 = (x - 0.5, x + w - 0.5) if orient == H else (y - 0.5, y + h - 0.5)
            c = y + (h - 1) / 2 if orient == H else x + (w - 1) / 2
            if a1 - a0 > px(p.max_opening_m):
                continue
            best = None
            for i, r in enumerate(runs):
                if r.orient != orient or abs(r.c - c) > r.th / 2 + px(0.08):
                    continue
                d = max(0.0, r.a0 - a1, a0 - r.a1)  # gap along the line
                if d <= px(0.15) and (best is None or d < best[0]):
                    best = (d, i)
            wid = None
            if best is not None:
                wid = best[1]
                r = runs[wid]
                c = r.c
                # opening edges = jambs of the neighbouring walls on the same line
                s0 = s1 = False
                for o in runs:
                    if o.orient == orient and abs(o.c - c) <= max(2.0, 0.25 * o.th):
                        if abs(o.a1 - a0) <= px(0.08):
                            a0, s0 = o.a1, True
                        if abs(o.a0 - a1) <= px(0.08):
                            a1, s1 = o.a0, True
                # an end next to a perpendicular wall runs up to that wall's face
                for o in runs:
                    if o.orient == orient or not (o.a0 - 3 <= c <= o.a1 + 3):
                        continue
                    if not s0 and 0 <= a0 - (o.c + o.th / 2) <= px(0.30):
                        a0, s0 = o.c + o.th / 2, True
                    if not s1 and 0 <= (o.c - o.th / 2) - a1 <= px(0.30):
                        a1, s1 = o.c - o.th / 2, True
            p0 = (a0, c) if orient == H else (c, a0)
            p1 = (a1, c) if orient == H else (c, a1)
            openings.append(
                OpeningPx(
                    id=len(openings),
                    kind=kind,
                    p0=(float(p0[0]), float(p0[1])),
                    p1=(float(p1[0]), float(p1[1])),
                    wall_id=wid,
                    confidence=0.8,
                )
            )

    # 3. rooms: free space enclosed by walls and openings, mostly predicted as room interior
    barrier = (wall | windows | doors).astype(np.uint8)
    barrier = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    free = (barrier == 0).astype(np.uint8)
    nf, flab, fstats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    border = np.unique(np.concatenate([flab[0], flab[-1], flab[:, 0], flab[:, -1]]))
    outside = np.zeros(nf, bool)
    outside[border] = True
    outside[0] = False
    room_px = np.bincount(flab.ravel(), weights=(label == ROOM).ravel(), minlength=nf)
    rooms: list[RoomRegion] = []
    min_area = px(1.0) ** 2 * p.min_room_m2
    for i in range(1, nf):
        area = fstats[i, 4]
        if outside[i] or area < min_area or room_px[i] < p.room_frac * area:
            continue
        x, y, w, h, _ = fstats[i]
        comp = (flab[y : y + h, x : x + w] == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        cnt = cv2.approxPolyDP(cnt, 1.5, True).reshape(-1, 2)
        if len(cnt) < 3:
            continue
        dist = cv2.distanceTransform(np.pad(comp, 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
        sy, sx = np.unravel_index(int(np.argmax(dist)), dist.shape)
        rooms.append(
            RoomRegion(
                id=len(rooms),
                polygon=[(float(a + x), float(b + y)) for a, b in cnt],
                seed=(float(sx + x), float(sy + y)),
                area_px=float(area),
            )
        )

    # 4. wall kinds from thickness and whether one side is the outside
    out_mask = outside[flab]
    hgt, wid_ = label.shape
    segs: list[WallSegment] = []
    for i, r in enumerate(runs):
        th_m = r.th * m_per_px
        ext = False
        for side in (-1, 1):
            cc = r.c + side * (r.th / 2 + 3)
            for t in (0.3, 0.5, 0.7):
                a = r.a0 + t * r.length
                xx, yy = r.xy(a, cc)
                xi, yi = int(round(xx)), int(round(yy))
                if 0 <= yi < hgt and 0 <= xi < wid_ and out_mask[yi, xi]:
                    ext = True
        if ext and th_m >= p.exterior_min_th_m:
            kind = WallKind.EXTERIOR
        elif th_m > p.drywall_max_th_m:
            kind = WallKind.INTERIOR
        else:
            kind = WallKind.DRYWALL
        segs.append(
            WallSegment(
                id=i,
                p0=r.xy(r.a0),
                p1=r.xy(r.a1),
                thickness_px=float(r.th),
                kind=kind,
                confidence=0.8,
            )
        )
    return Segmentation(
        wall_mask=wall,
        walls=segs,
        openings=openings,
        rooms=rooms,
        backend=backend,
        notes=[f"{backend}: {len(segs)} walls, {len(openings)} openings, {len(rooms)} rooms"],
    )


# --------------------------------------------------------------------------------------------
# The backend
# --------------------------------------------------------------------------------------------


class LearnedSegmenter:
    """Segmentation network + deterministic vectorisation.

    ``weights``: checkpoint path (``.pt`` TorchScript or ``.onnx``); defaults to the
    ``KELPOD_SEG_CHECKPOINT`` environment variable. ``predictor``: any callable that maps a
    grayscale page at the training scale to a class map (used to evaluate other networks, e.g.
    the CubiCasa5k model zero-shot, through the same post-processing).

    ``infer_scale``: the page is shrunk by this factor (INTER_AREA) before the network and the
    logits are resized back (bilinear). 0.65 with whole-page inference (``tile=2048``) cuts the CPU
    time of the network about 10x against 1024 px tiles at full scale (docs/EXPERIMENTS.md,
    learned-fast).
    """

    name = "learned"

    def __init__(
        self,
        weights: str | Path | None = None,
        *,
        predictor: LabelPredictor | None = None,
        params: PostParams | None = None,
        threads: int | None = None,
        tile: int = 2048,
        infer_scale: float = 0.65,
    ) -> None:
        env = os.environ.get(ENV_CHECKPOINT)
        self.weights = Path(weights) if weights else (Path(env) if env else None)
        self._predictor = predictor
        self.params = params
        self.threads = threads
        self.tile = tile
        self.infer_scale = infer_scale
        self._model = None

    def _predict(self, gray: np.ndarray) -> np.ndarray:
        if self._predictor is not None:
            return self._predictor(gray)
        if self._model is None:
            if self.weights is None:
                raise RuntimeError(
                    "the learned segmentation backend needs a checkpoint: pass weights=... or set "
                    f"{ENV_CHECKPOINT}=/path/to/model.pt (see training/README.md); or use "
                    "backend='classical'"
                )
            if not self.weights.is_file():
                raise FileNotFoundError(f"segmentation checkpoint not found: {self.weights}")
            self._model = load_model_cached(self.weights, threads=self.threads)
        h, w = gray.shape
        s = self.infer_scale
        if s != 1.0:
            gray = cv2.resize(
                gray, (max(32, round(w * s)), max(32, round(h * s))), interpolation=cv2.INTER_AREA
            )
        logits = predict_tiled(self._model, gray, tile=self.tile)
        if s != 1.0:
            logits = np.stack(
                [cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR) for c in logits]
            )
        return np.argmax(logits, axis=0).astype(np.uint8)

    def predict_labels(self, page: RasterPage) -> np.ndarray:
        """Class map in page px (resampled to the training scale and back)."""
        m_per_px = page.m_per_px_from_scale or TRAIN_M_PER_PX
        f = m_per_px / TRAIN_M_PER_PX
        gray = np.asarray(page.image, np.uint8)
        if abs(f - 1.0) > 0.02:
            size = (max(1, round(page.width * f)), max(1, round(page.height * f)))
            interp = cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR
            lab = self._predict(cv2.resize(gray, size, interpolation=interp))
            return cv2.resize(lab, (page.width, page.height), interpolation=cv2.INTER_NEAREST)
        return self._predict(gray)

    def segment(self, page: RasterPage) -> Segmentation:
        seg = masks_to_segmentation(self.predict_labels(page), page, params=self.params)
        if self.weights is not None:
            seg.notes.append(f"checkpoint: {self.weights.name}")
        return seg
