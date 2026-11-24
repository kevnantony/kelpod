"""Learned segmentation backend: deterministic vectoriser (main env) and TorchScript loading
(only where torch and segmentation_models_pytorch are installed, i.e. the training env)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from kelpod.contracts import OpeningKind, RasterPage
from kelpod.segment.learned import (
    DOOR,
    ROOM,
    WALL,
    WINDOW,
    LearnedSegmenter,
    masks_to_segmentation,
)


def _page(h: int, w: int) -> RasterPage:
    img = np.full((h, w), 255, np.uint8)
    return RasterPage(image=img, ink=img < 128, dpi=200.0, source="test", plan_scale=100.0)


def _two_room_label() -> np.ndarray:
    """Two rooms (x 50-650 and 650-1050, y 50-450 outer faces), 29 px exterior walls, 19 px
    partition with a door, a window in the top wall."""
    lab = np.zeros((500, 1100), np.uint8)
    cv2.rectangle(lab, (79, 79), (1021, 421), ROOM, -1)
    t = 29
    lab[50 : 50 + t, 50:1050] = WALL
    lab[450 - t : 450, 50:1050] = WALL
    lab[50:450, 50 : 50 + t] = WALL
    lab[50:450, 1050 - t : 1050] = WALL
    lab[50:450, 640:659] = WALL
    lab[200:270, 640:659] = DOOR  # 0.89 m door in the partition
    lab[50 : 50 + t, 200:300] = WINDOW  # 1.27 m window in the top wall
    return lab


def test_vectoriser_two_rooms():
    lab = _two_room_label()
    seg = masks_to_segmentation(lab, _page(*lab.shape))
    assert len(seg.rooms) == 2
    areas = sorted(r.area_px for r in seg.rooms)
    assert areas[0] == pytest.approx(361 * 342, rel=0.03)
    kinds = sorted(str(o.kind) for o in seg.openings)
    assert kinds == ["door", "window"]
    door = next(o for o in seg.openings if o.kind == OpeningKind.DOOR)
    assert abs(door.p0[0] - 649) < 2 and abs(abs(door.p1[1] - door.p0[1]) - 70) < 3
    # every wall is axis-parallel, thickness near the drawn one
    for w in seg.walls:
        assert w.p0[0] == pytest.approx(w.p1[0]) or w.p0[1] == pytest.approx(w.p1[1])
        assert 15 <= w.thickness_px <= 32
    # exterior corner: horizontal and vertical centrelines meet at one point
    ends = {(round(p[0]), round(p[1])) for w in seg.walls for p in (w.p0, w.p1)}
    assert (64, 64) in ends or (65, 65) in ends or (64, 65) in ends or (65, 64) in ends


def test_predictor_hook_and_missing_checkpoint(monkeypatch, tmp_path):
    lab = _two_room_label()
    seg = LearnedSegmenter(predictor=lambda g: lab).segment(_page(*lab.shape))
    assert seg.backend == "learned" and len(seg.rooms) == 2
    monkeypatch.delenv("KELPOD_SEG_CHECKPOINT", raising=False)
    with pytest.raises(RuntimeError, match="KELPOD_SEG_CHECKPOINT"):
        LearnedSegmenter().segment(_page(64, 64))
    with pytest.raises(FileNotFoundError):
        LearnedSegmenter(weights=tmp_path / "missing.pt").segment(_page(64, 64))


def test_torchscript_checkpoint_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    smp = pytest.importorskip("segmentation_models_pytorch")
    import json

    net = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=1, classes=5)
    net.eval()
    ts = torch.jit.trace(net, torch.ones(1, 1, 64, 64))
    path = tmp_path / "model.pt"
    ts.save(str(path), _extra_files={"meta.json": json.dumps({"classes": 5})})
    seg = LearnedSegmenter(weights=path, tile=256).segment(_page(300, 330))
    assert seg.wall_mask.shape == (300, 330)
    assert seg.backend == "learned"
