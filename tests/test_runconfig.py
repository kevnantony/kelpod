"""Run configuration: learned is the default backend, and without a usable checkpoint the pipeline
fails clearly instead of silently falling back to classical."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from kelpod import runconfig


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.delenv(runconfig.ENV_CHECKPOINT, raising=False)
    monkeypatch.delenv(runconfig.ENV_SEED, raising=False)
    p = tmp_path / "kelpod-run.toml"
    monkeypatch.setenv(runconfig.ENV_CONFIG, str(p))
    return p


def test_repo_config_defaults_to_learned():
    assert runconfig.DEFAULT_SEGMENT_BACKEND == "learned"
    assert runconfig.default_segment_backend() == "learned"


def test_missing_config_fails_clearly(cfg):
    with pytest.raises(runconfig.CheckpointError, match="KELPOD_SEG_CHECKPOINT"):
        runconfig.resolve_checkpoint()
    assert runconfig.default_segment_backend() == "learned"  # no config: still learned


def test_missing_file_and_bad_hash(cfg, tmp_path):
    ck = tmp_path / "m.onnx"
    cfg.write_text(f'[[checkpoints]]\nseed = 1\npath = "{ck}"\nsha256 = "00"\n')
    with pytest.raises(runconfig.CheckpointError, match="not found"):
        runconfig.resolve_checkpoint()
    ck.write_bytes(b"model")
    with pytest.raises(runconfig.CheckpointError, match="sha256"):
        runconfig.resolve_checkpoint()


def test_seed_selection_and_env_override(cfg, tmp_path, monkeypatch):
    a, b = tmp_path / "a.onnx", tmp_path / "b.onnx"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    sha = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (a, b)}
    cfg.write_text(
        f'segment_backend = "learned"\n'
        f'[[checkpoints]]\nseed = 1\npath = "{a}"\nsha256 = "{sha[a]}"\n'
        f'[[checkpoints]]\nseed = 2\npath = "{b}"\nsha256 = "{sha[b]}"\n'
    )
    assert runconfig.resolve_checkpoint().path == a
    monkeypatch.setenv(runconfig.ENV_SEED, "2")
    assert runconfig.resolve_checkpoint().path == b
    monkeypatch.setenv(runconfig.ENV_SEED, "3")
    with pytest.raises(runconfig.CheckpointError):
        runconfig.resolve_checkpoint()
    monkeypatch.setenv(runconfig.ENV_CHECKPOINT, str(a))
    assert runconfig.resolve_checkpoint().path == a


def test_pipeline_does_not_fall_back(cfg, tmp_path):
    import cv2

    from kelpod.pipeline import convert

    img = np.full((400, 400), 255, np.uint8)
    cv2.rectangle(img, (50, 50), (350, 350), 0, 8)
    src = tmp_path / "plan.png"
    cv2.imwrite(str(src), img)
    with pytest.raises(runconfig.CheckpointError):
        convert(src, None, ocr_backend="none")
    res = convert(src, None, ocr_backend="none", segment_backend="classical")
    assert res.graph is not None
