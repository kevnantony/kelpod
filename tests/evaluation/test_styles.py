"""Held-out test style (docs/EVAL_PROTOCOL.md §2): train sampling never yields a held-out option."""

from __future__ import annotations

import json

import numpy as np
import pytest

from kelpod.evaluation.synth import generate, load, save
from kelpod.evaluation.synth.styles import (
    AXES,
    FONTS,
    TRAIN_DEFAULT,
    heldout_font_files,
    pool,
    sample_style,
)


def test_pools_are_disjoint_and_cover_every_option():
    for axis, options in AXES.items():
        train, held = set(pool(axis, "train")), set(pool(axis, "heldout"))
        assert train and held, axis
        assert not train & held
        assert train | held == set(options)
        assert all(not options[n].heldout for n in train)
        assert all(options[n].heldout for n in held)


def test_train_sampling_never_yields_heldout():
    for seed in range(2000):
        c = sample_style("train", np.random.default_rng(seed))
        assert c.split == "train"
        assert not c.is_heldout_any
        for axis, options in AXES.items():
            assert not options[getattr(c, axis)].heldout


def test_heldout_sampling_only_yields_heldout_and_varies():
    seen = {axis: set() for axis in AXES}
    for seed in range(500):
        c = sample_style("heldout", np.random.default_rng(seed))
        for axis, options in AXES.items():
            assert options[getattr(c, axis)].heldout
            seen[axis].add(getattr(c, axis))
    for axis in AXES:
        assert seen[axis] == set(pool(axis, "heldout"))


def test_unknown_style_rejected():
    with pytest.raises(ValueError):
        pool("font", "test")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        generate(0, 0.0, style="test")  # type: ignore[arg-type]


def test_default_generation_is_train_style():
    for seed in (0, 1, 2):
        p = generate(seed, 0.5)
        assert p.style == TRAIN_DEFAULT
        assert generate(seed, 0.5, style="train").style == TRAIN_DEFAULT


def test_default_font_is_not_heldout():
    from kelpod.evaluation.synth.text import font_path

    assert font_path() not in heldout_font_files()


def test_heldout_font_override_refused_for_train(monkeypatch):
    import os

    from kelpod.evaluation.synth import text

    files = [f for f in sorted(heldout_font_files()) if os.path.isfile(f)]
    assert files, "held-out font files must be installed to build held-out splits"
    monkeypatch.setenv("KELPOD_SYNTH_FONT", files[0])
    text.font_path.cache_clear()
    try:
        with pytest.raises(ValueError, match="held-out"):
            generate(0, 0.0)
    finally:
        monkeypatch.delenv("KELPOD_SYNTH_FONT")
        text.font_path.cache_clear()


def test_heldout_same_layout_different_drawing(tmp_path):
    a, b = generate(21, 0.6), generate(21, 0.6, style="heldout")
    assert a.graph.model_dump() == b.graph.model_dump()
    assert not np.array_equal(a.clean_image, b.clean_image)
    assert b.style.split == "heldout" and b.style.is_heldout_any
    # deterministic
    c = generate(21, 0.6, style="heldout")
    assert np.array_equal(b.image, c.image) and b.style == c.style
    # save writes style.json only for held-out plans; load reads it back
    save(a, tmp_path / "a")
    save(b, tmp_path / "b")
    assert not (tmp_path / "a" / "style.json").exists()
    assert json.loads((tmp_path / "b" / "style.json").read_text())["split"] == "heldout"
    assert load(tmp_path / "b").style["font"] == b.style.font
    assert load(tmp_path / "a").style is None


def test_pdf_is_byte_reproducible(tmp_path):
    p = generate(3, 0.3)
    save(p, tmp_path / "x")
    save(p, tmp_path / "y")
    assert (tmp_path / "x" / "plan.pdf").read_bytes() == (tmp_path / "y" / "plan.pdf").read_bytes()


@pytest.mark.parametrize("axis", ["hatch", "line_weight", "damage"])
def test_every_heldout_option_renders(axis, monkeypatch):
    """Each held-out hatch / line weight / damage option draws and changes the image."""
    import kelpod.evaluation.synth as synth

    base = generate(5, 0.8)
    for name in pool(axis, "heldout"):
        forced = TRAIN_DEFAULT.__class__(
            **{**TRAIN_DEFAULT.to_dict(), "split": "heldout", axis: name}
        )
        monkeypatch.setattr(synth, "sample_style", lambda split, rng, forced=forced: forced)
        p = synth.generate(5, 0.8, style="heldout")
        assert p.style == forced
        assert p.image.dtype == np.uint8 and p.image.shape == base.image.shape
        assert not np.array_equal(p.image, base.image), (axis, name)
        assert p.graph.model_dump() == base.graph.model_dump()


def test_every_heldout_font_is_usable():
    from kelpod.evaluation.synth.text import render_text

    for o in FONTS.values():
        if o.heldout:
            alpha, _ = render_text("Küche 12,5⁵", 30.0, o.spec)
            assert alpha.max() > 0.5
            ys = np.nonzero(alpha.max(axis=1) > 0.3)[0]
            assert 25 <= ys.max() - ys.min() + 1 <= 80  # digit height respected roughly


def test_degrade_range_up_to_1_2():
    generate(1, 1.2)
    with pytest.raises(ValueError):
        generate(1, 1.3)
