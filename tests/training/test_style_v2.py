"""learned-v2 style sampler: train styles never use held-out or dev-proxy options; dev styles use
only the reserved ones; the guard rejects held-out look-alikes."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

TRAINING = Path(__file__).resolve().parents[2] / "training"
sys.path.insert(0, str(TRAINING))

from kelpod_train import style_v1 as V1  # noqa: E402
from kelpod_train import style_v2 as V2  # noqa: E402


def test_train_sampler_avoids_heldout_and_dev_options():
    rng = np.random.default_rng(7)
    seen = {"font": set(), "hatch": set(), "damage": set()}
    for _ in range(4000):
        st = V2.sample_train_style_v2(rng)  # runs the guard
        assert not st.dev
        assert st.font not in V2.DEV_FONTS
        assert st.hatch not in V2.DEV_HATCH
        assert st.damage not in V2.DEV_DAMAGE
        seen["font"].add(st.font)
        seen["hatch"].add(st.hatch)
        seen["damage"].add(st.damage)
    assert set(V2.NEW_TRAIN_FONTS) <= seen["font"]
    assert set(V2.NEW_TRAIN_HATCH) <= seen["hatch"]
    assert set(V2.NEW_TRAIN_DAMAGE) <= seen["damage"]


def test_dev_sampler_uses_only_reserved_options():
    rng = np.random.default_rng(3)
    for _ in range(300):
        st = V2.sample_dev_style(rng)
        assert st.dev and st.font in V2.DEV_FONTS and st.hatch in V2.DEV_HATCH
        assert st.damage in V2.DEV_DAMAGE


def test_new_fonts_are_not_heldout_families():
    for name in {**V2.NEW_TRAIN_FONTS, **V2.DEV_FONTS}.values():
        low = name.lower().replace("-", "").replace("_", "")
        assert not any(t in low for t in V1.FORBIDDEN_FONT_TOKENS)


@pytest.mark.parametrize(
    "change",
    [
        {"font": "urw_gothic"},
        {"damage": "diazo"},
        {"damage": "photocopy"},
        {"damage": "faded"},
        {"hatch": "cross"},
        {"hatch": "solid"},
        {"hatch": "diag135"},
        {"line_scale": 1.6},
        {"wall_line_scale": 0.7},
        {"augraphy": ("BadPhotoCopy",)},
        {"hatch": "diag20"},  # a dev-only option in a train style
    ],
)
def test_guard_rejects(change):
    base = V2.sample_train_style_v2(np.random.default_rng(0))
    with pytest.raises(V1.HeldOutStyleError):
        V2.check_not_heldout_v2(dataclasses.replace(base, **change))
