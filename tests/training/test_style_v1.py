"""learned-v1 training-style sampler: it must never produce an option that the generator flags as
held out for test-synth / fresh (``kelpod.evaluation.synth.styles``, docs/EVAL_PROTOCOL.md §2)."""

from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from kelpod.evaluation.synth import styles as S

TRAINING = Path(__file__).resolve().parents[2] / "training"
sys.path.insert(0, str(TRAINING))

from kelpod_train import style_v1 as V  # noqa: E402


def _heldout():
    return {
        "font_specs": {o.spec for o in S.FONTS.values() if o.heldout and o.spec},
        "hatch_patterns": {o.pattern for o in S.HATCHES.values() if o.heldout},
        "line_scales": [o.scale for o in S.LINE_WEIGHTS.values() if o.heldout],
        "damages": {n for n, o in S.DAMAGES.items() if o.heldout},
    }


def test_generator_still_flags_heldout_options():
    h = _heldout()
    assert h["font_specs"] and h["hatch_patterns"] and h["line_scales"] and h["damages"]


def test_sampler_never_produces_heldout_options():
    h = _heldout()
    hershey_heldout = {int(s.split(":")[1]) for s in h["font_specs"] if s.startswith("hershey:")}
    rng = np.random.default_rng(12345)
    seen = {"font": set(), "hatch": set(), "damage": set(), "aug": set()}
    for _ in range(20000):
        st = V.sample_train_style(rng)
        spec = V.TRAIN_FONTS[st.font]
        assert spec not in h["font_specs"]
        if spec and spec.startswith("hershey:"):
            assert int(spec.split(":")[1]) & 15 not in hershey_heldout
        if spec and not spec.startswith("hershey:"):
            name = Path(spec).name.lower()
            for fam in ("gothic", "nimbussans", "mono"):
                assert fam not in name.replace("-", "")
        assert st.hatch not in h["hatch_patterns"]
        assert st.hatch not in {"diag135", "cross", "solid"}
        if st.hatch in ("diag45", "double45", "dashed45"):
            assert st.hatch_spacing_mm >= 1.3  # held-out dense 135° = 1.0 mm (90° rotations)
        for s in (st.line_scale, st.wall_line_scale, st.thin_line_scale, st.hatch_line_scale):
            assert 0.85 - 1e-9 <= s <= 1.35 + 1e-9
            assert all(abs(s - x) >= 0.12 for x in h["line_scales"])
        assert st.damage not in h["damages"]
        assert not set(st.augraphy) & set(V.AUGRAPHY_FORBIDDEN)
        seen["font"].add(st.font)
        seen["hatch"].add(st.hatch)
        seen["damage"].add(st.damage)
        seen["aug"].update(st.augraphy)
    # the randomisation is real: every train option is drawn
    assert seen["font"] == set(V.TRAIN_FONTS)
    assert seen["hatch"] == set(V.HATCH_KINDS)
    assert seen["damage"] == set(V.DAMAGE_KINDS)
    assert seen["aug"] == set(V.AUGRAPHY_ALLOWED)


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
        {"hatch": "diag45", "hatch_spacing_mm": 1.0},
        {"line_scale": 1.6},
        {"wall_line_scale": 0.7},
        {"augraphy": ("BadPhotoCopy",)},
    ],
)
def test_guard_rejects_heldout(change):
    base = V.sample_train_style(np.random.default_rng(0))
    V.check_not_heldout(base)
    with pytest.raises(V.HeldOutStyleError):
        V.check_not_heldout(dataclasses.replace(base, **change))


def test_guard_rejects_heldout_font_specs():
    for name, spec in (("x", "/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-Regular.otf"),
                       ("y", "hershey:2"), ("z", "hershey:3")):
        V.TRAIN_FONTS[name] = spec
        try:
            with pytest.raises(V.HeldOutStyleError):
                V.check_not_heldout(
                    dataclasses.replace(V.sample_train_style(np.random.default_rng(1)), font=name)
                )
        finally:
            del V.TRAIN_FONTS[name]


def test_training_renderer_never_asks_for_the_heldout_style():
    """The v1 renderer and degrader do not call the generator's held-out style machinery."""
    for f in ("render_v1.py", "degrade_v1.py", "build.py"):
        src = (TRAINING / "kelpod_train" / f).read_text()
        assert "apply_damage" not in src
        assert not re.search(r"""style\s*=\s*["']heldout["']""", src)
        assert "sample_style(" not in src
