"""Drawing styles of the synthetic generator, split into **train** and **held-out** (test-only).

``docs/EVAL_PROTOCOL.md`` §2: the test-synth and fresh splits use a held-out style whose fonts,
hatching patterns, line weights and scan-damage profiles are never sampled for training. Every
option below carries a ``heldout`` flag; :func:`sample_style` only ever draws from the pool of the
requested split, and ``generate(..., style="train")`` (the default) is byte-identical to the
generator before the styles existed (the train pools hold exactly the old fixed choices).

Adding a new *train* option is allowed (it changes training data, not the frozen splits). Moving an
option from held-out to train is not: it leaks the test style into training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

StyleSplit = Literal["train", "heldout"]
STYLE_SPLITS: tuple[str, ...] = ("train", "heldout")


@dataclass(frozen=True)
class FontOption:
    name: str
    spec: str | None
    """``None`` = the generator default (:func:`kelpod.evaluation.synth.text.font_path`, DejaVu Sans
    with a Hershey-simplex fallback); a font file path; or ``hershey:<cv2 font face>``."""
    heldout: bool


@dataclass(frozen=True)
class HatchOption:
    name: str
    pattern: Literal["diag45", "diag135", "cross", "solid"]
    spacing_mm: float
    heldout: bool


@dataclass(frozen=True)
class LineWeightOption:
    name: str
    scale: float
    """Multiplies every line width (walls, hatching, thin lines, ticks)."""
    heldout: bool


@dataclass(frozen=True)
class DamageOption:
    name: str
    heldout: bool


_URW = "/usr/share/fonts/opentype/urw-base35"

FONTS: dict[str, FontOption] = {
    o.name: o
    for o in (
        FontOption("default", None, heldout=False),
        FontOption("urw_gothic", f"{_URW}/URWGothic-Book.otf", heldout=True),
        FontOption("nimbus_sans_narrow", f"{_URW}/NimbusSansNarrow-Regular.otf", heldout=True),
        FontOption("nimbus_mono", f"{_URW}/NimbusMonoPS-Regular.otf", heldout=True),
        FontOption("hershey_duplex", "hershey:2", heldout=True),  # cv2.FONT_HERSHEY_DUPLEX
        FontOption("hershey_complex", "hershey:3", heldout=True),  # cv2.FONT_HERSHEY_COMPLEX
    )
}
HATCHES: dict[str, HatchOption] = {
    o.name: o
    for o in (
        HatchOption("diag45", "diag45", 1.6, heldout=False),
        HatchOption("diag135_dense", "diag135", 1.0, heldout=True),
        HatchOption("cross", "cross", 2.0, heldout=True),
        HatchOption("solid", "solid", 0.0, heldout=True),
    )
}
LINE_WEIGHTS: dict[str, LineWeightOption] = {
    o.name: o
    for o in (
        LineWeightOption("standard", 1.0, heldout=False),
        LineWeightOption("heavy", 1.6, heldout=True),
        LineWeightOption("fine", 0.7, heldout=True),
    )
}
DAMAGES: dict[str, DamageOption] = {
    o.name: o
    for o in (
        DamageOption("scan", heldout=False),
        DamageOption("diazo", heldout=True),  # Lichtpause: tinted, mottled paper, pale lines
        DamageOption("photocopy", heldout=True),  # hard contrast, speckle, toner streaks, edge
        DamageOption("faded", heldout=True),  # patchy fading and broken lines
    )
}
AXES = {"font": FONTS, "hatch": HATCHES, "line_weight": LINE_WEIGHTS, "damage": DAMAGES}


def pool(axis: str, split: StyleSplit) -> list[str]:
    """Option names of one axis that belong to ``split`` (sorted, so sampling is stable)."""
    if split not in STYLE_SPLITS:
        raise ValueError(f"style must be one of {STYLE_SPLITS}, not {split!r}")
    want = split == "heldout"
    return sorted(name for name, o in AXES[axis].items() if o.heldout == want)


def heldout_font_files() -> set[str]:
    return {o.spec for o in FONTS.values() if o.heldout and o.spec and ":" not in o.spec}


@dataclass(frozen=True)
class StyleChoice:
    """The options one plan was drawn with."""

    split: StyleSplit
    font: str
    hatch: str
    line_weight: str
    damage: str

    @property
    def is_heldout_any(self) -> bool:
        return any(AXES[a][getattr(self, a)].heldout for a in AXES)

    def to_dict(self) -> dict:
        return asdict(self)


TRAIN_DEFAULT = StyleChoice("train", "default", "diag45", "standard", "scan")


def sample_style(split: StyleSplit, rng: np.random.Generator | None = None) -> StyleChoice:
    """Draw one option per axis from the pool of ``split``.

    For ``train`` an axis with a single option consumes no random numbers, so the default
    generator output does not depend on ``rng`` while the train pools hold one option each.
    """
    picks = {}
    for axis in AXES:
        names = pool(axis, split)
        if len(names) == 1:
            picks[axis] = names[0]
        else:
            if rng is None:
                raise ValueError("an rng is needed to sample a style with several options")
            picks[axis] = str(rng.choice(names))
    choice = StyleChoice(split, **picks)
    if split == "train" and choice.is_heldout_any:  # defensive; pool() already excludes them
        raise AssertionError(f"train style sampled a held-out option: {choice}")
    return choice
