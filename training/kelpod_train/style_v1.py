"""Training-only drawing-style and degradation randomisation (experiment ``learned-v1``).

Every option here is a **train** option. ``docs/EVAL_PROTOCOL.md`` §2 holds out a test style whose
fonts, hatching, line weights and scan-damage profiles are flagged in
``kelpod.evaluation.synth.styles`` (that module is protected and only read here). The pools below
are chosen to stay clear of every held-out option, with margins, and :func:`check_not_heldout`
re-checks each sampled style against the held-out flags at sampling time:

* fonts: never a held-out font file or Hershey face, and conservatively never one of their
  families (URW Gothic, Nimbus Sans / Sans Narrow, any monospace, Hershey duplex / complex /
  triplex / complex-small), nor condensed sans faces;
* hatching: no 135° hatching, no cross-hatching of any kind, no solid fill. 45° single hatching
  only at spacings >= 1.3 mm (the held-out dense 135° hatch is 1.0 mm; training rotates crops by
  90°, so a 45° hatch at 1.0 mm would *be* the held-out hatch);
* line weights: a global scale in [0.85, 1.35] (held out: 0.7 "fine", 1.6 "heavy"), small
  per-element jitter, the product kept in the same range;
* damage: none of ``diazo`` (Lichtpause: pale lines on tinted, mottled paper), ``photocopy``
  (hard contrast + toner speckle + feed streaks + dark edge) or ``faded`` (patchy fading + line
  breaks), and none of the Augraphy effects that mimic them (BadPhotoCopy, DirtyDrum,
  DirtyRollers, LowInk*, ColorPaper, PageBorder, Faxify). Paper tint is never combined with
  pale ink.

The sampler only uses numpy; it is imported by the test ``tests/training/test_style_v1.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

# --------------------------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------------------------

_DJ = "/usr/share/fonts/truetype/dejavu"
_URW = "/usr/share/fonts/opentype/urw-base35"

TRAIN_FONTS: dict[str, str | None] = {
    "default": None,  # DejaVu Sans (the generator default)
    "dejavu_sans_oblique": f"{_DJ}/DejaVuSans-Oblique.ttf",
    "dejavu_sans_bold": f"{_DJ}/DejaVuSans-Bold.ttf",
    "dejavu_serif": f"{_DJ}/DejaVuSerif.ttf",
    "nimbus_roman": f"{_URW}/NimbusRoman-Regular.otf",
    "c059": f"{_URW}/C059-Roman.otf",
    "p052": f"{_URW}/P052-Roman.otf",
    "bookman_light": f"{_URW}/URWBookman-Light.otf",
    "z003_hand": f"{_URW}/Z003-MediumItalic.otf",  # calligraphic, stands in for hand lettering
    "hershey_simplex": "hershey:0",
    "hershey_plain": "hershey:1",
    "hershey_script": "hershey:6",
    "hershey_simplex_italic": "hershey:16",
}
FORBIDDEN_FONT_TOKENS = (
    "gothic", "nimbussans", "narrow", "mono", "condensed", "courier",
)
"""Substrings (lower case, file name) that are never used: held-out families and look-alikes."""
FORBIDDEN_HERSHEY = {2, 3, 4, 5}
"""Hershey duplex (2, held out), complex (3, held out), triplex (4) and complex-small (5)."""

HATCH_KINDS = ("diag45", "diag30", "diag60", "horizontal", "double45", "dashed45",
               "stipple", "none")
"""Masonry hatching patterns. Angles are on screen, rising to the right. Never 135°, cross or
solid. ``none`` = outline only (empty wall bodies)."""
HATCH_SPACING_MM = (1.3, 3.0)
FORBIDDEN_HATCH_KINDS = {"diag135", "cross", "solid", "fill"}

LINE_SCALE = (0.85, 1.35)
HELDOUT_LINE_SCALES = (0.7, 1.6)
LINE_SCALE_MARGIN = 0.12

DOOR_KINDS = ("arc", "arc", "leaf_only", "double", "sliding", "rect_leaf", "dashed_arc", "gap")
WINDOW_KINDS = ("two", "three", "four", "frame", "sill", "single")
DAMAGE_KINDS = ("scan", "pencil", "ink_bleed", "bleed_through", "texture", "letterpress")
"""Own scan-damage profiles (train only). Never ``diazo``, ``photocopy`` or ``faded``."""
FORBIDDEN_DAMAGE = {"diazo", "photocopy", "faded"}
AUGRAPHY_ALLOWED = ("InkBleed", "BleedThrough", "NoiseTexturize", "SubtleNoise", "Letterpress",
                    "Markup", "Scribbles")
AUGRAPHY_FORBIDDEN = ("BadPhotoCopy", "DirtyDrum", "DirtyRollers", "LowInkRandomLines",
                      "LowInkPeriodicLines", "ColorPaper", "PageBorder", "Faxify", "ColorShift",
                      "LowLightNoise", "DepthSimulatedBlur")


@dataclass(frozen=True)
class TrainStyle:
    font: str
    hatch: str
    hatch_spacing_mm: float
    hatch_all_walls: bool
    wall_outline: bool
    line_scale: float
    wall_line_scale: float
    thin_line_scale: float
    hatch_line_scale: float
    door: str
    window: str
    clutter: float
    """0..1: how much furniture, grid lines, notes, stamps, title block, scribbles."""
    damage: str
    damage_strength: float
    scan_degrade: float
    """Strength of the generator's ordinary ``scan`` degradation (train option)."""
    perspective: float
    curl: float
    wobble: float
    scan_dpi: float
    jpeg_quality: int | None
    augraphy: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["augraphy"] = list(self.augraphy)
        d["split"] = "train"
        return d


# --------------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------------


def sample_train_style(rng: np.random.Generator, *, plain_p: float = 0.15) -> TrainStyle:
    """Draw one randomised **train** style. With probability ``plain_p`` the drawing style is the
    generator default (diag45 1.6 mm, standard lines, arc doors) so the default stays covered;
    degradations are still randomised."""
    plain = rng.random() < plain_p
    fonts = sorted(TRAIN_FONTS)
    font = "default" if plain else str(rng.choice(fonts))
    if plain:
        hatch, spacing = "diag45", 1.6
    else:
        hatch = str(rng.choice(HATCH_KINDS, p=[0.25, 0.1, 0.1, 0.1, 0.1, 0.1, 0.13, 0.12]))
        spacing = float(rng.uniform(*HATCH_SPACING_MM))
    g = 1.0 if plain else float(rng.uniform(*LINE_SCALE))

    def jit() -> float:
        return 1.0 if plain else float(np.clip(g * rng.uniform(0.9, 1.1), *LINE_SCALE))

    damage = str(rng.choice(DAMAGE_KINDS, p=[0.35, 0.15, 0.15, 0.12, 0.13, 0.10]))
    # scan dpi: 150-600, log-uniform-ish
    dpi = float(np.exp(rng.uniform(np.log(150), np.log(600))))
    q = None if rng.random() < 0.3 else int(rng.integers(20, 96))
    aug: list[str] = []
    for name, p in (("InkBleed", 0.15), ("BleedThrough", 0.1), ("NoiseTexturize", 0.2),
                    ("SubtleNoise", 0.2), ("Letterpress", 0.08), ("Markup", 0.12),
                    ("Scribbles", 0.08)):
        if rng.random() < p:
            aug.append(name)
    style = TrainStyle(
        font=font,
        hatch=hatch,
        hatch_spacing_mm=spacing,
        hatch_all_walls=bool(not plain and rng.random() < 0.15),
        wall_outline=bool(plain or hatch == "none" or rng.random() < 0.9),
        line_scale=g,
        wall_line_scale=jit(),
        thin_line_scale=jit(),
        hatch_line_scale=jit(),
        door="arc" if plain else str(rng.choice(DOOR_KINDS)),
        window=str(rng.choice(("two", "three"))) if plain else str(rng.choice(WINDOW_KINDS)),
        clutter=0.0 if plain and rng.random() < 0.5 else float(rng.uniform(0, 1)),
        damage=damage,
        damage_strength=float(rng.uniform(0.3, 1.0)),
        scan_degrade=float(rng.choice([0.0, rng.uniform(0.1, 1.0), 1.0], p=[0.15, 0.7, 0.15])),
        perspective=float(rng.uniform(0, 1)) if rng.random() < 0.25 else 0.0,
        curl=float(rng.uniform(0, 1)) if rng.random() < 0.2 else 0.0,
        wobble=float(rng.uniform(0, 1)) if rng.random() < 0.25 else 0.0,
        scan_dpi=dpi,
        jpeg_quality=q,
        augraphy=tuple(aug),
    )
    check_not_heldout(style)
    return style


# --------------------------------------------------------------------------------------------
# Guard against the held-out test style
# --------------------------------------------------------------------------------------------


class HeldOutStyleError(AssertionError):
    pass


def _heldout_pools() -> dict:
    """Held-out options as flagged in the protected generator module (read only)."""
    from kelpod.evaluation.synth import styles as S

    return {
        "fonts": {o.spec for o in S.FONTS.values() if o.heldout and o.spec},
        "font_names": {n for n, o in S.FONTS.items() if o.heldout},
        "hatches": [(o.pattern, o.spacing_mm) for o in S.HATCHES.values() if o.heldout],
        "line_scales": [o.scale for o in S.LINE_WEIGHTS.values() if o.heldout],
        "damages": {n for n, o in S.DAMAGES.items() if o.heldout},
    }


def check_not_heldout(style: TrainStyle) -> None:
    """Raise :class:`HeldOutStyleError` if any part of ``style`` is a held-out option or too close
    to one (see the module docstring for the margins)."""
    H = _heldout_pools()
    spec = TRAIN_FONTS.get(style.font, "<unknown>")
    if style.font not in TRAIN_FONTS:
        raise HeldOutStyleError(f"font {style.font!r} is not in the train pool")
    if style.font in H["font_names"] or (spec is not None and spec in H["fonts"]):
        raise HeldOutStyleError(f"font {style.font!r} is held out")
    if spec is not None:
        low = spec.lower().replace("-", "").replace("_", "")
        if any(t in low for t in FORBIDDEN_FONT_TOKENS):
            raise HeldOutStyleError(f"font {spec} belongs to a held-out family")
        if spec.startswith("hershey:") and (int(spec.split(":")[1]) & 15) in FORBIDDEN_HERSHEY:
            raise HeldOutStyleError(f"Hershey face {spec} is held out (or of its family)")
    if style.hatch in FORBIDDEN_HATCH_KINDS or style.hatch not in HATCH_KINDS:
        raise HeldOutStyleError(f"hatch {style.hatch!r} is not allowed")
    for pattern, spacing in H["hatches"]:
        if pattern in ("solid", "cross") and style.hatch in ("solid", "cross"):
            raise HeldOutStyleError(f"hatch {style.hatch!r} is held out")
        # a 45° hatch rotated by 90° in training is a 135° hatch: keep spacings apart
        if pattern in ("diag135", "diag45") and style.hatch in ("diag45", "double45", "dashed45"):
            if spacing > 0 and abs(style.hatch_spacing_mm - spacing) < 0.25:
                raise HeldOutStyleError(
                    f"hatch spacing {style.hatch_spacing_mm:.2f} mm too close to held-out "
                    f"{pattern} {spacing} mm")
    for s in (style.line_scale, style.wall_line_scale, style.thin_line_scale,
              style.hatch_line_scale):
        if not LINE_SCALE[0] - 1e-9 <= s <= LINE_SCALE[1] + 1e-9:
            raise HeldOutStyleError(f"line scale {s:.3f} outside the train range {LINE_SCALE}")
        for h in H["line_scales"]:
            if abs(s - h) < LINE_SCALE_MARGIN:
                raise HeldOutStyleError(f"line scale {s:.3f} too close to held-out {h}")
    if style.damage in H["damages"] or style.damage in FORBIDDEN_DAMAGE:
        raise HeldOutStyleError(f"damage profile {style.damage!r} is held out")
    if style.damage not in DAMAGE_KINDS:
        raise HeldOutStyleError(f"damage profile {style.damage!r} is not in the train pool")
    for a in style.augraphy:
        if a in AUGRAPHY_FORBIDDEN or a not in AUGRAPHY_ALLOWED:
            raise HeldOutStyleError(f"Augraphy effect {a!r} is not allowed")
