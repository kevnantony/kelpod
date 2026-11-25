"""Training-only style randomisation, version 2 (experiment ``learned-v2``).

Builds on :mod:`kelpod_train.style_v1` (same guard against the held-out test style of
``kelpod.evaluation.synth.styles``, which is only read) and adds *new train options*:

* fonts: open-licensed (SIL OFL) Google Fonts, normal-width sans, serif and hand-lettering faces
  (downloaded to ``~/kelpod-data/fonts-v2``, see ``fetch_fonts.sh`` and ``CREDITS.md``). No
  monospace, no narrow/condensed, no geometric "gothic" faces (look-alikes of the held-out URW
  Gothic / Nimbus Sans Narrow / Nimbus Mono);
* hatching: further single-direction angles (15, 75, 90 degrees). Never 135 degrees, cross or solid;
  training rotates crops by 90 degrees, so 15/75 become 105/165, and 90 becomes 0;
* damage: ``aged_paper`` (uniform grey paper, dark ink: not mottled, not tinted, not pale ink),
  ``defocus`` (out-of-focus scan) and ``scan_bands`` (faint scanner brightness bands). Never
  ``diazo``, ``photocopy``, ``faded`` or their Augraphy look-alikes.

**Dev proxy split.** To measure generalisation to *unseen* options without touching the test style,
some of the new options are reserved (``DEV_*``): they are never sampled for training and only used
by :func:`sample_dev_style` to render the dev-proxy pages (``kelpod_train.build dev_proxy``).
Baseline-v1 has never seen any of the new options either, so baseline-v1 and learned-v2 are
compared on equal footing there.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kelpod_train import style_v1 as V1
from kelpod_train.style_v1 import HeldOutStyleError, TrainStyle

FONT_DIR = Path.home() / "kelpod-data" / "fonts-v2"

NEW_TRAIN_FONTS = {
    "lato": "Lato-Regular.ttf", "barlow": "Barlow-Regular.ttf", "pt_serif": "PT-Serif-Web-Regular.ttf",
    "cardo": "Cardo-Regular.ttf", "kalam": "Kalam-Regular.ttf", "patrick_hand": "PatrickHand-Regular.ttf",
    "shadows_light": "ShadowsIntoLight.ttf", "handlee": "Handlee-Regular.ttf",
    "open_sans": "OpenSans.ttf", "lora": "Lora.ttf", "caveat": "Caveat.ttf",
    "plex_serif": "IBMPlexSerif-Regular.ttf", "covered_grace": "CoveredByYourGrace.ttf",
    "gaegu": "Gaegu-Regular.ttf", "rubik": "Rubik.ttf", "work_sans": "WorkSans.ttf",
    "cabin": "Cabin.ttf", "source_sans3": "SourceSans3.ttf",
}
DEV_FONTS = {
    "pt_sans": "PT-Sans-Web-Regular.ttf", "architects_daughter": "ArchitectsDaughter-Regular.ttf",
    "gloria": "GloriaHallelujah.ttf", "reenie": "ReenieBeanie.ttf", "karla": "Karla.ttf",
    "crimson": "CrimsonPro.ttf", "dm_sans": "DMSans.ttf", "baskerville": "LibreBaskerville.ttf",
}
NEW_TRAIN_HATCH = ("diag15", "diag75", "vertical")
DEV_HATCH = ("diag20", "diag70", "diag45_sparse")  # 20/70 deg; 45 deg at 3.3-4.5 mm
NEW_TRAIN_DAMAGE = ("aged_paper", "defocus", "scan_bands")
DEV_DAMAGE = ("uneven_light", "sensor_noise", "halftone")

HATCH_ANGLES = {"diag15": 15.0, "diag75": 75.0, "vertical": 90.0, "diag20": 20.0, "diag70": 70.0,
                "diag45_sparse": 45.0}
DEV_SPARSE_MM = (3.3, 4.5)


def font_table() -> dict[str, str | None]:
    """name -> spec for every font the v2 renderer can draw (train + dev)."""
    t: dict[str, str | None] = dict(V1.TRAIN_FONTS)
    for d in (NEW_TRAIN_FONTS, DEV_FONTS):
        t.update({k: str(FONT_DIR / v) for k, v in d.items()})
    return t


@dataclass(frozen=True)
class TrainStyleV2(TrainStyle):
    font_spec: str | None = None
    """Resolved font file for fonts that are not in ``style_v1.TRAIN_FONTS``."""
    dev: bool = False
    """True only for the dev-proxy pages (uses DEV_* options; never for training)."""

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["version"] = 2
        return d


def _finish(ts: TrainStyle, rng: np.random.Generator, *, dev: bool) -> TrainStyleV2:
    d = dataclasses.asdict(ts)
    d["augraphy"] = tuple(ts.augraphy)
    ft = font_table()
    d["font_spec"] = ft.get(ts.font) if ts.font not in V1.TRAIN_FONTS else None
    st = TrainStyleV2(**d, dev=dev)
    return st


def sample_train_style_v2(rng: np.random.Generator) -> TrainStyleV2:
    """One randomised train style: v1's sampler, then new options swapped in with some probability.
    Never uses a DEV_* option."""
    ts = V1.sample_train_style(rng, plain_p=0.10)
    ch: dict = {}
    if ts.font != "default" and rng.random() < 0.45:
        ch["font"] = str(rng.choice(sorted(NEW_TRAIN_FONTS)))
    if ts.hatch not in ("none", "stipple") and ts.hatch_spacing_mm > 0 and rng.random() < 0.18:
        ch["hatch"] = str(rng.choice(NEW_TRAIN_HATCH))
    if rng.random() < 0.28:
        ch["damage"] = str(rng.choice(NEW_TRAIN_DAMAGE))
        ch["damage_strength"] = float(rng.uniform(0.3, 1.0))
    # v1 already has clutter / geometry / dpi / JPEG; widen the scan resolution a little
    ts = dataclasses.replace(ts, **ch)
    st = _finish(ts, rng, dev=False)
    check_not_heldout_v2(st)
    return st


def sample_dev_style(rng: np.random.Generator) -> TrainStyleV2:
    """Dev-proxy style: font, hatching and damage are all drawn from the options that no training
    run sees (and baseline-v1 has never seen); the other axes are as in training."""
    ts = V1.sample_train_style(rng, plain_p=0.0)
    hatch = str(rng.choice(DEV_HATCH))
    ts = dataclasses.replace(
        ts, font=str(rng.choice(sorted(DEV_FONTS))), hatch=hatch,
        hatch_spacing_mm=float(rng.uniform(*DEV_SPARSE_MM)) if hatch == "diag45_sparse"
        else ts.hatch_spacing_mm,
        damage=str(rng.choice(DEV_DAMAGE)), damage_strength=float(rng.uniform(0.3, 1.0)),
        wall_outline=True)
    st = _finish(ts, rng, dev=True)
    check_not_heldout_v2(st)
    return st


def check_not_heldout_v2(st: TrainStyleV2) -> None:
    """Guard: the v1 checks on everything v1 knows, explicit checks on the new options."""
    H = V1._heldout_pools()
    new_fonts = {**NEW_TRAIN_FONTS, **DEV_FONTS}
    if st.font in new_fonts:
        low = new_fonts[st.font].lower().replace("-", "").replace("_", "")
        if any(t in low for t in V1.FORBIDDEN_FONT_TOKENS) or st.font in H["font_names"]:
            raise HeldOutStyleError(f"font {st.font} is a held-out family")
        if st.font_spec is None or Path(st.font_spec).name != new_fonts[st.font]:
            raise HeldOutStyleError(f"font {st.font}: spec mismatch")
        if st.font_spec in H["fonts"]:
            raise HeldOutStyleError(f"font {st.font} is held out")
    elif st.font not in V1.TRAIN_FONTS:
        raise HeldOutStyleError(f"unknown font {st.font}")
    uses_dev = [st.font in DEV_FONTS, st.hatch in DEV_HATCH, st.damage in DEV_DAMAGE]
    if (not st.dev and any(uses_dev)) or (st.dev and not all(uses_dev)):
        raise HeldOutStyleError("dev options only in dev styles, and dev styles use all three")
    new_h = set(NEW_TRAIN_HATCH) | set(DEV_HATCH)
    if st.hatch in new_h:
        ang = HATCH_ANGLES[st.hatch]
        # after the 90-degree crop rotations the angle is also (ang + 90) mod 180; keep >= 15 deg
        # away from the held-out 135 degrees (dense, 1.0 mm), and never cross / solid
        for a in ((ang, (ang + 90.0) % 180.0) if st.hatch != "diag45_sparse" else ()):
            if abs(a - 135.0) < 15.0:
                raise HeldOutStyleError(f"hatch angle {ang} (or its rotation) near 135 degrees")
        if st.hatch == "diag45_sparse" and st.hatch_spacing_mm < 3.0:
            raise HeldOutStyleError("sparse hatch spacing too dense")
    if st.damage in set(NEW_TRAIN_DAMAGE) | set(DEV_DAMAGE):
        if st.damage in V1.FORBIDDEN_DAMAGE or st.damage in H["damages"]:
            raise HeldOutStyleError(f"damage {st.damage} is held out")
    # everything else: run the v1 guard with the new options replaced by v1 stand-ins
    new_h_all, new_d_all = set(NEW_TRAIN_HATCH) | set(DEV_HATCH), set(NEW_TRAIN_DAMAGE) | set(DEV_DAMAGE)
    stand = dataclasses.replace(
        TrainStyle(**{f.name: getattr(st, f.name) for f in dataclasses.fields(TrainStyle)}),
        font="default" if st.font in new_fonts else st.font,
        hatch="diag45" if st.hatch in new_h_all else st.hatch,
        hatch_spacing_mm=2.0 if st.hatch in new_h_all else st.hatch_spacing_mm,
        damage="scan" if st.damage in new_d_all else st.damage,
    )
    V1.check_not_heldout(stand)


def sample_dev_hard_style(rng: np.random.Generator) -> TrainStyleV2:
    """Harder dev-proxy style: unseen font + hatch + damage together (as ``sample_dev_style``) and a
    stronger scan: full scan degradation, 150-220 dpi resampling, JPEG quality 20-55."""
    ts = sample_dev_style(rng)
    ts = dataclasses.replace(
        ts, scan_degrade=1.0, scan_dpi=float(rng.uniform(150, 220)),
        jpeg_quality=int(rng.integers(20, 56)), damage_strength=float(rng.uniform(0.6, 1.0)))
    st = dataclasses.replace(ts, dev=True)
    check_not_heldout_v2(st)
    return st
