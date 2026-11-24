"""Room names on historic plans: hyphenated line breaks, old spellings, exterior spaces.

Old plans break names over lines (``SPEISE-`` / ``S.``, ``ANRICHTE-`` / ``Z.``) and use
abbreviations (``Schlafz.``, ``Frühstz.``, ``Boud.``). Names are returned as written (title-cased,
fragments joined), abbreviations are NOT expanded. Exterior spaces (garden, courtyard, balcony,
loggia ...) are recognised so that they are not turned into rooms.
"""

from __future__ import annotations

import difflib
import re
import unicodedata

from kelpod.contracts import TextItem
from kelpod.roomnames import fold as _fold
from kelpod.roomnames import is_exterior  # noqa: F401  (re-exported)

# stems of historic room names (folded: lower case, umlauts and ß removed)
HISTORIC_STEMS = (
    "speise", "anricht", "lichth", "silberk", "ankleide", "schlaf", "boud", "pfoert", "pfort",
    "fest", "empfang", "raume", "altan", "garten", "vorhof", "hof", "wintergarten", "kammer",
    "stube", "kuche", "flur", "stall", "treppe", "fremden", "fremdenz", "fruhst", "schrank", "bad",
    "wohn", "salon", "diele", "vorplatz", "veranda", "terrasse", "balkon", "loggia", "heuboden",
    "tenne", "vestib", "halle", "zimmer", "gang", "abtritt", "waschk", "keller", "dach",
)
HISTORIC_WORDS = (
    "Schlafz.", "Frühstz.", "Fremdenz.", "Schrankz.", "Ankleidez.", "Speisez.", "Wohnz.",
    "Anrichtez.", "Altan", "Boudoir", "Wintergarten", "Vorhof", "Garten", "Treppe",
)
"""Abbreviations as written on the plans; used only to repair OCR misreadings (fuzzy match)."""


def pretty(text: str) -> str:
    """``SPEISE-S.`` -> ``Speise-S.``; ``ANKLEIDE Z.`` -> ``Ankleide Z.``: title case per word,
    short abbreviation tails keep their capital."""
    out = []
    for w in text.split():
        parts = w.split("-")
        out.append("-".join(p[:1].upper() + p[1:].lower() if p else p for p in parts))
    return " ".join(out)


def match_historic_name(text: str) -> str | None:
    """The name as written if ``text`` starts with a known historic room-name stem, else None."""
    t = text.strip()
    if not 3 <= len(t) <= 40 or sum(c.isdigit() for c in t) > 1:
        return None
    f = _fold(t)
    if len(f) < 3:
        return None
    if any(f.startswith(s) for s in HISTORIC_STEMS):
        return pretty(t)
    if len(f) >= 5:  # OCR noise on small scans: Schlatr. / Framdenz. -> closest known word
        best = difflib.get_close_matches(f, [_fold(w) for w in HISTORIC_WORDS if _fold(w)[0] == f[0]], 1, 0.68)
        if best:
            return next(w for w in HISTORIC_WORDS if _fold(w) == best[0])
    return None


def join_hyphenated(items: list[TextItem], *, max_gap_h: float = 2.2) -> list[TextItem]:
    """Join a line ending in a hyphen with the line right below it (same centre column).

    ``SPEISE-`` + ``S.`` -> ``SPEISE-S.`` (short abbreviation tail: hyphen kept), ``FEST-`` +
    ``RÄUME`` -> ``FESTRÄUME`` (full word: hyphen dropped). Only upright text is joined."""
    used: set[int] = set()
    out: list[TextItem] = []
    for i, a in enumerate(items):
        if i in used:
            continue
        txt = a.text.strip()
        if not re.search(r"[-–—]$", txt) or a.angle_deg % 180 != 0:
            out.append(a)
            continue
        ah = max(p[1] for p in a.quad) - min(p[1] for p in a.quad)
        ax = sum(p[0] for p in a.quad) / 4
        ay = max(p[1] for p in a.quad)
        aw = max(p[0] for p in a.quad) - min(p[0] for p in a.quad)
        best, best_d = None, None
        for j, b in enumerate(items):
            if j == i or j in used or b.angle_deg % 180 != 0:
                continue
            bx = sum(p[0] for p in b.quad) / 4
            by = min(p[1] for p in b.quad)
            gap = by - ay
            if -0.3 * ah <= gap <= max_gap_h * ah and abs(bx - ax) <= 0.8 * max(aw, 1.5 * ah):
                if best_d is None or gap < best_d:
                    best, best_d = j, gap
        if best is None:
            out.append(a)
            continue
        b = items[best]
        used.add(best)
        tail = b.text.strip()
        head = re.sub(r"[-–—]$", "", txt)
        joined = head + ("-" if len(re.sub(r"\W", "", tail)) <= 2 else "") + tail
        xs = [p[0] for p in a.quad + b.quad]
        ys = [p[1] for p in a.quad + b.quad]
        quad = [(min(xs), min(ys)), (max(xs), min(ys)), (max(xs), max(ys)), (min(xs), max(ys))]
        out.append(
            a.model_copy(
                update={"text": joined, "quad": quad, "confidence": min(a.confidence, b.confidence)}
            )
        )
    return out
