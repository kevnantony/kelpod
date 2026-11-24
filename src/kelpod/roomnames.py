"""Room-name helpers shared by the ocr and geometry steps (no dependency on either)."""

from __future__ import annotations

import re
import unicodedata

EXTERIOR_STEMS = ("garten", "vorhof", "hof", "altan", "terrasse", "balkon", "loggia", "vorplatz")
"""``Wintergarten`` is excluded: an enclosed glazed room, treated as a room."""


def fold(t: str) -> str:
    t = t.strip().lower().replace("ß", "ss")
    t = unicodedata.normalize("NFKD", t)
    return re.sub(r"[^a-z]", "", "".join(c for c in t if not unicodedata.combining(c)))


def is_exterior(name: str | None) -> bool:
    """True for garden, courtyard (Vorhof, Hof, Innenhof), Altan, terrace, balcony, loggia,
    forecourt. ``Wintergarten`` and ``Lichthof`` (a light well inside the building) are not."""
    if not name:
        return False
    for word in re.split(r"[\s/,-]+", name):
        f = fold(word)
        if not f or f in ("wintergarten", "lichthof"):
            continue
        if f in ("hof", "innenhof") or any(f.startswith(s) for s in EXTERIOR_STEMS if s != "hof"):
            return True
    return False
