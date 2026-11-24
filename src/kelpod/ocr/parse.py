"""Pure parsers for the text found on German floor plans.

Nothing here touches images, so every rule can be tested on strings alone.

* :func:`parse_number` reads German numbers (``3,51``, ``0,885``, ``1,01⁵``) and fixes common OCR
  confusions (``O`` -> ``0``, ``l`` -> ``1``).
* :func:`parse_length_m` turns a dimension string into metres, using the German convention that
  lengths under 1 m may be written in centimetres (``88,5``, ``76``, ``88⁵``).
* :func:`parse_sill_height` reads ``BRH 0,90`` / ``BRH 87,5`` / ``Brh.=90``.
* :func:`parse_area` reads ``14,52 m²`` / ``14,52 qm`` / ``14,52 m2``.
* :func:`match_room_name` maps an (umlaut-damaged) word to a canonical room name.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass

_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")

# Characters OCR engines often return instead of digits inside a number.
_DIGIT_FIXES = str.maketrans(
    {"O": "0", "o": "0", "D": "0", "Q": "0", "C": "0",
     "l": "1", "I": "1", "|": "1", "i": "1", "!": "1",
     "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2"}
)

_NUMBER_CHARS = r"0-9OoDQClI|i!SsBZz⁰¹²³⁴⁵⁶⁷⁸⁹"
_NUMBER_RE = re.compile(rf"^[{_NUMBER_CHARS}]+(?:\s?[,.]\s?[{_NUMBER_CHARS}]+)?$")

MIN_LENGTH_M = 0.02
MAX_LENGTH_M = 60.0


@dataclass(frozen=True)
class ParsedNumber:
    """A number read from text, before any unit decision."""

    value: float
    integer_digits: int
    decimal_digits: int
    """Digits after the comma, without a superscript digit."""
    has_comma: bool
    superscript: bool
    """True if a trailing superscript digit (``1,01⁵``) was present."""
    fixed: bool
    """True if letters had to be read as digits (lower confidence)."""


def parse_number(text: str) -> ParsedNumber | None:
    """Parse one German number. Returns ``None`` if ``text`` is not a plain number.

    ``1,01⁵`` -> 1.015, ``88⁵`` -> 88.5, ``3,51`` -> 3.51, ``3.51`` -> 3.51 (dot accepted),
    ``O,885`` -> 0.885 (OCR fix).
    """
    s = text.strip().strip(".:;'\"`´-–_ ")
    s = re.sub(r"\s*([,.])\s*", r"\1", s)
    if not s or not _NUMBER_RE.match(s):
        return None
    # At least one real digit, otherwise words like "Bad" or "IS" would become numbers.
    if not re.search(r"[0-9⁰¹²³⁴⁵⁶⁷⁸⁹]", s):
        return None
    letters = re.findall(r"[A-Za-z|!]", s)
    if len(letters) > max(1, len(s) // 3):
        return None
    superscript = False
    sup = re.search(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+$", s)
    sup_digits = ""
    if sup:
        superscript = True
        sup_digits = sup.group(0).translate(_SUPERSCRIPT)
        s = s[: sup.start()]
    if re.search(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]", s):
        return None
    fixed = bool(letters)
    s = s.translate(_DIGIT_FIXES)
    if not re.fullmatch(r"[0-9]+(?:[,.][0-9]+)?", s):
        return None
    has_comma = "," in s or "." in s
    if has_comma:
        int_part, dec_part = re.split(r"[,.]", s)
    else:
        int_part, dec_part = s, ""
        if len(int_part) >= 3 and int_part[0] == "0":  # "0365": comma lost -> 0,365
            int_part, dec_part, has_comma, fixed = "0", int_part[1:], True, True
    value = float(f"{int_part}.{dec_part}{sup_digits}" if (dec_part or sup_digits) else int_part)
    return ParsedNumber(
        value=value,
        integer_digits=len(int_part),
        decimal_digits=len(dec_part),
        has_comma=has_comma,
        superscript=superscript,
        fixed=fixed,
    )


def number_to_metres(n: ParsedNumber) -> float:
    """Apply the German metre/centimetre convention to a parsed number.

    * With a comma and two or three decimals (``3,51``, ``0,885``, ``11,24``): metres.
    * With a comma, one decimal and an integer part of two digits (``88,5``, ``36,5``): centimetres.
    * With a comma and one decimal otherwise (``2,5``): metres.
    * Superscript without comma (``88⁵``): centimetres with a half (0.885 m).
    * Plain integers (``76``, ``24``): centimetres. A three-digit integer ending in 5 (``885``)
      is read as a lost superscript, i.e. ``88⁵`` = 0.885 m.
    """
    if n.has_comma:
        if n.decimal_digits == 1 and n.integer_digits >= 2 and not n.superscript:
            return n.value / 100.0
        return n.value
    if n.superscript:
        return n.value / 100.0
    if n.integer_digits == 3 and int(n.value) % 10 == 5:
        return n.value / 1000.0
    return n.value / 100.0


def parse_length_m(text: str) -> float | None:
    """Dimension text -> metres, or ``None`` if it is not a plausible length."""
    n = parse_number(text)
    if n is None:
        return None
    v = number_to_metres(n)
    if not (MIN_LENGTH_M <= v <= MAX_LENGTH_M):
        return None
    return round(v, 4)


_BRH_RE = re.compile(
    rf"^\s*(?:[B8][RrP][HhN]|brh|Br[üu]h?)\.?\s*[=:\-]?\s*([{_NUMBER_CHARS},. ]+?)\s*(m|cm)?\s*$",
    re.IGNORECASE,
)


def is_brh_keyword(text: str) -> bool:
    """True for a lone ``BRH`` / ``Brh.`` token (number in a separate box)."""
    return bool(re.fullmatch(r"\s*(?:[B8][RrP][HhN]|brh|Br[üu]h?)\.?\s*[=:]?\s*", text, re.I))


def parse_sill_value(num: str, unit: str | None = None) -> float | None:
    """The number part of a sill height -> metres (``0,90`` m, ``87,5`` cm, ``90`` cm)."""
    n = parse_number(num)
    if n is None:
        return None
    if unit and unit.lower() == "cm":
        v = n.value / 100.0
    elif unit and unit.lower() == "m":
        v = n.value
    elif n.has_comma and n.integer_digits == 1:
        v = n.value  # 0,90 / 1,10
    else:
        v = n.value / 100.0  # 87,5 / 90 / 88⁵
    if not (0.0 <= v <= 3.0):
        return None
    return round(v, 4)


def parse_sill_height(text: str) -> float | None:
    """``BRH 0,90`` -> 0.90, ``BRH 87,5`` -> 0.875, ``Brh.=90`` -> 0.90. ``None`` otherwise."""
    m = _BRH_RE.match(text)
    if not m:
        return None
    return parse_sill_value(m.group(1), m.group(2))


_AREA_RE = re.compile(
    rf"(?:^|(?<=\s)|(?<=[:=]))(?P<num>[{_NUMBER_CHARS}]+(?:\s?[,.]\s?[{_NUMBER_CHARS}]+)?)"
    r"\s*(?P<unit>(?i:m\s?²|m\s?2|m\s?\^\s?2|qm|m[FfIl1|]))\.?\s*$",
)


def parse_area(text: str) -> tuple[float, str] | None:
    """Find an area at the end of ``text``: returns ``(m², text before the area)`` or ``None``.

    ``14,52 m²`` -> (14.52, ""), ``Wohnen 14,52 m2`` -> (14.52, "Wohnen"), ``7,9 qm`` -> (7.9, "").
    A lost comma (``2091m²``, three or more digits) means two decimals; ``mF`` / ``m1`` are
    accepted as a misread ``m²``.
    """
    m = _AREA_RE.search(text.strip())
    if not m:
        return None
    n = parse_number(m.group("num"))
    if n is None:
        return None
    value = n.value
    if not n.has_comma and n.integer_digits >= 3:
        value = n.value / 100.0  # comma lost: areas are written to the cm², "2091 m²" = 20,91
    if not (0.3 <= value <= 500.0):
        return None
    return value, text.strip()[: m.start()].strip(" :=-")


# Canonical room names (as written on stamps) and accepted variants. All lower case and without
# umlauts on the left side; see :func:`_fold`.
ROOM_NAMES: dict[str, str] = {
    "wohnen": "Wohnen", "wohnzimmer": "Wohnzimmer", "wohnraum": "Wohnraum",
    "essen": "Essen", "esszimmer": "Esszimmer", "wohnen/essen": "Wohnen/Essen",
    "kuche": "Küche", "kochen": "Kochen", "kochnische": "Kochnische",
    "bad": "Bad", "dusche": "Dusche", "du": "Du", "wc": "WC", "bad/wc": "Bad/WC",
    "flur": "Flur", "diele": "Diele", "gang": "Gang", "windfang": "Windfang", "vorraum": "Vorraum",
    "schlafen": "Schlafen", "schlafzimmer": "Schlafzimmer", "eltern": "Eltern",
    "kind": "Kind", "kinderzimmer": "Kinderzimmer", "kinder": "Kinder",
    "zimmer": "Zimmer", "zi": "Zi.", "gast": "Gast", "gastezimmer": "Gästezimmer",
    "arbeiten": "Arbeiten", "arbeitszimmer": "Arbeitszimmer", "buro": "Büro",
    "abst": "Abst.", "abstellraum": "Abstellraum", "abstell": "Abstell.", "ar": "AR",
    "speis": "Speis", "speisekammer": "Speisekammer", "vorrat": "Vorrat",
    "hauswirtschaft": "Hauswirtschaft", "hausw": "Hausw.", "hwr": "HWR", "hw": "HW",
    "garderobe": "Garderobe", "ankleide": "Ankleide",
    "treppe": "Treppe", "treppenhaus": "Treppenhaus", "trh": "Trh.",
    "loggia": "Loggia", "balkon": "Balkon", "terrasse": "Terrasse",
    "keller": "Keller", "heizung": "Heizung", "heizraum": "Heizraum", "technik": "Technik",
    "waschkuche": "Waschküche", "garage": "Garage", "werkstatt": "Werkstatt",
}


def _fold(text: str) -> str:
    """Lower case, umlauts and ß folded (``Küche``/``Kueche``/``Kuche`` -> ``kuche``)."""
    t = text.strip().lower()
    t = t.replace("ae", "a").replace("oe", "o").replace("ue", "u").replace("ß", "ss")
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z/ ]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def match_room_name(text: str, *, cutoff: float = 0.8) -> str | None:
    """Return the canonical room name for ``text`` or ``None``.

    Tolerates missing umlauts, ``ae``/``ue`` spellings, a trailing dot, case and small OCR errors
    (difflib ratio >= ``cutoff`` for words of five letters or more). A trailing room number
    (``Kind 1``, ``Zi. 2``) is kept.
    """
    raw = text.strip()
    if not raw or len(raw) > 40:
        return None
    m = re.match(r"^(.*?)[\s.]*(\d{1,2})$", raw)
    suffix = ""
    if m and m.group(1).strip():
        raw, suffix = m.group(1), " " + m.group(2)
    key = _fold(raw)
    if not key:
        return None
    if key in ROOM_NAMES:
        return ROOM_NAMES[key] + suffix
    # Fuzzy only for longer words: short ones ("Bad", "WC", "Haus") would match too much.
    if len(key) >= 5:
        best = difflib.get_close_matches(key, [k for k in ROOM_NAMES if len(k) >= 5], 1, cutoff)
        if best:
            return ROOM_NAMES[best[0]] + suffix
    return None


_DOOR_LABEL_RE = re.compile(r"^\s*[0-9OolI,.⁰-⁹¹²³]+\s*/\s*[0-9OolI,.⁰-⁹¹²³]+\s*$")


def is_opening_label(text: str) -> bool:
    """``88,5/2,01``-style width/height labels of doors and windows (not dimensions)."""
    return bool(_DOOR_LABEL_RE.match(text))


def split_merged_numbers(text: str) -> list[str] | None:
    """Split two dimension values that OCR read as one string, e.g. ``0,6351,15`` ->
    ``["0,635", "1,15"]`` (neighbouring labels on a crowded chain). The second value is assumed to
    have a one-digit integer part (lengths under 10 m), else two digits. ``None`` if no split
    gives two plausible lengths."""
    s = re.sub(r"\s+", "", text)
    seps = [i for i, c in enumerate(s) if c in ",."]
    if len(seps) != 2:
        return None
    j = seps[1]
    for k in (1, 2):
        a, b = s[: j - k], s[j - k :]
        if "," in a or "." in a:
            if parse_length_m(a) is not None and parse_length_m(b) is not None:
                if len(a) - seps[0] - 1 >= 2:  # first value keeps 2-3 decimals
                    return [a, b]
    return None


_TOKEN_RE = re.compile(rf"[{_NUMBER_CHARS},.]+")


def number_tokens(text: str) -> list[tuple[int, int, str]]:
    """Plausible lengths inside a string with junk around them, as ``(start, end, token)``
    character spans: ``L 88,5 L`` -> ``[(2, 6, "88,5")]``, ``1,26」76」`` -> two tokens.
    Remnants of tick marks and extension lines next to a label are read as ``L``, ``[``, ``」``.
    Merged pairs such as ``0,6351,15`` are split with :func:`split_merged_numbers`."""
    out: list[tuple[int, int, str]] = []
    # Words or a scale ("Grundriss EG M 1:100") are not junk around a label.
    if len(re.findall(r"[A-Za-zÄÖÜäöüß]", text)) > 3 or re.search(r"\d\s*:\s*\d", text):
        return out
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0).strip(",.")
        start = m.start() + m.group(0).index(tok) if tok else m.start()
        if parse_length_m(tok) is not None and re.search(r"[0-9]", tok):
            out.append((start, start + len(tok), tok))
            continue
        parts = split_merged_numbers(tok)
        if parts:
            k = start
            for p in parts:
                out.append((k, k + len(p), p))
                k += len(p)
    return out
