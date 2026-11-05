"""YAML regex rule packs (``configs/pii/rules_*.yaml``) and their checksum validators.

A rule pack is a list of :class:`RuleSpec`; :func:`match_rule` applies one rule to a text and
returns plain :class:`Match` tuples (no Presidio dependency, so it is easy to unit-test). The
Presidio adapter lives in :mod:`zllf.stages.sanitize.analyzer`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from zllf.core.config import CONFIG_DIR
from zllf.core.hashing import sha256_file

PII_CONFIG_DIR = CONFIG_DIR / "pii"
COMMON = "common"
_MACRO = re.compile(r"<<([A-Z_][A-Z0-9_]*)>>")
_FLAGS: dict[str, int] = {"IGNORECASE": re.IGNORECASE, "MULTILINE": re.MULTILINE,
                          "DOTALL": re.DOTALL}


# ---------------------------------------------------------------- validators
def _digits(text: str) -> str:
    return "".join(ch for ch in text if ch.isdigit())


def _alnum_upper(text: str) -> str:
    return "".join(ch for ch in text if ch.isalnum()).upper()


def validate_iban(text: str) -> bool:
    """ISO 13616 mod-97 check."""
    iban = _alnum_upper(text)
    if len(iban) < 15 or len(iban) > 34 or not iban[:2].isalpha() or not iban[2:4].isdigit():
        return False
    rearranged = iban[4:] + iban[:4]
    try:
        number = int("".join(str(int(ch, 36)) for ch in rearranged))
    except ValueError:
        return False
    return number % 97 == 1


def kvnr_check_digit(letter: str, digits8: str) -> int:
    """Check digit of a German KVNR (letter + 8 digits + check digit)."""
    pos = ord(letter.upper()) - ord("A") + 1
    seq = f"{pos:02d}{digits8}"
    total = 0
    for i, ch in enumerate(seq):
        prod = int(ch) * (1 if i % 2 == 0 else 2)
        total += prod // 10 + prod % 10
    return total % 10


def validate_kvnr(text: str) -> bool:
    raw = _alnum_upper(text)
    if len(raw) != 10 or not raw[0].isalpha() or not raw[1:].isdigit():
        return False
    return kvnr_check_digit(raw[0], raw[1:9]) == int(raw[9])


def de_tax_id_check_digit(first10: str) -> int:
    """ISO 7064 MOD 11,10 check digit used by the German Steuer-IdNr."""
    product = 10
    for ch in first10:
        total = (int(ch) + product) % 10
        if total == 0:
            total = 10
        product = (total * 2) % 11
    check = 11 - product
    return 0 if check == 10 else check


def validate_de_tax_id(text: str) -> bool:
    digits = _digits(text)
    if len(digits) != 11 or digits[0] == "0":
        return False
    return de_tax_id_check_digit(digits[:10]) == int(digits[10])


def validate_us_ssn(text: str) -> bool:
    digits = _digits(text)
    if len(digits) != 9:
        return False
    area, group, serial = digits[:3], digits[3:5], digits[5:]
    if area in ("000", "666") or area.startswith("9") or group == "00" or serial == "0000":
        return False
    return len(set(digits)) > 1


def validate_phone(text: str) -> bool:
    n = len(_digits(text))
    return 7 <= n <= 15


VALIDATORS: dict[str, Callable[[str], bool]] = {
    "iban": validate_iban,
    "kvnr": validate_kvnr,
    "de_tax_id": validate_de_tax_id,
    "us_ssn": validate_us_ssn,
    "phone": validate_phone,
}


# ---------------------------------------------------------------- specs
@dataclass(frozen=True)
class Match:
    """A raw rule hit: character span + score (never stored with its text)."""

    start: int
    end: int
    entity_type: str
    score: float
    rule: str


@dataclass
class RuleSpec:
    name: str
    entity_type: str
    patterns: list[re.Pattern[str]]
    score: float = 0.8
    context: list[str] = field(default_factory=list)
    context_boost: float = 0.35
    context_window: int = 40
    context_stop: str = "\n"
    requires_context: bool = False
    validator: str | None = None
    invalid_score: float = 0.0
    exclude: re.Pattern[str] | None = None
    language: str = COMMON


@dataclass
class RulePack:
    language: str
    rules: list[RuleSpec]
    source: str  # file name (or "config" for custom patterns)
    sha256: str | None = None


def expand_macros(pattern: str, macros: Mapping[str, str], depth: int = 0) -> str:
    """Replace ``<<NAME>>`` references (recursively, max depth 10)."""
    if depth > 10:
        raise ValueError("macro recursion too deep")

    def repl(m: re.Match[str]) -> str:
        key = m.group(1)
        if key not in macros:
            raise KeyError(f"unknown macro <<{key}>>")
        return f"(?:{macros[key]})"

    out = _MACRO.sub(repl, pattern)
    return expand_macros(out, macros, depth + 1) if _MACRO.search(out) else out


def _terms_pattern(terms: Iterable[str]) -> str:
    alts = sorted({t.strip() for t in terms if t.strip()}, key=lambda t: (-len(t), t))
    return r"(?<!\w)(?:" + "|".join(re.escape(t) for t in alts) + r")(?!\w)"


def compile_rule(raw: Mapping[str, Any], macros: Mapping[str, str], language: str) -> RuleSpec:
    """Build a :class:`RuleSpec` from one YAML mapping."""
    flags = 0
    for name in raw.get("flags", []) or []:
        flags |= _FLAGS[str(name).upper()]
    sources = [expand_macros(str(p), macros) for p in raw.get("patterns", []) or []]
    if raw.get("terms"):
        sources.append(_terms_pattern(raw["terms"]))
    if not sources:
        raise ValueError(f"rule {raw.get('name')!r} has neither patterns nor terms")
    validator = raw.get("validator")
    if validator is not None and validator not in VALIDATORS:
        raise ValueError(f"rule {raw.get('name')!r}: unknown validator {validator!r}")
    exclude = raw.get("exclude")
    return RuleSpec(
        name=str(raw["name"]),
        entity_type=str(raw["entity_type"]),
        patterns=[re.compile(s, flags) for s in sources],
        score=float(raw.get("score", 0.8)),
        context=[str(c).lower() for c in raw.get("context", []) or []],
        context_boost=float(raw.get("context_boost", 0.35)),
        context_window=int(raw.get("context_window", 40)),
        context_stop=str(raw.get("context_stop", "\n")),
        requires_context=bool(raw.get("requires_context", False)),
        validator=validator,
        invalid_score=float(raw.get("invalid_score", 0.0)),
        exclude=re.compile(expand_macros(str(exclude), macros)) if exclude else None,
        language=language,
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def load_rule_packs(
    languages: Iterable[str], rules_dir: Path = PII_CONFIG_DIR
) -> dict[str, RulePack]:
    """Load ``rules_common.yaml`` plus ``rules_<lang>.yaml`` for each language (if present).

    Macros of the common pack are visible to all language packs.
    """
    packs: dict[str, RulePack] = {}
    common_path = rules_dir / "rules_common.yaml"
    macros: dict[str, str] = {}
    files: list[tuple[str, Path]] = []
    if common_path.exists():
        files.append((COMMON, common_path))
    for lang in languages:
        path = rules_dir / f"rules_{lang}.yaml"
        if path.exists():
            files.append((lang, path))
    raws = [(lang, path, _read_yaml(path)) for lang, path in files]
    for _, _, data in raws:  # common first, so language packs may override nothing silently
        for key, value in (data.get("macros") or {}).items():
            macros.setdefault(str(key), str(value))
    for lang, path, data in raws:
        rules = [compile_rule(r, macros, lang) for r in data.get("recognizers") or []]
        packs[lang] = RulePack(language=lang, rules=rules, source=path.name,
                               sha256=sha256_file(path))
    return packs


def custom_pattern_pack(custom_patterns: Mapping[str, str], score: float = 0.85) -> RulePack:
    """``config.sanitize.custom_patterns`` (entity name -> regex) as a common rule pack."""
    rules = []
    for name, pattern in sorted(custom_patterns.items()):
        entity = re.sub(r"[^A-Z0-9_]", "_", name.upper())
        rules.append(RuleSpec(name=f"custom_{entity.lower()}", entity_type=entity,
                              patterns=[re.compile(pattern)], score=score))
    return RulePack(language=COMMON, rules=rules, source="config.sanitize.custom_patterns")


# ---------------------------------------------------------------- matching
def _context_found(rule: RuleSpec, text: str, start: int) -> bool:
    if not rule.context:
        return False
    window = text[max(0, start - rule.context_window):start]
    cut = max((window.rfind(ch) for ch in rule.context_stop), default=-1)
    window = window[cut + 1:].lower()
    return any(word in window for word in rule.context)


def match_rule(rule: RuleSpec, text: str) -> list[Match]:
    """All hits of one rule in ``text`` (span = named group ``value`` if present)."""
    out: list[Match] = []
    seen: set[tuple[int, int]] = set()
    for pattern in rule.patterns:
        has_group = "value" in pattern.groupindex
        for m in pattern.finditer(text):
            start, end = m.span("value") if has_group else m.span()
            if start < 0 or end <= start:
                continue
            # trim whitespace/trailing punctuation that belongs to the sentence
            while end > start and text[end - 1] in " \t\n,;:":
                end -= 1
            if (start, end) in seen:
                continue
            value = text[start:end]
            if rule.exclude is not None and rule.exclude.fullmatch(value):
                continue
            score = rule.score
            if rule.validator is not None and not VALIDATORS[rule.validator](value):
                score = rule.invalid_score
            has_context = _context_found(rule, text, start)
            if rule.requires_context and not has_context:
                continue
            if has_context and not rule.requires_context:
                score = min(1.0, score + rule.context_boost)
            if score <= 0:
                continue
            seen.add((start, end))
            out.append(Match(start, end, rule.entity_type, round(score, 4), rule.name))
    return out


def match_pack(pack: RulePack, text: str) -> list[Match]:
    hits: list[Match] = []
    for rule in pack.rules:
        hits.extend(match_rule(rule, text))
    return hits
