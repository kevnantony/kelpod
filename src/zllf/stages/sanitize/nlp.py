"""NLP layer for the sanitizer: local spaCy models (NER) or a regex-only fallback.

spaCy models are never downloaded at runtime. They are looked up in
``$ZLLF_MODELS_DIR/spacy/<name>`` (default ``~/.cache/zllf/models``), as unpacked by
``scripts/fetch_spacy_models.py``. When a language's model is missing, a blank spaCy pipeline
(tokenizer only, ships with spaCy) is used instead: Presidio still runs all pattern recognisers,
but names/locations come only from the title/label heuristics in the rule packs.
``ner_mode`` records which mode was used per language.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

NerMode = Literal["spacy", "regex"]

SPACY_MODELS: dict[str, str] = {"en": "en_core_web_sm", "de": "de_core_news_sm"}

# spaCy NER labels -> Presidio entity types we act on; everything else is ignored.
NER_LABEL_MAP: dict[str, str] = {
    "PER": "PERSON", "PERSON": "PERSON",
    "LOC": "LOCATION", "GPE": "LOCATION", "FAC": "LOCATION",
}
NER_IGNORED_LABELS: tuple[str, ...] = (
    "ORG", "MISC", "NORP", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL",
    "CARDINAL", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "PRODUCT",
)

# Presidio logs analysed tokens at DEBUG level (document text): never let that through.
for _name in ("presidio-analyzer", "presidio_analyzer", "presidio-anonymizer"):
    logging.getLogger(_name).setLevel(logging.WARNING)


def models_dir() -> Path:
    """Root for locally stored auxiliary models (``$ZLLF_MODELS_DIR`` or ~/.cache/zllf/models)."""
    root = os.environ.get("ZLLF_MODELS_DIR")
    return Path(root).expanduser() if root else Path.home() / ".cache" / "zllf" / "models"


def find_spacy_model(name: str, root: Path | None = None) -> Path | None:
    """Directory containing ``config.cfg`` + ``meta.json`` for model ``name``, or None."""
    base = (root or models_dir()) / "spacy" / name
    if not base.is_dir():
        return None
    if (base / "config.cfg").exists():
        return base
    for cfg in sorted(base.glob("*/config.cfg")):  # wheel layout <name>/<name>-<version>/
        return cfg.parent
    return None


@dataclass
class NlpBundle:
    """Loaded spaCy pipelines per language plus the resulting NER mode."""

    nlp: dict[str, Any]
    modes: dict[str, NerMode]
    model_versions: dict[str, str] = field(default_factory=dict)

    @property
    def ner_mode(self) -> str:
        values = set(self.modes.values())
        if values == {"spacy"}:
            return "spacy"
        if values == {"regex"} or not values:
            return "regex"
        return "partial"


def load_nlp(
    languages: Sequence[str],
    root: Path | None = None,
    prefer: Literal["auto", "regex"] = "auto",
) -> NlpBundle:
    """Load local spaCy models where available, else blank tokenizer-only pipelines."""
    import spacy

    nlp: dict[str, Any] = {}
    modes: dict[str, NerMode] = {}
    versions: dict[str, str] = {}
    for lang in languages:
        path = find_spacy_model(SPACY_MODELS[lang], root) if lang in SPACY_MODELS else None
        if path is not None and prefer == "auto":
            try:
                pipe = spacy.load(path, exclude=["parser", "senter"])
                nlp[lang] = pipe
                modes[lang] = "spacy"
                versions[lang] = f"{pipe.meta.get('name', '')}-{pipe.meta.get('version', '')}"
                continue
            except Exception as exc:  # incompatible / corrupt model -> regex fallback
                logging.getLogger("zllf.sanitize").warning(
                    "spacy_model_load_failed lang=%s error=%s", lang, type(exc).__name__)
        nlp[lang] = spacy.blank(lang)
        modes[lang] = "regex"
    return NlpBundle(nlp=nlp, modes=modes, model_versions=versions)


def make_nlp_engine(bundle: NlpBundle) -> Any:
    """A Presidio ``SpacyNlpEngine`` around already loaded pipelines (never downloads)."""
    from presidio_analyzer.nlp_engine import NerModelConfiguration, NlpArtifacts, SpacyNlpEngine

    class LocalSpacyNlpEngine(SpacyNlpEngine):
        def __init__(self, pipelines: dict[str, Any]) -> None:
            super().__init__(
                models=[{"lang_code": k, "model_name": k} for k in pipelines],
                ner_model_configuration=NerModelConfiguration(
                    model_to_presidio_entity_mapping=dict(NER_LABEL_MAP),
                    labels_to_ignore=list(NER_IGNORED_LABELS),
                ),
            )
            self.nlp = dict(pipelines)  # type: ignore[assignment]

        def load(self) -> None:  # models are pre-loaded; never download
            return None

        def _doc_to_nlp_artifact(self, doc: Any, language: str) -> NlpArtifacts:
            art = super()._doc_to_nlp_artifact(doc, language)
            # blank pipelines have no lemmatizer: fall back to lower-cased token text so
            # Presidio's context enhancer still sees the surrounding words
            if any(not lemma for lemma in art.lemmas):
                lemmas = [lemma or tok.text.lower() for lemma, tok in zip(art.lemmas, doc,
                                                                         strict=True)]
                art.lemmas = lemmas
                art.keywords = NlpArtifacts.set_keywords(self, lemmas, language)
            return art

    return LocalSpacyNlpEngine(bundle.nlp)


# ---------------------------------------------------------------- language detection
_STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset({"the", "and", "of", "to", "in", "is", "was", "with", "for", "on", "at",
                     "by", "an", "be", "this", "that", "are", "from", "has", "no", "not"}),
    "de": frozenset({"der", "die", "das", "und", "ist", "mit", "zu", "den", "von", "im",
                     "nicht", "ein", "eine", "des", "auf", "für", "dem", "wurde", "sich", "bei"}),
}


class LanguageDetector:
    """Per-chunk language id restricted to the configured languages (lingua, offline).

    Falls back to a stopword vote if lingua is unavailable or unsure.
    """

    def __init__(self, languages: Sequence[str]) -> None:
        self.languages = list(languages)
        self.backend = "stopwords"
        self._detector: Any = None
        self._codes: dict[Any, str] = {}
        if len(self.languages) > 1:
            try:
                from lingua import IsoCode639_1, Language, LanguageDetectorBuilder

                langs = []
                for code in self.languages:
                    lang = Language.from_iso_code_639_1(getattr(IsoCode639_1, code.upper()))
                    langs.append(lang)
                    self._codes[lang] = code
                self._detector = LanguageDetectorBuilder.from_languages(*langs).build()
                self.backend = "lingua"
            except Exception:
                self._detector = None

    def detect(self, text: str) -> str:
        if len(self.languages) == 1:
            return self.languages[0]
        if self._detector is not None and text.strip():
            lang = self._detector.detect_language_of(text)
            if lang is not None and lang in self._codes:
                return self._codes[lang]
        return self._stopword_vote(text)

    def _stopword_vote(self, text: str) -> str:
        words = [w.strip(".,;:!?()").lower() for w in text.split()]
        best, best_n = self.languages[0], -1
        for code in self.languages:
            stop = _STOPWORDS.get(code, frozenset())
            n = sum(1 for w in words if w in stop)
            if n > best_n:
                best, best_n = code, n
        return best

