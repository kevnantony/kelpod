"""Presidio ``AnalyzerEngine`` assembled from local rule packs, built-ins and optional NER.

Per language the registry holds:

* one :class:`RuleRecognizer` per rule of ``rules_common.yaml``, ``rules_<lang>.yaml`` and
  ``config.sanitize.custom_patterns``;
* Presidio's checksum-based ``IbanRecognizer`` (offline). The e-mail/URL built-ins are not used
  (``tldextract`` may go online) and ``PhoneRecognizer`` is not used (it reads dates such as
  ``06.06.2024`` as phone numbers; the rule packs cover DE/intl/US formats);
* Presidio's ``SpacyRecognizer`` (PERSON, LOCATION) when a local spaCy model was found.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from zllf.stages.sanitize.nlp import LanguageDetector, NlpBundle, load_nlp, make_nlp_engine
from zllf.stages.sanitize.policy import Detection
from zllf.stages.sanitize.rules import (
    COMMON,
    PII_CONFIG_DIR,
    RulePack,
    RuleSpec,
    custom_pattern_pack,
    load_rule_packs,
    match_rule,
)


def _rule_recognizer_cls() -> type:
    from presidio_analyzer import LocalRecognizer, RecognizerResult

    class RuleRecognizer(LocalRecognizer):
        """Presidio adapter for one YAML :class:`RuleSpec` (context handled by the rule)."""

        def __init__(self, rule: RuleSpec, language: str) -> None:
            self.rule = rule
            self.label = f"{rule.name}@{language}"
            super().__init__(supported_entities=[rule.entity_type],
                             name=self.label,
                             supported_language=language, context=None)

        def load(self) -> None:
            return None

        def analyze(self, text: str, entities: list[str], nlp_artifacts: Any = None,
                    ) -> list[Any]:
            if entities and self.rule.entity_type not in entities:
                return []
            meta = {RecognizerResult.RECOGNIZER_NAME_KEY: self.label,
                    RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: self.id}
            return [
                RecognizerResult(
                    entity_type=m.entity_type, start=m.start, end=m.end, score=m.score,
                    recognition_metadata=dict(meta),
                )
                for m in match_rule(self.rule, text)
            ]

    return RuleRecognizer


class PiiAnalyzer:
    """Language-aware PII detector returning plain :class:`Detection` spans."""

    def __init__(
        self,
        languages: Sequence[str] = ("en", "de"),
        custom_patterns: Mapping[str, str] | None = None,
        rules_dir: Path = PII_CONFIG_DIR,
        models_root: Path | None = None,
        ner: Literal["auto", "regex"] = "auto",
        use_builtins: bool = True,
    ) -> None:
        from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
        from presidio_analyzer.predefined_recognizers import (
            IbanRecognizer,
            SpacyRecognizer,
        )

        self.languages = list(dict.fromkeys(languages))
        self.packs: dict[str, RulePack] = load_rule_packs(self.languages, rules_dir)
        self.custom_pack = custom_pattern_pack(custom_patterns or {})
        self.bundle: NlpBundle = load_nlp(self.languages, models_root, prefer=ner)
        self.language_detector = LanguageDetector(self.languages)

        rule_cls = _rule_recognizer_cls()
        recognizers: list[Any] = []
        for lang in self.languages:
            for pack in self._packs_for(lang):
                recognizers.extend(rule_cls(rule, lang) for rule in pack.rules)
            if use_builtins:
                recognizers.append(IbanRecognizer(supported_language=lang,
                                                  name=f"presidio_iban@{lang}"))
            if self.bundle.modes.get(lang) == "spacy":
                recognizers.append(SpacyRecognizer(supported_language=lang,
                                                   supported_entities=["PERSON", "LOCATION"],
                                                   name=f"spacy_ner@{lang}"))
        registry = RecognizerRegistry(recognizers=recognizers,
                                      supported_languages=self.languages)
        self.engine = AnalyzerEngine(registry=registry,
                                     nlp_engine=make_nlp_engine(self.bundle),
                                     supported_languages=self.languages)

    def _packs_for(self, lang: str) -> list[RulePack]:
        packs = [p for key, p in sorted(self.packs.items()) if key in (COMMON, lang)]
        if self.custom_pack.rules:
            packs.append(self.custom_pack)
        return packs

    @property
    def ner_mode(self) -> str:
        return self.bundle.ner_mode

    def entity_types(self) -> list[str]:
        types = {"IBAN_CODE", "PHONE_NUMBER", "PERSON", "LOCATION"}
        for pack in [*self.packs.values(), self.custom_pack]:
            types.update(r.entity_type for r in pack.rules)
        return sorted(types)

    def rule_sources(self) -> list[dict[str, str | None]]:
        return [{"language": p.language, "file": p.source, "sha256": p.sha256}
                for _, p in sorted(self.packs.items())]

    def detect_language(self, text: str) -> str:
        return self.language_detector.detect(text)

    def analyze(self, text: str, language: str | None = None) -> list[Detection]:
        """All candidate spans (no threshold applied), sorted deterministically."""
        if not text.strip():
            return []
        lang = language or self.detect_language(text)
        if lang not in self.languages:
            lang = self.languages[0]
        results = self.engine.analyze(text=text, language=lang, score_threshold=0.0)
        dets = {
            Detection(r.start, r.end, r.entity_type, round(float(r.score), 4),
                      str((r.recognition_metadata or {}).get("recognizer_name", "")))
            for r in results
            if r.end > r.start
        }
        return sorted(dets, key=lambda d: (d.start, d.end, d.entity_type, -d.score, d.source))
