"""NLP layer: model lookup, regex fallback, language detection, optional NER mode."""

from __future__ import annotations

from pathlib import Path

import pytest

from zllf.stages.sanitize.analyzer import PiiAnalyzer
from zllf.stages.sanitize.nlp import LanguageDetector, find_spacy_model, load_nlp, models_dir


def spacy_models_available() -> bool:
    return all(find_spacy_model(n, models_dir()) is not None
               for n in ("en_core_web_sm", "de_core_news_sm"))


def test_models_dir_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ZLLF_MODELS_DIR", str(tmp_path))
    assert models_dir() == tmp_path
    monkeypatch.delenv("ZLLF_MODELS_DIR")
    assert models_dir() == Path.home() / ".cache" / "zllf" / "models"


def test_find_spacy_model_layouts(tmp_path: Path) -> None:
    assert find_spacy_model("en_core_web_sm", tmp_path) is None
    nested = tmp_path / "spacy" / "en_core_web_sm" / "en_core_web_sm-3.8.0"
    nested.mkdir(parents=True)
    (nested / "config.cfg").write_text("")
    assert find_spacy_model("en_core_web_sm", tmp_path) == nested


def test_regex_fallback_without_models(empty_models_dir: Path) -> None:
    bundle = load_nlp(["en", "de"], empty_models_dir)
    assert bundle.modes == {"en": "regex", "de": "regex"}
    assert bundle.ner_mode == "regex"
    an = PiiAnalyzer(("en", "de"), models_root=empty_models_dir)
    assert an.ner_mode == "regex"
    found = {(d.entity_type, d.start) for d in an.analyze("Herr Kesselring kam.", "de")}
    assert ("PERSON", 5) in found


@pytest.mark.parametrize(("text", "lang"), [
    ("The patient was admitted with chest pain and shortness of breath.", "en"),
    ("Der Patient wurde mit Brustschmerzen und Atemnot aufgenommen.", "de"),
    ("Lager getauscht, Pumpe läuft wieder ohne Auffälligkeiten.", "de"),
])
def test_language_detection(text: str, lang: str) -> None:
    det = LanguageDetector(["en", "de"])
    assert det.backend == "lingua"
    assert det.detect(text) == lang


def test_language_detection_single_language() -> None:
    assert LanguageDetector(["de"]).detect("hello world") == "de"


@pytest.mark.skipif(not spacy_models_available(), reason="local spaCy models not installed")
def test_ner_mode_finds_bare_names() -> None:
    an = PiiAnalyzer(("en", "de"))
    assert an.ner_mode == "spacy"
    text = "Michael Fairbanks was admitted to the ward in Boston yesterday."
    types = {d.entity_type for d in an.analyze(text, "en") if d.score >= 0.5}
    assert "PERSON" in types
