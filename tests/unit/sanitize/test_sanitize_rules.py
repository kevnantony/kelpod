"""Recognisers: validators and positive/negative cases (EN + DE)."""

from __future__ import annotations

from pathlib import Path

import pytest

from zllf.stages.sanitize.analyzer import PiiAnalyzer
from zllf.stages.sanitize.rules import (
    custom_pattern_pack,
    expand_macros,
    kvnr_check_digit,
    load_rule_packs,
    match_pack,
    validate_de_tax_id,
    validate_iban,
    validate_kvnr,
    validate_us_ssn,
)


def _kvnr(letter: str, digits8: str) -> str:
    return f"{letter}{digits8}{kvnr_check_digit(letter, digits8)}"


VALID_KVNR = _kvnr("A", "12345678")
VALID_TAX_ID = "86095742719"  # published example Steuer-IdNr (valid check digit)
VALID_IBAN = "DE89 3704 0044 0532 0130 00"  # standard documentation example


# ---------------------------------------------------------------- validators
def test_validators() -> None:
    assert validate_iban(VALID_IBAN)
    assert validate_iban("GB82WEST12345698765432")
    assert not validate_iban("DE89 3704 0044 0532 0130 01")
    assert validate_kvnr(VALID_KVNR)
    assert not validate_kvnr(VALID_KVNR[:-1] + str((int(VALID_KVNR[-1]) + 1) % 10))
    assert validate_de_tax_id(VALID_TAX_ID)
    assert validate_de_tax_id("86 095 742 719")
    assert not validate_de_tax_id("86095742710")
    assert not validate_de_tax_id("06095742719")
    assert validate_us_ssn("123-45-6789")
    for bad in ("000-12-3456", "666-12-3456", "912-12-3456", "123-00-4567", "123-45-0000"):
        assert not validate_us_ssn(bad)


def test_macro_expansion_and_unknown_macro() -> None:
    assert expand_macros("a<<X>>b", {"X": "<<Y>>", "Y": "z"}) == "a(?:(?:z))b"
    with pytest.raises(KeyError):
        expand_macros("<<NOPE>>", {})


# ---------------------------------------------------------------- recogniser cases
def _types(analyzer: PiiAnalyzer, text: str, lang: str, min_score: float = 0.5,
           ) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for d in analyzer.analyze(text, lang):
        if d.score >= min_score:
            out.setdefault(d.entity_type, []).append(text[d.start:d.end])
    return out


POSITIVE = [
    # (lang, text, entity_type, expected substring)
    ("en", "Contact: anna.k@example-hospital.org today", "EMAIL_ADDRESS",
     "anna.k@example-hospital.org"),
    ("de", "Rückfragen an m.beispiel@klinikum-example.de", "EMAIL_ADDRESS",
     "m.beispiel@klinikum-example.de"),
    ("de", "Tel.: 030 12345678", "PHONE_NUMBER", "030 12345678"),
    ("de", "Bitte zurückrufen unter 0171-2345678, danke.", "PHONE_NUMBER", "0171-2345678"),
    ("de", "Erreichbar unter +49 (0) 89 123 4567.", "PHONE_NUMBER", "+49 (0) 89 123 4567"),
    ("en", "Call +44 20 7946 0958 after 5pm.", "PHONE_NUMBER", "+44 20 7946 0958"),
    ("en", "Phone (617) 555-0142.", "PHONE_NUMBER", "(617) 555-0142"),
    ("de", f"IBAN: {VALID_IBAN}", "IBAN_CODE", VALID_IBAN),
    ("de", f"Versichertennummer {VALID_KVNR}", "DE_KVNR", VALID_KVNR),
    ("de", f"Steuer-ID: {VALID_TAX_ID}", "DE_TAX_ID", VALID_TAX_ID),
    ("en", "SSN 123-45-6789 on file", "US_SSN", "123-45-6789"),
    ("en", "Patient: John Whitaker\nDOB: 03/12/1975", "DATE_OF_BIRTH", "03/12/1975"),
    ("en", "She was born on March 12, 1975 in a small town.", "DATE_OF_BIRTH",
     "March 12, 1975"),
    ("de", "Max Beispiel, geb. 12.03.1975", "DATE_OF_BIRTH", "12.03.1975"),
    ("de", "Geburtsdatum: 12. März 1975", "DATE_OF_BIRTH", "12. März 1975"),
    ("en", "MRN: 48213377", "PATIENT_ID", "48213377"),
    ("de", "Fallnummer: 2024/118274", "PATIENT_ID", "2024/118274"),
    ("en", "see PID-2023-004512", "PATIENT_ID", "PID-2023-004512"),
    ("en", "Mrs. Holloway reports pain.", "PERSON", "Holloway"),
    ("en", "Seen by Dr. Grace Pembroke today.", "PERSON", "Grace Pembroke"),
    ("en", "Technician: Owen Marlow\nPhone: 555", "PERSON", "Owen Marlow"),
    ("en", "Inspected by Ruby Sutton on Monday.", "PERSON", "Ruby Sutton"),
    ("en", "Best regards,\nSamuel Ridley", "PERSON", "Samuel Ridley"),
    ("de", "Herr Kesselring klagt über Schmerzen.", "PERSON", "Kesselring"),
    ("de", "Frau Dr. med. Birgit Obermaier untersuchte.", "PERSON", "Birgit Obermaier"),
    ("de", "Patientin: Lena Tiedemann", "PERSON", "Lena Tiedemann"),
    ("de", "Mit freundlichen Grüßen\nDr. Volker Rautenberg", "PERSON", "Volker Rautenberg"),
    ("de", "gez. Ilse Vogelsang, Oberärztin", "PERSON", "Ilse Vogelsang"),
    ("de", "Anschrift: Lindenstraße 14, 10969 Berlin.", "LOCATION", "Lindenstraße 14"),
    ("de", "Anschrift: Lindenstraße 14, 10969 Berlin.", "LOCATION", "10969 Berlin"),
    ("de", "Die Patientin ist wohnhaft in Lüneburg.", "LOCATION", "Lüneburg"),
    ("en", "Address: 221 Maple Street, Springfield, IL 62704.", "LOCATION", "221 Maple Street"),
    ("en", "Address: 221 Maple Street, Springfield, IL 62704.", "LOCATION",
     "Springfield, IL 62704"),
    ("en", "SECRET//NOFORN", "CLASSIFICATION_MARKING", "SECRET//NOFORN"),
    ("en", "CONFIDENTIAL - do not distribute", "CLASSIFICATION_MARKING", "CONFIDENTIAL"),
    ("de", "VS-NUR FÜR DEN DIENSTGEBRAUCH", "CLASSIFICATION_MARKING",
     "VS-NUR FÜR DEN DIENSTGEBRAUCH"),
    ("de", "Vertraulich\nInhalt folgt", "CLASSIFICATION_MARKING", "Vertraulich"),
]


@pytest.mark.parametrize(("lang", "text", "etype", "expected"), POSITIVE)
def test_positive(analyzer: PiiAnalyzer, lang: str, text: str, etype: str,
                  expected: str) -> None:
    found = _types(analyzer, text, lang)
    assert expected in found.get(etype, []), found


NEGATIVE = [
    # (lang, text, entity_type that must NOT be found)
    ("en", "Admitted on 03/12/2024 with chest pain.", "DATE_OF_BIRTH"),
    ("de", "Aufnahme am 12.03.2024, Entlassung am 20.03.2024.", "DATE_OF_BIRTH"),
    ("de", "Patientin (* 01.02.1960) wurde am 05.06.2024 aufgenommen.", "PHONE_NUMBER"),
    ("de", "Kreatinin 1,2 mg/dl, CRP 0,5 mg/l, Dosis 0,25 mg.", "PHONE_NUMBER"),
    ("en", "Order WO-48213 replaced part 4711-0815.", "PATIENT_ID"),
    ("de", "IBAN DE89 3704 0044 0532 0130 01 (Tippfehler)", "IBAN_CODE"),
    ("de", "Seriennummer A123456789 am Typenschild.", "DE_KVNR"),
    ("en", "Batch 12345678901 shipped.", "DE_TAX_ID"),
    ("en", "Lot 000-12-3456 rejected.", "US_SSN"),
    ("en", "The patient reports mild pain and was seen in clinic.", "PERSON"),
    ("de", "Der Patient klagt über Dyspnoe.", "PERSON"),
    ("de", "Kontakt: Tel. 030 1234567", "PERSON"),
    ("de", "Lager getauscht; 45943 Betriebsstunden.", "LOCATION"),
    ("en", "This is confidential between us.", "CLASSIFICATION_MARKING"),
    ("de", "Er wollte Geheimnisse nicht teilen.", "CLASSIFICATION_MARKING"),
    ("en", "No contact data here at all.", "EMAIL_ADDRESS"),
]


@pytest.mark.parametrize(("lang", "text", "etype"), NEGATIVE)
def test_negative(analyzer: PiiAnalyzer, lang: str, text: str, etype: str) -> None:
    found = _types(analyzer, text, lang)
    assert etype not in found, found


def test_kvnr_invalid_checksum_needs_context(analyzer: PiiAnalyzer) -> None:
    bad = VALID_KVNR[:-1] + str((int(VALID_KVNR[-1]) + 1) % 10)
    assert "DE_KVNR" not in _types(analyzer, f"Bauteil {bad}", "de")
    assert "DE_KVNR" in _types(analyzer, f"Krankenkasse Versichertennr. {bad}", "de")


def test_language_specific_packs_are_separate() -> None:
    packs = load_rule_packs(["en", "de"])
    assert set(packs) == {"common", "en", "de"}
    en_names = {r.name for r in packs["en"].rules}
    de_names = {r.name for r in packs["de"].rules}
    assert "person_en_title" in en_names and "person_de_title" in de_names
    assert not en_names & de_names
    # German title rule does not run for English chunks
    assert not match_pack(packs["en"], "Herr Kesselring kam.")
    assert match_pack(packs["de"], "Herr Kesselring kam.")


def test_custom_patterns_from_config(tmp_path: Path) -> None:
    pack = custom_pattern_pack({"employee_id": r"EMP-\d{5}",
                                "ticket": r"Ticket (?P<value>T\d+)"})
    hits = match_pack(pack, "EMP-12345 opened Ticket T991")
    assert {(h.entity_type, h.start, h.end) for h in hits} == {
        ("EMPLOYEE_ID", 0, 9), ("TICKET", 24, 28)}
    an = PiiAnalyzer(("en",), custom_patterns={"employee_id": r"EMP-\d{5}"}, ner="regex",
                     models_root=tmp_path)
    assert "EMPLOYEE_ID" in an.entity_types()
    assert any(d.entity_type == "EMPLOYEE_ID" for d in an.analyze("id EMP-00001", "en"))
