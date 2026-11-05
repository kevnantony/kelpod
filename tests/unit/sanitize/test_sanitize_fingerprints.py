"""Salted fingerprints + n-gram leakage scan."""

from __future__ import annotations

from pathlib import Path

from zllf.stages.sanitize.fingerprints import (
    FingerprintBuilder,
    contains_fingerprint,
    derive_salt,
    find_fingerprint_hits,
    fingerprint_value,
    load_fingerprints,
    salt_id,
)


def _fp(values: dict[str, str], salt: bytes, tmp_path: Path) -> frozenset[str]:
    b = FingerprintBuilder(salt)
    for value, etype in values.items():
        b.add(etype, value)
    b.write(tmp_path / "fp.json")
    return load_fingerprints(tmp_path / "fp.json").hashes


def test_salt_is_deterministic_and_run_specific() -> None:
    assert derive_salt("run1", 1) == derive_salt("run1", 1)
    assert derive_salt("run1", 1) != derive_salt("run1", 2)
    assert derive_salt("run1", 1) != derive_salt("run2", 1)
    assert len(salt_id(derive_salt("r", 1))) == 16


def test_contains_fingerprint_detects_variants(tmp_path: Path) -> None:
    salt = derive_salt("run", 7)
    fps = _fp({"Anna Kesselring": "PERSON", "DE89 3704 0044 0532 0130 00": "IBAN_CODE",
               "anna.k@example.org": "EMAIL_ADDRESS"}, salt, tmp_path)
    assert contains_fingerprint("The patient ANNA kesselring, aged 50", fps, salt)
    assert contains_fingerprint("pay to DE89370400440532013000 now", fps, salt)
    assert contains_fingerprint("pay to de89-3704-0044-0532-0130-00", fps, salt)
    assert contains_fingerprint("mail: anna.k@example.org.", fps, salt)
    assert not contains_fingerprint("Anna met Mr Kessel ring", fps, salt)
    assert not contains_fingerprint("nothing here", fps, salt)
    # wrong salt -> no hits
    assert not contains_fingerprint("ANNA kesselring", fps, derive_salt("other", 7))


def test_hits_report_hashes_only(tmp_path: Path) -> None:
    salt = b"s"
    fps = _fp({"Volker Rautenberg": "PERSON"}, salt, tmp_path)
    hits = find_fingerprint_hits("Dr. Volker Rautenberg signed", fps, salt)
    assert hits and hits <= set(fps)


def test_file_contains_no_raw_values(tmp_path: Path) -> None:
    salt = derive_salt("run", 1)
    _fp({"Lena Tiedemann": "PERSON", "0171-2345678": "PHONE_NUMBER"}, salt, tmp_path)
    raw = (tmp_path / "fp.json").read_text()
    for needle in ("Lena", "tiedemann", "2345678", salt.hex()):
        assert needle.lower() not in raw.lower()
    data = load_fingerprints(tmp_path / "fp.json")
    assert data.salt_id == salt_id(salt)
    assert set(data.by_type) == {"PERSON", "PHONE_NUMBER"}


def test_short_values_are_not_fingerprinted() -> None:
    assert fingerprint_value("Al", b"s") == []
    assert len(fingerprint_value("Anna Kesselring", b"s")) == 2  # spaced + compact
