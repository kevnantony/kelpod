#!/usr/bin/env python3
"""Generate labelled synthetic clinical notes + maintenance logs (EN/DE) for PII evaluation.

Ground truth for FR-03 recall (DEVELOPMENT_PLAN P0.5.3). Entirely fictional, template based,
deterministic for a given ``--seed``; no model needed. Output JSONL rows::

    {"id": "n0001", "language": "de", "kind": "clinical",
     "text": "...", "spans": [{"start": 10, "end": 22, "entity_type": "PERSON"}, ...]}

Usage::

    uv run python scripts/make_synthetic_notes.py --n 200 --out bench/pii_eval/notes.jsonl

Some person/location mentions are deliberately *not* introduced by a title or label (e.g. a
sentence starting with a bare name), so the regex-only mode is not graded on easy cases only.
Unlabelled look-alikes (admission dates, work orders, serial numbers, operating hours) test
precision.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "bench" / "pii_eval" / "notes.jsonl"

# ---------------------------------------------------------------- pools (fictional)
FIRST_EN = ["James", "Mary", "Robert", "Linda", "Michael", "Susan", "David", "Karen", "Thomas",
            "Nancy", "Daniel", "Emily", "Andrew", "Olivia", "Brian", "Grace", "Kevin", "Megan",
            "Owen", "Hannah", "Ethan", "Chloe", "Samuel", "Ruby"]
LAST_EN = ["Whitaker", "Holloway", "Prescott", "Langley", "Fairbanks", "Thornton", "Ashworth",
           "Kendall", "Mercer", "Callahan", "Dunmore", "Hartley", "Pembroke", "Sutton",
           "Ridley", "Garrison", "Bellamy", "Crawford", "Hensley", "Marlow"]
FIRST_DE = ["Anna", "Lukas", "Sophie", "Jonas", "Marie", "Felix", "Lea", "Paul", "Hannah",
            "Moritz", "Lena", "Tobias", "Katrin", "Jürgen", "Ursula", "Dieter", "Sabine",
            "Matthias", "Birgit", "Florian", "Greta", "Henrik", "Ilse", "Volker"]
LAST_DE = ["Brandauer", "Kesselring", "Wendlandt", "Hollerbach", "Steinmetz", "Ruppert",
           "Lindqvist", "Obermaier", "Faßbender", "Grünewald", "Kowalczyk", "Pfannenstiel",
           "Rautenberg", "Schwarzkopf", "Tiedemann", "Ulbricht", "Vogelsang", "Wohlfahrt",
           "Zimmermann-Kühn", "Albrecht"]
CITIES_DE = ["Berlin", "Hamburg", "München", "Köln", "Lüneburg", "Bamberg", "Paderborn",
             "Görlitz", "Tübingen", "Cottbus", "Leipzig", "Flensburg", "Passau", "Hildesheim"]
CITIES_EN = ["Boston", "Chicago", "Springfield", "Fairview", "Riverton", "Ashford", "London",
             "Manchester", "Brookhaven", "Millbrook", "Denver", "Oakridge"]
STREETS_DE = ["Lindenstraße", "Hauptstraße", "Bahnhofstr.", "Goethestraße", "Am Mühlbach",
              "Birkenweg", "Karl-Marx-Allee", "Schillerplatz", "Rosengasse", "Kastanienallee"]
STREETS_EN = ["Maple Street", "Oak Avenue", "Willow Lane", "Harbor Road", "Elm Drive",
              "Kingsway Boulevard", "Chestnut Court", "Mill Road"]
STATES = ["MA", "IL", "OH", "CO", "NY", "TX", "WA"]
MONTHS_EN = ["January", "February", "March", "April", "May", "June", "July", "August",
             "September", "October", "November", "December"]
MONTHS_DE = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August",
             "September", "Oktober", "November", "Dezember"]
MARKINGS_EN = ["CONFIDENTIAL", "STRICTLY CONFIDENTIAL", "SECRET//NOFORN", "INTERNAL USE ONLY",
               "TOP SECRET"]
MARKINGS_DE = ["VS-NUR FÜR DEN DIENSTGEBRAUCH", "VERTRAULICH", "VS-NfD", "STRENG GEHEIM"]


# ---------------------------------------------------------------- value generators
def _iban_de(rng: random.Random) -> str:
    bban = f"{rng.randint(10000000, 89999999)}{rng.randint(0, 9999999999):010d}"
    num = int("".join(str(int(c, 36)) for c in bban + "DE00"))
    iban = f"DE{98 - num % 97:02d}{bban}"
    if rng.random() < 0.7:
        return " ".join(iban[i:i + 4] for i in range(0, len(iban), 4))
    return iban


def _kvnr(rng: random.Random) -> str:
    letter = chr(ord("A") + rng.randint(0, 25))
    digits = f"{rng.randint(0, 99999999):08d}"
    seq = f"{ord(letter) - 64:02d}{digits}"
    total = 0
    for i, ch in enumerate(seq):
        prod = int(ch) * (1 if i % 2 == 0 else 2)
        total += prod // 10 + prod % 10
    return f"{letter}{digits}{total % 10}"


def _tax_id(rng: random.Random) -> str:
    first = str(rng.randint(1, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(9))
    product = 10
    for ch in first:
        s = (int(ch) + product) % 10 or 10
        product = (s * 2) % 11
    check = 11 - product
    tid = first + str(0 if check == 10 else check)
    return f"{tid[:2]} {tid[2:5]} {tid[5:8]} {tid[8:]}" if rng.random() < 0.4 else tid


def _ssn(rng: random.Random) -> str:
    area = rng.choice([a for a in range(1, 900) if a != 666])
    return f"{area:03d}-{rng.randint(1, 99):02d}-{rng.randint(1, 9999):04d}"


def _phone(rng: random.Random, lang: str) -> str:
    if lang == "de":
        return rng.choice([
            f"+49 {rng.choice(['30', '40', '89', '221', '511'])} {rng.randint(1000000, 9999999)}",
            f"0{rng.choice(['30', '40', '89', '221'])} {rng.randint(1000000, 99999999)}",
            f"01{rng.choice(['51', '60', '71', '76'])}-{rng.randint(1000000, 9999999)}",
            f"0{rng.choice(['221', '511', '761'])}/{rng.randint(100000, 9999999)}",
            f"+49 (0) {rng.choice(['30', '89'])} {rng.randint(100, 999)} {rng.randint(1000, 9999)}",
        ])
    return rng.choice([
        f"({rng.randint(201, 989)}) 555-{rng.randint(1000, 9999)}",
        f"{rng.randint(201, 989)}-555-{rng.randint(1000, 9999)}",
        f"+1 {rng.randint(201, 989)} 555 {rng.randint(1000, 9999)}",
        f"+44 20 7946 {rng.randint(1000, 9999)}",
    ])


def _dob(rng: random.Random, lang: str) -> str:
    y, m, d = rng.randint(1932, 2005), rng.randint(1, 12), rng.randint(1, 28)
    if lang == "de":
        return rng.choice([f"{d:02d}.{m:02d}.{y}", f"{d}.{m}.{y}", f"{d}. {MONTHS_DE[m - 1]} {y}",
                           f"{y}-{m:02d}-{d:02d}"])
    return rng.choice([f"{m:02d}/{d:02d}/{y}", f"{y}-{m:02d}-{d:02d}",
                       f"{MONTHS_EN[m - 1]} {d}, {y}", f"{d} {MONTHS_EN[m - 1]} {y}"])


def _date(rng: random.Random, lang: str) -> str:
    y, m, d = rng.randint(2021, 2025), rng.randint(1, 12), rng.randint(1, 28)
    return f"{d:02d}.{m:02d}.{y}" if lang == "de" else f"{m:02d}/{d:02d}/{y}"


def _pid(rng: random.Random) -> str:
    return rng.choice([
        f"PID-{rng.randint(2019, 2025)}-{rng.randint(1, 999999):06d}",
        f"MRN{rng.randint(10000000, 99999999)}",
        f"{rng.randint(2019, 2025)}/{rng.randint(100000, 999999)}",
        f"C-{rng.randint(10000, 99999)}",
        f"FALL-{rng.randint(1000000, 9999999)}",
    ])


class Persona:
    """Consistent fictional identity for one note."""

    def __init__(self, rng: random.Random, lang: str) -> None:
        firsts, lasts = (FIRST_DE, LAST_DE) if lang == "de" else (FIRST_EN, LAST_EN)
        self.first = rng.choice(firsts)
        self.last = rng.choice(lasts)
        self.doctor = f"{rng.choice(firsts)} {rng.choice(lasts)}"
        self.doc_last = self.doctor.split()[-1]
        self.tech = f"{rng.choice(firsts)} {rng.choice(lasts)}"
        self.relative = rng.choice(firsts)
        self.relfull = f"{self.relative} {self.last}"
        self.city = rng.choice(CITIES_DE if lang == "de" else CITIES_EN)
        self.city2 = rng.choice(CITIES_DE if lang == "de" else CITIES_EN)
        if lang == "de":
            self.street = f"{rng.choice(STREETS_DE)} {rng.randint(1, 140)}"
            self.zipcity = f"{rng.randint(10000, 99999)} {self.city}"
        else:
            self.street = f"{rng.randint(1, 4999)} {rng.choice(STREETS_EN)}"
            self.zipcity = f"{self.city}, {rng.choice(STATES)} {rng.randint(10000, 99999)}"
        domain = rng.choice(["klinikum-example.de", "praxis-beispiel.de", "example-hospital.org",
                             "wartung-example.com"])
        self.email = (f"{_ascii(self.first)}.{_ascii(self.last)}@{domain}").lower()
        self.phone = _phone(rng, lang)
        self.iban = _iban_de(rng)
        self.kvnr = _kvnr(rng)
        self.tax = _tax_id(rng)
        self.ssn = _ssn(rng)
        self.pid = _pid(rng)
        self.dob = _dob(rng, lang)
        self.mark = rng.choice(MARKINGS_DE if lang == "de" else MARKINGS_EN)


def _ascii(s: str) -> str:
    return (s.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
             .replace("Ä", "Ae").replace("Ö", "Oe").replace("Ü", "Ue"))


# slot -> (attribute or callable, entity_type | None)
SLOTS: dict[str, tuple[str, str | None]] = {
    "NAME": ("full", "PERSON"), "FIRST": ("first", "PERSON"), "LAST": ("last", "PERSON"),
    "DOCTOR": ("doctor", "PERSON"), "DOCLAST": ("doc_last", "PERSON"), "TECH": ("tech", "PERSON"),
    "RELATIVE": ("relative", "PERSON"), "RELFULL": ("relfull", "PERSON"),
    "CITY": ("city", "LOCATION"), "CITY2": ("city2", "LOCATION"), "STREET": ("street", "LOCATION"),
    "ZIPCITY": ("zipcity", "LOCATION"), "EMAIL": ("email", "EMAIL_ADDRESS"),
    "PHONE": ("phone", "PHONE_NUMBER"), "IBAN": ("iban", "IBAN_CODE"),
    "KVNR": ("kvnr", "DE_KVNR"), "TAXID": ("tax", "DE_TAX_ID"), "SSN": ("ssn", "US_SSN"),
    "PID": ("pid", "PATIENT_ID"), "DOB": ("dob", "DATE_OF_BIRTH"),
    "MARK": ("mark", "CLASSIFICATION_MARKING"),
}

# ---------------------------------------------------------------- templates
CLINICAL_EN = [
    "Patient: {NAME}\nDOB: {DOB}\nMRN: {PID}",
    "Patient Name: {NAME}, born {DOB}. Case No. {PID}.",
    "{NAME} is a {AGE}-year-old patient admitted on {DATE} with chest pain.",
    "Mr. {LAST} reports shortness of breath since {DATE}. BP {BP} mmHg, HR {HR}/min.",
    "Mrs. {LAST} was seen by Dr. {DOCTOR} in the emergency department.",
    "Attending physician: Dr. {DOCLAST}. Referred by Dr. {DOCTOR}.",
    "Contact: {PHONE}, e-mail {EMAIL}.",
    "Emergency contact: {RELFULL}, phone {PHONE}.",
    "Address: {STREET}, {ZIPCITY}.",
    "The patient lives in {CITY} and works night shifts.",
    "She recently returned from a trip to {CITY2}.",
    "SSN {SSN} on file for billing; insurance verified on {DATE}.",
    "Medication: Metoprolol {DOSE} mg twice daily, Ramipril 5 mg once daily.",
    "Labs on {DATE}: Hb {HB} g/dL, creatinine 1.{HR} mg/dL, CRP {DOSE} mg/L.",
    "Plan: echocardiography, follow-up in clinic in {AGE} days.",
    "His wife {RELATIVE} accompanied him and will collect the discharge letter.",
    "Discharge summary dictated on {DATE}.\nBest regards,\nDr. {DOCTOR}",
    "Signed: {DOCTOR}, MD",
]
CLINICAL_DE = [
    "Patient: {NAME}, geb. {DOB}\nFallnummer: {PID}",
    "Patientin {NAME} (* {DOB}) wurde am {DATE} stationär aufgenommen.",
    "{NAME} stellte sich am {DATE} mit Thoraxschmerz in der Notaufnahme vor.",
    "Herr {LAST} berichtet über Dyspnoe seit dem {DATE}. RR {BP} mmHg, HF {HR}/min.",
    "Frau {LAST} wurde von Dr. med. {DOCTOR} untersucht.",
    "Behandelnder Arzt: Dr. {DOCLAST}. Überwiesen von: Dr. {DOCTOR}.",
    "Kontakt: Tel. {PHONE}, E-Mail {EMAIL}.",
    "Angehörige: Tochter {RELFULL}, erreichbar unter {PHONE}.",
    "Anschrift: {STREET}, {ZIPCITY}.",
    "Die Patientin ist wohnhaft in {CITY} und lebt allein.",
    "Zuletzt Urlaub in {CITY2}, keine Auffälligkeiten.",
    "Versichertennummer (KVNR): {KVNR}. Steuer-ID: {TAXID}.",
    "Erstattung auf IBAN {IBAN} veranlasst.",
    "Medikation: Metoprolol {DOSE} mg 1-0-1, Ramipril 5 mg 1-0-0.",
    "Labor vom {DATE}: Hb {HB} g/dl, Kreatinin 1,{HR} mg/dl, CRP {DOSE} mg/l.",
    "Procedere: Echokardiographie, Wiedervorstellung in {AGE} Tagen.",
    "Mit freundlichen Grüßen\nDr. {DOCTOR}",
    "gez. {DOCTOR}, Oberarzt",
]
MAINT_EN = [
    "Work order WO-{WO} opened on {DATE} for pump P-{HR}.",
    "Technician: {TECH}\nPhone: {PHONE}",
    "Inspected by {TECH} on {DATE}; vibration within limits ({HB} mm/s).",
    "Site address: {STREET}, {ZIPCITY}.",
    "Contractor contact {EMAIL}, invoice to account {IBAN}.",
    "Replaced bearing SN {WO}-{HR}; operating hours {OPH}.",
    "{TECH} reported a hydraulic leak at valve V-{HR}.",
    "Approved by: Mr. {LAST}",
    "Next inspection due {DATE}. Torque checked at {DOSE} Nm.",
    "Unit transferred from {CITY2} depot after overhaul.",
    "Best regards,\n{TECH}",
]
MAINT_DE = [
    "Auftrag WA-{WO} am {DATE} für Pumpe P-{HR} eröffnet.",
    "Techniker: {TECH}\nTel.: {PHONE}",
    "Geprüft von: {TECH} am {DATE}; Schwingung im Grenzbereich ({HB} mm/s).",
    "Standort: {STREET}, {ZIPCITY}.",
    "Ansprechpartner: Herr {LAST}, E-Mail {EMAIL}.",
    "Lager getauscht, Seriennummer {WO}-{HR}; {OPH} Betriebsstunden.",
    "{TECH} meldete eine Leckage an Ventil V-{HR}.",
    "Freigegeben von: Frau {LAST}",
    "Nächste Prüfung fällig am {DATE}. Drehmoment {DOSE} Nm.",
    "Aggregat aus {CITY2} überführt, Rechnung auf IBAN {IBAN}.",
    "Mit freundlichen Grüßen\n{TECH}",
]
TEMPLATES = {("en", "clinical"): CLINICAL_EN, ("de", "clinical"): CLINICAL_DE,
             ("en", "maintenance"): MAINT_EN, ("de", "maintenance"): MAINT_DE}
_SLOT = re.compile(r"\{([A-Z0-9]+)\}")


def _filler(name: str, rng: random.Random, lang: str) -> str:
    fillers: dict[str, Callable[[], str]] = {
        "DATE": lambda: _date(rng, lang),
        "AGE": lambda: str(rng.randint(18, 92)),
        "BP": lambda: f"{rng.randint(100, 180)}/{rng.randint(60, 100)}",
        "HR": lambda: str(rng.randint(50, 120)),
        "HB": lambda: f"{rng.randint(8, 16)}{',' if lang == 'de' else '.'}{rng.randint(0, 9)}",
        "DOSE": lambda: str(rng.choice([5, 10, 25, 47, 95, 120])),
        "WO": lambda: str(rng.randint(10000, 99999)),
        "OPH": lambda: str(rng.randint(10000, 99999)),
    }
    return fillers[name]()


def render(template: str, persona: Persona, rng: random.Random, lang: str,
           offset: int) -> tuple[str, list[dict[str, object]]]:
    parts: list[str] = []
    spans: list[dict[str, object]] = []
    pos = offset
    last = 0
    for m in _SLOT.finditer(template):
        lit = template[last:m.start()]
        parts.append(lit)
        pos += len(lit)
        slot = m.group(1)
        if slot in SLOTS:
            attr, etype = SLOTS[slot]
            value = f"{persona.first} {persona.last}" if attr == "full" else getattr(persona,
                                                                                    attr)
        else:
            value, etype = _filler(slot, rng, lang), None
        parts.append(value)
        if etype is not None:
            spans.append({"start": pos, "end": pos + len(value), "entity_type": etype})
        pos += len(value)
        last = m.end()
    parts.append(template[last:])
    return "".join(parts), spans


def make_note(idx: int, rng: random.Random, lang: str, kind: str) -> dict[str, object]:
    persona = Persona(rng, lang)
    pool = TEMPLATES[(lang, kind)]
    k = rng.randint(4, min(8, len(pool)))
    chosen = sorted(rng.sample(range(len(pool)), k))
    lines: list[str] = []
    spans: list[dict[str, object]] = []
    offset = 0
    if rng.random() < 0.15:
        text, sp = render("{MARK}", persona, rng, lang, offset)
        lines.append(text)
        spans.extend(sp)
        offset += len(text) + 1
    for t in chosen:
        text, sp = render(pool[t], persona, rng, lang, offset)
        lines.append(text)
        spans.extend(sp)
        offset += len(text) + 1
    return {"id": f"n{idx:04d}", "language": lang, "kind": kind, "text": "\n".join(lines),
            "spans": spans}


def generate(n: int, seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    combos = [("en", "clinical"), ("de", "clinical"), ("en", "maintenance"),
              ("de", "maintenance")]
    return [make_note(i, rng, *combos[i % len(combos)]) for i in range(n)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    notes = generate(args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for note in notes:
            f.write(json.dumps(note, ensure_ascii=False) + "\n")
    n_spans = sum(len(n["spans"]) for n in notes)  # type: ignore[arg-type]
    print(f"wrote {len(notes)} notes / {n_spans} labelled spans -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
