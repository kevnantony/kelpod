"""Generate the synthetic ingest fixture corpus in ``tests/fixtures/corpus/`` (deterministic).

All content is ENTIRELY FICTIONAL (company "Helios Grid Services", made-up equipment, people,
phone numbers, e-mails and German KVNR-like IDs). The fake PII is intentional so the sanitize
stage can be tested on the same corpus.

Usage: ``uv run python scripts/make_fixture_corpus.py [--out tests/fixtures/corpus]``
Needs: fpdf2, Pillow, python-docx, pyarrow (all installed with the project).
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import random
import shutil
import zipfile
from pathlib import Path

FIXED_DATE = dt.datetime(2026, 1, 15, 9, 30, tzinfo=dt.UTC)
ZIP_DATE = (2026, 1, 15, 9, 30, 0)
ROOT = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------ content
PUMP_SECTIONS: list[tuple[str, list[str]]] = [
    ("1 Overview", [
        "The HX-200 coolant pump circulates glycol coolant through the inverter cabinets of "
        "the Nordhafen solar park and through the backup chiller of the Ostfeld dialysis "
        "centre. Each pump is rated for 42 litres per minute at a differential pressure of "
        "2.4 bar. The pump is driven by a 1.1 kW motor with an integrated frequency converter.",
        "This manual is valid for serial numbers HX2-1000 to HX2-1999. Older units with the "
        "grey terminal box use the procedures in the legacy manual HX-150.",
    ]),
    ("2 Safety", [
        "Isolate the pump electrically before opening the terminal box. Lock out breaker "
        "Q12 and attach a personal lock. Wait five minutes for the DC link capacitors of "
        "the converter to discharge.",
        "Coolant can reach 65 degrees Celsius. Wear heat resistant gloves and eye protection "
        "when venting the system.",
    ]),
    ("3 Fault codes", [
        "The converter shows a three digit fault code on its display. Codes starting with "
        "E1 relate to the hydraulic side, codes starting with E2 to the electrical side.",
        "Fault E104 means dry running: the pressure at the inlet fell below 0.3 bar for "
        "more than ten seconds. The pump stops and restarts automatically after 120 seconds "
        "for at most three attempts.",
    ]),
    ("3.1 Hydraulic faults", [
        "If E104 repeats, check the inlet strainer S3 for clogging and verify that the "
        "expansion vessel pre-charge is 1.5 bar. A clogged strainer is the most common "
        "cause of E104 in the first year of operation.",
        "Fault E117 indicates cavitation detected by the vibration sensor. Reduce the speed "
        "set point to 80 percent and inspect the impeller for pitting at the next "
        "planned maintenance window.",
    ]),
    ("3.2 Electrical faults", [
        "Fault E203 is an over-temperature of the converter heat sink above 85 degrees "
        "Celsius. Clean the cooling fins and check that the cabinet fan F2 is running.",
        "Fault E219 signals a phase loss on the supply. Measure all three phases at "
        "terminal X1 and tighten the terminals to 1.2 Nm.",
    ]),
    ("4 Maintenance schedule", [
        "Every 3 months: inspect the strainer S3, check for leaks at the mechanical seal "
        "and read out the fault log of the converter.",
        "Every 12 months: replace the coolant filter cartridge, check the expansion vessel "
        "pre-charge and verify the calibration of pressure sensor P1.",
        "Every 5 years or 30000 operating hours: replace the mechanical seal and the "
        "bearings of the motor.",
    ]),
    ("5 Contacts", [
        "Technical owner: Dr. Lena Marquardt, phone +49 30 5550 1234, e-mail "
        "lena.marquardt@helios-grid.example. Out of hours service desk: +49 171 5550 9876.",
    ]),
]

FAULT_TABLE = [
    ("Code", "Meaning", "First action"),
    ("E104", "Dry running", "Check strainer S3"),
    ("E117", "Cavitation", "Reduce speed to 80 %"),
    ("E203", "Heat sink over-temperature", "Clean fins, check fan F2"),
    ("E219", "Phase loss", "Measure phases at X1"),
]

SOP_MD = """# SOP-07 Inverter reset after grid fault

Revision: {revision}
Owner: Aylin Demir (aylin.demir@helios-grid.example, +49 40 5550 2233)

## Purpose

This standard operating procedure describes how to reset a string inverter of type SI-60 \
after the grid protection relay has tripped. It applies to the Nordhafen and Ostfeld sites.

## Preconditions

- The grid operator has confirmed that the fault is cleared.
- No alarm is active on the medium-voltage switchgear.
- The technician holds a valid switching authorisation of level B or higher.

## Procedure

1. Open the inverter monitoring page and note the fault code shown in the event list.
2. Acknowledge the fault on the inverter display with the OK key held for three seconds.
3. Wait until the DC voltage is stable above 600 V for at least sixty seconds.
4. Switch the AC breaker Q2 back on and observe the feed-in for five minutes.
5. Record the reset in the shift log with time, inverter number and fault code.

## Fault codes that must not be reset locally

| Code | Meaning | Escalation |
|---|---|---|
| G31 | Insulation fault on DC side | Call the duty engineer |
| G44 | Earth fault current above 300 mA | Call the duty engineer |
| G52 | Firmware integrity check failed | Open a ticket with the vendor |

## Notes

An inverter that trips more than three times in 24 hours must be taken out of service and \
inspected. Escalate to the duty engineer Marek Nowak on +49 160 5550 4411.
"""

HANDOVER_TXT = """SHIFT HANDOVER NOTE - OSTFELD DIALYSIS CENTRE, TECHNICAL SERVICES

Date: 2026-03-14, night shift to early shift
Handed over by: Jonas Feldmann (phone +49 30 5550 7788)
Received by: Sabine Krüger

1. Chiller CH-2 tripped twice with fault E203 between 02:10 and 03:40. The heat sink fins
were dusty and were cleaned. Fan F2 runs normally. Please monitor the converter temper-
ature during the morning peak.

2. Patient room 3 reported a noisy ventilation unit. No technical fault found; the filter
was replaced. The ward nurse asked that patient Greta Lindqvist (KVNR B987654321, born
12.07.1954) is informed before any further work in room 3.

3. The water treatment unit WT-1 completed its nightly disinfection cycle without alarms.
Conductivity after rinse: 2.1 microsiemens per centimetre.

OPEN ITEMS
- Order spare filter cartridges for HX-200 pumps (stock is down to two).
- Calibration of pressure sensor P1 on pump HX2-1042 is due next week.
"""

FAQ_HTML = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Battery storage FAQ</title>
<style>body {{ font-family: sans-serif; }}</style>
<script>console.log("analytics disabled");</script></head>
<body>
<nav><a href="#">Home</a> | <a href="#">Sites</a></nav>
<h1>Battery storage BS-500: frequently asked questions</h1>
<p>The BS-500 battery storage system at the Nordhafen site stores up to 500 kWh in
lithium iron phosphate modules. This page answers common questions of the operations team.</p>
<h2>What does alarm B12 mean?</h2>
<p>Alarm B12 means that the cell voltage difference inside one module exceeds 80 mV. The
battery management system starts a balancing cycle automatically. If the alarm persists for
more than two hours, the module must be inspected.</p>
<h2>How often is the fire suppression system tested?</h2>
<p>The aerosol fire suppression system is inspected every six months by the certified
contractor. The last inspection report was signed by {name}.</p>
<h2>Which modules were replaced?</h2>
<table>
<tr><th>Module</th><th>Replaced on</th><th>Reason</th></tr>
<tr><td>M-07</td><td>2025-11-03</td><td>Swollen cell</td></tr>
<tr><td>M-21</td><td>2026-02-18</td><td>Repeated alarm B12</td></tr>
</table>
<h2>Who do I call?</h2>
<ul><li>Duty engineer: Marek Nowak, +49 160 5550 4411</li>
<li>Battery vendor hotline: +49 89 5550 1000</li></ul>
<footer>Helios Grid Services - internal use only</footer>
</body></html>
"""

SITES_XML = """<?xml version="1.0" encoding="UTF-8"?>
<siteRegistry operator="Helios Grid Services">
  <title>Site registry 2026</title>
  <site>
    <id>NH-01</id>
    <name>Nordhafen solar park</name>
    <capacityMW>48.5</capacityMW>
    <contact>Aylin Demir</contact>
  </site>
  <site>
    <id>OF-02</id>
    <name>Ostfeld dialysis centre backup plant</name>
    <capacityMW>1.2</capacityMW>
    <contact>Jonas Feldmann</contact>
  </site>
  <note>All sites report fault statistics monthly to the reliability team.</note>
</siteRegistry>
"""

SCANNED_LINES = [
    "SERVICE NOTE SN-2291",
    "",
    "Site: Nordhafen solar park, inverter row C",
    "Technician: Jonas Feldmann",
    "",
    "Inverter SI-60 number C-14 showed fault G44 after",
    "heavy rain. Insulation resistance measured 0.8 MOhm",
    "on string 3. The connector was replaced and the",
    "inverter was returned to service at 14:20.",
    "",
    "Follow-up: re-measure string 3 in two weeks.",
]


# ------------------------------------------------------------------ writers
def make_pump_pdf(path: Path) -> None:
    from fpdf import FPDF

    class ManualPDF(FPDF):
        def header(self) -> None:
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 8, "Helios Grid Services - HX-200 Coolant Pump Maintenance Manual",
                      new_x="LMARGIN", new_y="NEXT", align="C")

        def footer(self) -> None:
            self.set_y(-12)
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 8, f"Page {self.page_no()} of {{nb}}", align="C")

    pdf = ManualPDF(format="A4")
    pdf.set_creation_date(FIXED_DATE)
    pdf.set_title("HX-200 Maintenance Manual")
    pdf.set_producer("zllf fixture generator")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)
    for i, (heading, paragraphs) in enumerate(PUMP_SECTIONS):
        if i in (0, 2, 5):
            pdf.add_page()
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 9, heading, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for para in paragraphs:
            pdf.multi_cell(0, 5, para, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)
        if heading.startswith("3 "):
            # explicit hyphenated line break to exercise de-hyphenation
            pdf.multi_cell(0, 5, "Persistent faults require an inspection of the mainte-\n"
                                 "nance history before the pump is restarted.",
                           new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)
            with pdf.table(col_widths=(20, 60, 60)) as table:
                for row in FAULT_TABLE:
                    r = table.row()
                    for cell in row:
                        r.cell(cell)
            pdf.ln(3)
    pdf.output(str(path))


def make_scanned_pdf(path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("L", (1240, 1754), 255)  # A4 at 150 dpi
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=30)
    y = 120
    for line in SCANNED_LINES:
        draw.text((110, y), line, fill=0, font=font)
        y += 48
    img.save(path, "PDF", resolution=150.0, title="scan", producer="zllf fixture generator",
             creationDate=FIXED_DATE.timetuple(), modDate=FIXED_DATE.timetuple())


def make_nameplate_png(path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("L", (900, 360), 255)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=34)
    lines = ["HX-200 COOLANT PUMP", "Serial HX2-1042", "Rated flow 42 l/min at 2.4 bar",
             "Motor 1.1 kW 400 V 50 Hz"]
    for i, line in enumerate(lines):
        draw.text((40, 40 + i * 70), line, fill=0, font=font)
    img.save(path, "PNG", optimize=True)


def _fix_zip_dates(path: Path) -> None:
    """Rewrite a zip (DOCX) with fixed timestamps so the output is byte-reproducible."""
    src = zipfile.ZipFile(io.BytesIO(path.read_bytes()))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as dst:
        for info in sorted(src.infolist(), key=lambda i: i.filename):
            new = zipfile.ZipInfo(info.filename, date_time=ZIP_DATE)
            new.compress_type = zipfile.ZIP_DEFLATED
            new.external_attr = 0o644 << 16
            dst.writestr(new, src.read(info.filename))
    path.write_bytes(buf.getvalue())


def make_incident_docx(path: Path) -> None:
    import docx

    d = docx.Document()
    d.core_properties.author = "fixture"
    d.core_properties.created = FIXED_DATE.replace(tzinfo=None)
    d.core_properties.modified = FIXED_DATE.replace(tzinfo=None)
    d.core_properties.last_modified_by = "fixture"
    d.add_heading("Incident report IR-0412: chiller CH-2 outage", level=0)
    d.add_heading("Summary", level=1)
    d.add_paragraph(
        "On 2026-02-09 the backup chiller CH-2 of the Ostfeld dialysis centre stopped for "
        "47 minutes. The HX-200 coolant pump reported fault E104 (dry running) three times and "
        "locked out. Dialysis treatments continued on the primary chiller; no patient was harmed.")
    d.add_heading("Root cause", level=1)
    d.add_paragraph(
        "The inlet strainer S3 was clogged with debris from the January pipe works. The "
        "expansion vessel pre-charge was 0.9 bar instead of the required 1.5 bar, which made "
        "the inlet pressure drop faster during the start-up of the second compressor.")
    d.add_heading("Actions", level=1)
    for item in ("Clean strainer S3 and flush the inlet line.",
                 "Restore the expansion vessel pre-charge to 1.5 bar.",
                 "Add the strainer to the monthly inspection round for six months."):
        d.add_paragraph(item, style="List Bullet")
    d.add_heading("Timeline", level=2)
    table = d.add_table(rows=1, cols=3)
    for cell, text in zip(table.rows[0].cells, ("Time", "Event", "By"), strict=True):
        cell.text = text
    for row in (("06:12", "E104 first occurrence", "converter"),
                ("06:31", "Duty technician informed", "Jonas Feldmann"),
                ("06:59", "Strainer cleaned, pump restarted", "Jonas Feldmann")):
        cells = table.add_row().cells
        for cell, text in zip(cells, row, strict=True):
            cell.text = text
    d.add_heading("Contacts", level=1)
    d.add_paragraph("Report author: Sabine Krüger, sabine.krueger@helios-grid.example, "
                    "+49 30 5550 6611. Patient liaison: KVNR A123456780 (anonymised test ID).")
    d.save(str(path))
    _fix_zip_dates(path)


def make_shift_log_csv(path: Path) -> None:
    rng = random.Random(42)
    techs = [("Jonas Feldmann", "+49 30 5550 7788"), ("Aylin Demir", "+49 40 5550 2233"),
             ("Marek Nowak", "+49 160 5550 4411")]
    assets = ["HX2-1042", "HX2-1043", "SI-60 C-14", "SI-60 C-15", "BS-500", "CH-2"]
    faults = ["E104", "E117", "E203", "E219", "G31", "G44", "B12", "none"]
    lines = ["date,shift,technician,phone,asset,fault_code,minutes_down,remark"]
    for day in range(1, 31):
        for shift in ("early", "late", "night"):
            if rng.random() < 0.55:
                continue
            name, phone = rng.choice(techs)
            fault = rng.choice(faults)
            minutes = 0 if fault == "none" else rng.randint(5, 180)
            remark = "routine round" if fault == "none" else f"fault {fault} cleared"
            lines.append(f"2026-03-{day:02d},{shift},{name},{phone},{rng.choice(assets)},"
                         f"{fault},{minutes},{remark}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_fault_parquet(path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [
        ("E104", "HX-200", "Dry running, inlet below 0.3 bar", "hydraulic", 2),
        ("E117", "HX-200", "Cavitation detected by vibration sensor", "hydraulic", 2),
        ("E203", "HX-200", "Converter heat sink above 85 C", "electrical", 1),
        ("E219", "HX-200", "Supply phase loss", "electrical", 1),
        ("G31", "SI-60", "Insulation fault on DC side", "electrical", 1),
        ("G44", "SI-60", "Earth fault current above 300 mA", "electrical", 1),
        ("G52", "SI-60", "Firmware integrity check failed", "software", 2),
        ("B12", "BS-500", "Cell voltage difference above 80 mV", "battery", 3),
    ]
    table = pa.table({
        "code": [r[0] for r in rows],
        "equipment": [r[1] for r in rows],
        "description": [r[2] for r in rows],
        "category": [r[3] for r in rows],
        "priority": pa.array([r[4] for r in rows], pa.int32()),
    })
    pq.write_table(table, path, compression="zstd")


def build(out: Path) -> list[Path]:
    if out.exists():
        shutil.rmtree(out)
    for sub in ("manuals", "sop", "logs", "data", "web", "reports", "scans", "notes", "misc"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    make_pump_pdf(out / "manuals" / "hx200_pump_manual.pdf")
    (out / "sop" / "sop07_inverter_reset.md").write_text(
        SOP_MD.format(revision="3 (2026-01-10)"), encoding="utf-8")
    # near-duplicate: same SOP, only the revision line differs
    (out / "sop" / "sop07_inverter_reset_copy.md").write_text(
        SOP_MD.format(revision="3a (2026-01-12, typo fixed)"), encoding="utf-8")
    make_shift_log_csv(out / "logs" / "shift_log_2026-03.csv")
    make_fault_parquet(out / "data" / "fault_codes.parquet")
    (out / "web" / "battery_storage_faq.html").write_text(
        FAQ_HTML.format(name="Aylin Demir"), encoding="utf-8")
    (out / "data" / "site_registry.xml").write_text(SITES_XML, encoding="utf-8")
    make_incident_docx(out / "reports" / "incident_IR-0412.docx")
    make_scanned_pdf(out / "scans" / "service_note_SN-2291.pdf")
    make_nameplate_png(out / "scans" / "nameplate_HX2-1042.png")
    (out / "notes" / "shift_handover_2026-03-14.txt").write_text(HANDOVER_TXT, encoding="utf-8")
    # unsupported: legacy CAD drawing (fake bytes)
    (out / "misc" / "pump_drawing.dwg").write_bytes(b"AC1032" + bytes(range(64)))
    return sorted(p for p in out.rglob("*") if p.is_file())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "tests" / "fixtures" / "corpus")
    args = ap.parse_args()
    files = build(args.out)
    total = sum(p.stat().st_size for p in files)
    for p in files:
        print(f"{p.stat().st_size:>9}  {p.relative_to(args.out)}")
    print(f"{len(files)} files, {total / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
