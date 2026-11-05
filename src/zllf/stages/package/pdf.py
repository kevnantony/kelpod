"""Model Readiness Scorecard PDF (PRD §7.1 step 4, P2.7.3) rendered with fpdf2 + matplotlib.

Page 1: verdict banner, plain-language gate traffic lights, headline metrics for all four
variants. Page 2: side-by-side example answers (synthetic questions over *masked* text
already present in the run artefacts; canary secrets and e-mail-like strings are scrubbed
again as defence in depth). Appendix: trials, loss curves, lineage.

Pure Python, no network: the Unicode font is DejaVu Sans shipped inside matplotlib.
"""

from __future__ import annotations

import datetime as dt
import io
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from zllf.stages.package.artefacts import GATE_ORDER, VARIANTS, Example
from zllf.stages.package.report import GATE_LABELS

FONT = "DejaVu"
VARIANT_LABELS: dict[str, str] = {
    "base": "Base model", "base_rag": "Base + search", "adapter": "New model",
    "adapter_rag": "New model + search",
}
GATE_EXPLANATIONS: dict[str, str] = {
    "G1_domain": "Answers questions about your documents clearly better than the base model.",
    "G2_forgetting": "General knowledge, reasoning and instruction following did not degrade.",
    "G3_hallucination": "Says it does not know instead of inventing answers.",
    "G4_memorization": "No planted secret or private string was reproduced.",
    "G5_schema": "Structured (JSON) outputs follow the required schema.",
}
GREEN, RED, GREY = (22, 140, 72), (196, 43, 43), (150, 150, 150)
INK, MUTED, RULE = (25, 25, 25), (100, 100, 100), (215, 215, 215)
MAX_EXAMPLE_CHARS = 320
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


# ---------------------------------------------------------------- helpers
def font_paths() -> dict[str, Path]:
    """DejaVu Sans regular/bold/oblique from matplotlib's bundled fonts."""
    import matplotlib

    ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    return {"": ttf / "DejaVuSans.ttf", "B": ttf / "DejaVuSans-Bold.ttf",
            "I": ttf / "DejaVuSans-Oblique.ttf"}


def fmt(value: Any) -> str:
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if not isinstance(value, int | float):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if value != 0 and abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"


def scrub(text: str, secrets: Iterable[str] = (), limit: int = MAX_EXAMPLE_CHARS) -> str:
    """Remove canary secrets / e-mail-like strings / control chars and truncate."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[removed]")
    text = _EMAIL.sub("[EMAIL]", text)
    text = _CONTROL.sub(" ", text).replace("\t", " ")
    text = re.sub(r"\s+\n", "\n", text).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def loss_chart_png(curves: dict[str, list[list[float]]]) -> bytes | None:
    """Render loss curves to PNG bytes (Agg, no pyplot global state)."""
    if not curves:
        return None
    from matplotlib.figure import Figure

    fig = Figure(figsize=(7.0, 3.0), dpi=150)
    ax = fig.add_subplot()
    colors = ["#2563eb", "#d97706", "#059669", "#7c3aed"]
    for i, (name, points) in enumerate(sorted(curves.items())):
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        ax.plot(xs, ys, label=name, color=colors[i % len(colors)], linewidth=1.6)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, color="#e5e7eb", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", metadata={"Software": None})
    return buf.getvalue()


def headline_metrics(metrics: dict[str, dict[str, Any]], limit: int = 8) -> list[str]:
    """Metric names to show on page 1: domain/gold first, then others, alphabetically."""
    names = sorted({m for per_variant in metrics.values() for m in per_variant})

    def rank(name: str) -> tuple[int, str]:
        low = name.lower()
        if "domain" in low:
            return (0, name)
        if "gold" in low:
            return (1, name)
        if "unanswer" in low or "abstain" in low or "halluc" in low:
            return (2, name)
        return (3, name)

    return sorted(names, key=rank)[:limit]


# ---------------------------------------------------------------- document
class _Scorecard(FPDF):
    def __init__(self, run_label: str) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.run_label = run_label
        for style, path in font_paths().items():
            self.add_font(FONT, style, str(path))
        self.set_auto_page_break(auto=True, margin=16)
        self.set_margins(16, 16, 16)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font(FONT, "", 7.5)
        self.set_text_color(*MUTED)
        self.cell(0, 5, f"ZLLF Model Readiness Scorecard · {self.run_label} · page "
                        f"{self.page_no()}/{{nb}}", align="C")

    # -- primitives
    def h1(self, text: str) -> None:
        self.set_font(FONT, "B", 20)
        self.set_text_color(*INK)
        self.cell(0, 10, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def h2(self, text: str) -> None:
        self.ln(3)
        self.set_font(FONT, "B", 13)
        self.set_text_color(*INK)
        self.cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*RULE)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)

    def para(self, text: str, size: float = 9.5, style: str = "",
             color: tuple[int, int, int] = INK, h: float = 4.8) -> None:
        self.set_font(FONT, style, size)
        self.set_text_color(*color)
        self.multi_cell(0, h, text, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def simple_table(self, header: Sequence[str], rows: Sequence[Sequence[str]],
              widths: Sequence[float], size: float = 8.0) -> None:
        self.set_draw_color(*RULE)
        self.set_font(FONT, "B", size)
        self.set_fill_color(242, 242, 242)
        self.set_text_color(*INK)
        for text, w in zip(header, widths, strict=True):
            self.cell(w, 6, text, border="B", fill=True)
        self.ln()
        self.set_font(FONT, "", size)
        for row in rows:
            if self.get_y() > self.h - 24:
                self.add_page()
            for text, w in zip(row, widths, strict=True):
                self.cell(w, 5.5, _fit(self, text, w), border="B")
            self.ln()


def _fit(pdf: FPDF, text: str, width: float) -> str:
    if pdf.get_string_width(text) <= width - 1.5:
        return text
    while text and pdf.get_string_width(text + "…") > width - 1.5:
        text = text[:-1]
    return text + "…"


# ---------------------------------------------------------------- pages
def _page_summary(pdf: _Scorecard, report: dict[str, Any]) -> None:
    pdf.add_page()
    pdf.h1("Model Readiness Scorecard")
    ws = report.get("workspace") or "–"
    pdf.para(f"Workspace: {ws}   ·   Run: {report['run_id']}   ·   Goal: Mode {report['mode']} "
             f"({report['profile']})   ·   Created: {report.get('created_at') or '–'}",
             size=9, color=MUTED)
    pdf.ln(3)

    ready = bool(report["ready"])
    pdf.set_fill_color(*(GREEN if ready else RED))
    pdf.set_text_color(255, 255, 255)
    pdf.set_font(FONT, "B", 22)
    pdf.cell(0, 16, "READY" if ready else "NOT READY", align="C", fill=True,
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font(FONT, "", 9.5)
    msg = ("All quality and safety checks passed. This model can be deployed."
           if ready else "At least one quality or safety check failed. Deployment is blocked "
           "unless an administrator overrides it with a written justification.")
    pdf.multi_cell(0, 6, msg, align="C", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.h2("Quality and safety checks")
    gates = {g["gate"]: g for g in report["evaluation"]["gates"]}
    for gate in GATE_ORDER:
        g = gates.get(gate)
        color = GREY if g is None else GREEN if g["passed"] else RED
        y = pdf.get_y()
        pdf.set_fill_color(*color)
        pdf.ellipse(pdf.l_margin, y + 1.2, 5.5, 5.5, style="F")
        pdf.set_x(pdf.l_margin + 9)
        pdf.set_font(FONT, "B", 10.5)
        pdf.set_text_color(*INK)
        status = "not applicable" if g is None else "passed" if g["passed"] else "FAILED"
        pdf.cell(110, 6, GATE_LABELS[gate])
        pdf.set_text_color(*color)
        pdf.cell(0, 6, status, align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_x(pdf.l_margin + 9)
        detail = GATE_EXPLANATIONS[gate]
        if g is not None:
            detail += f"  Measured {fmt(g['value'])}, threshold {fmt(g['threshold'])}."
        pdf.set_font(FONT, "", 8.5)
        pdf.set_text_color(*MUTED)
        pdf.multi_cell(0, 4.3, detail, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1.5)

    pdf.h2("Headline metrics")
    metrics = report["evaluation"]["metrics"]
    names = headline_metrics(metrics)
    if not names:
        pdf.para("No metrics were recorded by the evaluation stage.", color=MUTED)
        return
    rows = []
    for name in names:
        row = [name]
        for variant in VARIANTS:
            m = metrics.get(variant, {}).get(name)
            if m is None:
                row.append("–")
            elif m.get("ci_low") is not None and m.get("ci_high") is not None:
                row.append(f"{fmt(m['value'])} ({fmt(m['ci_low'])}–{fmt(m['ci_high'])})")
            else:
                row.append(fmt(m["value"]))
        rows.append(row)
    pdf.simple_table(["Metric", *(VARIANT_LABELS[v] for v in VARIANTS)], rows,
              [42, 34, 34, 34, 34], size=7.5)
    pdf.para("Values in brackets are 95 % bootstrap confidence intervals. “Search” = the "
             "model answers with retrieved passages from your (privacy-filtered) documents.",
             size=7.5, color=MUTED, h=4)


def _page_examples(pdf: _Scorecard, examples: Sequence[Example], secrets: Sequence[str]) -> None:
    pdf.add_page()
    pdf.h1("Example answers")
    pdf.para("Synthetic test questions generated from your privacy-filtered documents "
             "(personal data was masked before generation). Texts are shortened.",
             size=9, color=MUTED)
    if not examples:
        pdf.ln(4)
        pdf.para("No example predictions were available for this run.", color=MUTED)
        return
    for ex in examples:
        if pdf.get_y() > pdf.h - 70:
            pdf.add_page()
        pdf.ln(2)
        pdf.set_draw_color(*RULE)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(1.5)
        pdf.para(f"Q ({ex.suite}): {scrub(ex.question, secrets)}", size=9.5, style="B")
        if ex.reference:
            pdf.para(f"Expected: {scrub(ex.reference, secrets, 200)}", size=8.5, style="I",
                     color=MUTED)
        for variant in VARIANTS:
            if variant in ex.outputs:
                pdf.set_font(FONT, "B", 8.5)
                pdf.set_text_color(*INK)
                pdf.cell(40, 4.6, VARIANT_LABELS[variant] + ":")
                pdf.set_font(FONT, "", 8.5)
                pdf.multi_cell(0, 4.6, scrub(ex.outputs[variant], secrets) or "(empty)", align="L",
                               new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _page_appendix(pdf: _Scorecard, report: dict[str, Any]) -> None:
    pdf.add_page()
    pdf.h1("Appendix")
    training = report["training"] if report["training"].get("present") else {}
    pdf.h2("Training trials")
    trials = training.get("trials") or []
    if trials:
        rows = []
        for t in trials:
            p = t["params"]
            rows.append([str(t["number"]), t["state"], str(p.get("rank")), fmt(p.get("alpha")),
                         fmt(p.get("learning_rate")), str(p.get("target_modules")),
                         fmt(p.get("dropout")), fmt(p.get("replay_ratio")),
                         fmt(t.get("objective")), fmt(t.get("seconds"))])
        pdf.simple_table(["#", "state", "rank", "alpha", "lr", "modules", "dropout", "replay",
                   "objective", "sec"], rows, [8, 20, 12, 14, 20, 20, 16, 16, 24, 28], size=7)
    else:
        pdf.para("No trial records available.", color=MUTED)
    if final := training.get("final_params"):
        pdf.ln(1)
        pdf.para("Final parameters: " + ", ".join(f"{k}={fmt(v)}" for k, v in final.items()),
                 size=8)

    pdf.h2("Loss curves")
    png = loss_chart_png(training.get("loss_curves") or {})
    if png:
        pdf.image(io.BytesIO(png), x=pdf.l_margin, w=pdf.w - pdf.l_margin - pdf.r_margin)
    else:
        pdf.para("No loss curves were recorded.", color=MUTED)

    pdf.h2("Lineage")
    inputs, adapter = report["inputs"], report["adapter"]
    synth = report["synthesis"]
    redaction = report["redaction"]
    lines = [
        ("Run id", report["run_id"]),
        ("Config sha256", report["config_sha256"]),
        ("Seed", str(report["seeds"]["seed"])),
        ("Adapter sha256", adapter["sha256"]),
        ("Adapter size", f"{adapter['size_bytes']:,} bytes in {adapter['n_files']} files"),
        ("Corpus", f"{inputs['n_documents']} documents, sha256 {inputs['corpus_sha256']}"
         if inputs.get("present") else "not recorded"),
        ("Redactions", f"{redaction['counts']['n_redactions']} masked/dropped spans"
         if redaction.get("present") and redaction.get("counts") else "not recorded"),
    ]
    if synth.get("present"):
        lines += [(f"Data {name}", digest) for name, digest in synth["files_sha256"].items()]
    v = report["versions"]
    lines += [
        ("ZLLF / Python", f"{v['zllf']} / {v['python']} on {v['platform'].get('platform')}"),
        ("Libraries", ", ".join(f"{k} {val}" for k, val in v["libraries"].items() if val)),
        ("Models", "; ".join(f"{m['name']} ({'/'.join(m['roles'])}, {m['license'] or '?'})"
                             for m in report["licenses"]["models"])),
        ("Signing key", f"Ed25519 {report['signing'].get('key_id') or '–'}"),
    ]
    for label, value in lines:
        pdf.set_font(FONT, "B", 7.5)
        pdf.set_text_color(*INK)
        pdf.cell(32, 4.2, label)
        pdf.set_font(FONT, "", 7.5)
        pdf.multi_cell(0, 4.2, value, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def render_scorecard(report: dict[str, Any], examples: Sequence[Example], out_path: Path,
                     *, secrets: Sequence[str] = ()) -> int:
    """Write the scorecard PDF for a validated ``report`` dict. Returns the page count."""
    pdf = _Scorecard(run_label=str(report["run_id"]))
    pdf.set_title("Model Readiness Scorecard")
    pdf.set_author("ZLLF appliance")
    pdf.set_creator("zllf")
    created = report.get("created_at")
    if created:
        pdf.set_creation_date(dt.datetime.fromisoformat(created))
    _page_summary(pdf, report)
    _page_examples(pdf, examples, secrets)
    _page_appendix(pdf, report)
    pdf.output(str(out_path))
    return int(pdf.page_no())
