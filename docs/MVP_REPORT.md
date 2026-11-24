# Kelpod MVP report (milestone 1)

Status: 2026-11-24. **Every number in this report comes from synthetic plans** made by our own
generator (`kelpod.evaluation.synth`). Nothing has been measured on real *Bestandspläne* yet. These
numbers show that the pipeline works on drawings that follow our conventions. They do not show how
it does on real scans.

## What works

- **End to end:** `kelpod convert plan.pdf -o model.ifc` (or `POST /convert` on `kelpod serve`) runs
  six steps: clean, segment, OCR, geometry, model and export. It writes IFC4 or IFC2X3, which
  IfcOpenShell re-opens, validates and builds geometry for. Every run also produces a plain-language
  check report (dangling walls, open rooms, stamp-area mismatches, dimension fit).
- **Steps** (details in `docs/HANDOFF.md`):
  - **clean:** PDF/TIFF/PNG/JPG, deskew (max error 0.015° over ±10°), fold and stain suppression.
  - **segment:** classical baseline. Hatched exterior and interior walls, partitions, doors (arc),
    windows (parallel lines), rooms.
  - **ocr:** RapidOCR on 0° and 90° passes, German number, BRH and room-stamp parsers, ticks
    associated with dimension lines.
  - **geometry:** scale from the dimension chains, snapping, T-splits, a least-squares fit to
    the dimension chains, closed rooms, NetworkX checks.
  - **export:** closed corners, openings, spaces with areas, IsExternal.
- **Viewer:** `scripts/demo.py --seed N` writes the input plan next to the 3D IFC model.
- **Tests:** 298 passing (`uv run pytest`, about 2 min on the CPU node).

## Measured on synthetic plans

Pipeline after the segmentation closure fixes (fe5a361). The seeds below were never used for
tuning. CPU node, one plan at a time.

| Synthetic set | Plans | Fully closed | Room IoU | Room recall | Wall F1 | Door recall | Window recall | Area error | s/plan |
|---|---|---|---|---|---|---|---|---|---|
| seeds 100-129, degrade 0.5 | 30 | 0.800 | 0.962 | 0.983 | 0.994 | 0.986 | 1.000 | 0.017 | 7.9 |
| seeds 100-129, degrade 1.0 | 30 | 0.533 | 0.869 | 0.907 | 0.960 | 0.922 | 0.952 | 0.062 | 5.8 |

The same seeds before the fixes closed 0.733 and 0.067 of the plans, at room IoU 0.936 and 0.774.
The source tables are in `docs/results/closurefix_*.md`.

**Evaluation protocol v1** (`docs/EVAL_PROTOCOL.md`, `scripts/locked_eval.py`): smoke run of
the current baseline on the frozen val-synth split. It has 200 plans in the training style, at
degrade 0 to 1.2.

- Correct-closed: 0.525. A plan counts as correct-closed when it is closed, every room has
  IoU ≥ 0.85, the room count is right and the area error is ≤ 3 %.
- Closed: 0.585.
- Room IoU 0.901, wall F1 0.974, door precision/recall 0.93/0.94, window precision/recall 0.96/0.98.
- No crashes.
- Under full node load: median about 24 s per plan, p95 about 38 s.

A 5-plan smoke run on the held-out test style scored much lower: correct-closed 0.2, room IoU 0.64.
Five plans are too few to conclude anything, but they suggest the classical segmenter is tuned to
our training-style drawings.

Runtime is 5 to 9 s per plan when the node is not loaded. The target is under 3 minutes. OCR takes
about 75 % of that time.

## Learned segmentation (fine-tuned)

A U-Net with an ImageNet ResNet-34 encoder predicts five classes: background, wall, window, door,
room. A deterministic vectoriser turns the masks into walls, openings and rooms. Training took
1.01 GPU hours on one RTX A6000. Training data:

- 4000 synthetic pages;
- Swiss Dwellings (CC BY 4.0) and ResPlan (CC BY 4.0), re-rendered in our drawing style;
- CubiCasa5k (CC BY-NC 4.0).

The model is available as `--segment-backend learned` (checkpoint from `KELPOD_SEG_CHECKPOINT`,
needs PyTorch). **Classical stays the default** until the switch passes the promotion protocol.
Details are in `training/README.md`.

These are dev and val numbers on synthetic and re-rendered plans. They are not from a locked run,
and none of them come from real German plans.

| Set | Metric | Classical | Zero-shot CubiCasa5k | Fine-tuned |
|---|---|---|---|---|
| val-synth (200, frozen) | correct-closed | 0.720 | n/a | 0.980 |
| val-synth | edit cost (edits/plan) | 4.14 | n/a | 0.45 |
| synth test seeds, degrade 0.5 (50) | closed / room IoU | 0.88 / 0.970 | 0.36 / 0.883 | 1.00 / 0.983 |
| synth test seeds, degrade 1.0 (50) | closed / room IoU | 0.70 / 0.937 | 0.16 / 0.892 | 1.00 / 0.980 |
| Swiss Dwellings, held-out sites (96) | room IoU (mask level) | 0.31 | 0.68 | 0.97 |
| ResPlan test (100) | room IoU (mask level) | 0.60 | 0.54 | 0.98 |
| CubiCasa5k test (first 100) | room IoU (mask level) | 0.20 | 0.84 | 0.86 |

Caveats:

- **Training style:** only the default synthetic drawing style was used. The held-out test style is
  likely to cost accuracy, and it has not been measured.
- **CubiCasa5k:** these are real but Finnish plans, and the fine-tuned model is only slightly ahead
  of zero-shot there.
- **CPU runtime:** the segment step takes 3.4–5.4 s per page alone, against 0.3–0.5 s for
  classical.
- **Seeds:** there is one training seed only; promotion needs two.

## What does not work yet

- **Degraded or unfamiliar drawings:** at degrade 1.0 only about half of the plans close. On the
  held-out drawing style, quality drops sharply (preliminary, n = 5).
- **Rooms** must be rectilinear (0°/90°, some 45°). There are no curved walls, no stairs, and only
  one storey per page.
- **OCR:** dimension recall is about 72 % on synthetic plans. Superscript millimetres and
  handwriting are untested on real scans. OCR quality is not yet part of the locked panel.
- **API:** no authentication.
- **HottCAD import is untested.** Open questions for the pilot: IfcRelSpaceBoundary, schema/MVD,
  and the wall-height convention.

## What needs real plans

- A real-scan benchmark (RealBench-DE: 40 or more annotated pages, sealed). Collection of openly
  licensed real plans has started (`~/kelpod-data/real/`, outside the repo). There are no real-scan
  results yet.
- Real scan damage, real hatching variants, handwriting, stamps and coloured plans: the synthetic
  generator only approximates these.
- Fine-tuning on real German plans and expert correction time measured with the pilot. Once expert
  correction time is available, it outranks every automatic metric.
