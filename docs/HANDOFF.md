# Hand-over: MVP milestone 1 (plan to closed IFC on 3 sample plans)

Read `KELPOD_CONTEXT.md`, `CLAUDE.md` and `docs/ARCHITECTURE.md` first. Work on branch
`claude/kelpod-mvp-fa1abc`; commit and push that branch after every finished step; never push `main`.

## Definition of done

The pipeline runs end to end on at least 3 synthetic sample plans, writes IFC files that IfcOpenShell
re-opens, the closure metric and runtime are reported honestly, and `uv run pytest` passes.

## Status (keep this up to date; newest first)

- [x] Old LoRA-factory code deleted (it stays in git history).
- [x] Package skeleton, contracts (`src/kelpod/contracts.py`), pipeline, CLI.
- [x] Research first (rule: check what exists before building anything):
  docs/research/OPEN_SOURCE_SURVEY.md (2026-11-23). Key points: no open tool does scan to closed IFC;
  all published floor-plan weights are GPL or non-commercial; Hottgenroth has its own "Hott-KI" plan
  recognition (check in pilot interviews); keep the classical pipeline, IfcOpenShell, RapidOCR.
- [ ] Modules, one subagent each (briefs in docs/AGENT_BRIEFS.md): clean, segment, ocr, geometry,
  export, evaluation. A first attempt was stopped part-way on 2026-11-23 when work moved to the
  server, so module folders may hold partial, untested WIP: review it, keep what is good.
  - [x] export: IFC4 + IFC2X3 via `ifcopenshell.api`, closed L-corners, openings, spaces, optional
    slab; re-open/validate/geometry checks; 21 tests (hand-built plans only). Open HottCAD questions
    (space boundaries, schema/view, wall height convention) go to the pilot.
  - [x] clean: pypdfium2/Pillow loaders (multi-page PDF/TIFF, DPI fallback 300), background
    flattening, profile deskew (max error 0.015° over ±10°), fold/stain suppression, Otsu plus
    faint-line rescue; ~0.7 s per A3 page at 200 dpi; 24 tests (own fixtures, not yet real scans).
  - [ ] segment, ocr, geometry, evaluation (WIP committed in 5dc6a52, agents finishing).
- [ ] Integration: `kelpod convert` on 3 synthetic plans, `scripts/benchmark.py` report.
- [ ] Fine-tune a state-of-the-art segmentation model (brief "finetune" in docs/AGENT_BRIEFS.md;
  approved 2026-11-23). Starts after integration; GPUs via Slurm.
- [ ] Minimal FastAPI endpoint (`kelpod.api`), if time allows.
- [ ] Final report in `docs/MVP_REPORT.md`: what works, what doesn't, what needs real plans.

## Compute

The MVP runs on CPU. If a step needs a GPU (e.g. fine-tuning or running a learned segmenter, a
heavier OCR model), use the cluster's GPUs through Slurm from the login host; never run GPU jobs on
the login node itself. Partitions seen on 2026-11-23: `pascal` (GTX 1080 Ti), `volta` (V100),
`ampere` (RTX A6000, A100). Example: `srun -p ampere --gres=gpu:1 --time=02:00:00 uv run python …`
or an `sbatch` script under `scripts/slurm/`. Keep jobs small and release GPUs when done.
