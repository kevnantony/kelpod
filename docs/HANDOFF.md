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
- [x] Modules, one subagent each (briefs in docs/AGENT_BRIEFS.md): clean, segment, ocr, geometry,
  export, evaluation. A first attempt was stopped part-way on 2026-11-23 when work moved to the
  server, so module folders may hold partial, untested WIP: review it, keep what is good.
  - [x] export: IFC4 + IFC2X3 via `ifcopenshell.api`, closed L-corners, openings, spaces, optional
    slab; re-open/validate/geometry checks; 21 tests (hand-built plans only). Open HottCAD questions
    (space boundaries, schema/view, wall height convention) go to the pilot.
  - [x] clean: pypdfium2/Pillow loaders (multi-page PDF/TIFF, DPI fallback 300), background
    flattening, profile deskew (max error 0.015° over ±10°), fold/stain suppression, Otsu plus
    faint-line rescue; ~0.7 s per A3 page at 200 dpi; 24 tests (own fixtures, not yet real scans).
  - [x] evaluation: seeded German-style synth generator (synth plans only, rectangular rooms),
    metrics (closure, Hungarian room IoU, wall F1 @10 cm, opening recall; best-of shift alignment),
    `scripts/benchmark.py`; 16 tests. First 10-plan run (modules still in flux): 10/10 ran,
    7/10 fully closed, room IoU 0.935, ~14 s/plan (OCR 80-90 %). Synthetic only.
  - [x] ocr: RapidOCR (bundled models, offline) on 0°/90° passes with line removal and 1600 px
    tiles; German number/BRH/area/room-name parsers; dimension-line and tick association with a
    page-wide px/m check. Synth seeds 0-4: dimension recall 72 % (precision 78 %), linked spans 88 %
    within 4 px of the ticks, 24/25 room names; 11-17 s per A3 page. 106 tests. Synthetic only.
  - [x] geometry: scale from dimension spans (fallback plan scale), 0/90/45° snapping, parallel
    clustering, gap bridging, extend/trim, T-splits, spur removal, least-squares fit to dimension
    chains (faces vs centrelines), shapely rooms, NetworkX checks; never raises. Simulated synth
    input: 100/100 closed at 1.5 px noise (room IoU 0.99), 75 % at 3 px; 3 % drawing error fixed to
    0.18 % area error. 41 tests. Synth window head aligned to 2.285 m (0.90 + 1.385).
  - [x] segment: classical baseline (wall mask, centreline vectorisation, thickness classes, door
    arcs / window lines in gaps, rooms as enclosed free space); `Segmenter` registry, stub learned
    backend; 12 tests. `tests/segment/synth_eval.py`, synth seeds 0-4 (degrade 0 and 0.5): wall
    recall 0.997, precision 1.00, doors/windows 1.00, rooms 1.00 recall, room IoU 0.97, ~0.3 s/page;
    seeds 5-9 at degrade 1.0: wall recall 0.94, door recall 0.81, room recall 0.72, room IoU 0.93.
    Synthetic only.
- [x] Integration: `kelpod convert` on 3 synthetic plans, `scripts/benchmark.py` report.
  When it works, also run `uv run python scripts/demo.py --seed N` (no `--ground-truth`) for the
  same 3 plans: it writes `out/demo/seed_N/viewer.html` (input plan next to the 3D IFC model; the
  viewer is done, see `src/kelpod/viz`). The user wants to see these. Known generator bug for the
  evaluation module: superscript millimetre digits (e.g. `36⁵`) render as empty boxes because the
  font lacks the glyphs; draw them as small raised digits instead. (Fixed: drawn as raised digits
  with both fonts, tested against an empty-box render.)
  Result (2026-11-23, pascal-node02 CPU): `kelpod convert` on synth seeds 0-2 writes IFC4 that
  re-opens; seeds 0 and 1 closed, seed 2 not (Küche not enclosed, 2 dangling wall ends).
  `scripts/benchmark.py --n 10 --degrade 0.5` (docs/results/benchmark_synth10_degrade05.md):
  10/10 ran, 8/10 fully closed (seeds 2 and 9 not), room IoU 0.923, room recall 0.938, wall F1
  0.997, doors 0.986, windows 0.958, 4.7 s/plan mean (OCR ~75 %). Demo viewers written to
  `out/demo/seed_{0000,0001,0002}/viewer.html` (git-ignored; regenerate with scripts/demo.py).
  Fixed: ONNX Runtime affinity-error spam under Slurm (rapidocr ignores per-module thread keys).
  Synthetic only.
- [ ] Fine-tune a state-of-the-art segmentation model (brief "finetune" in docs/AGENT_BRIEFS.md;
  approved 2026-11-23). Starts after integration; GPUs via Slurm.
- [x] Minimal FastAPI endpoint (`kelpod.api`): `GET /health`, `POST /convert` (multipart upload,
  extension + magic-byte check, 415/413/422, pipeline in a worker thread), `GET /jobs/{id}` and
  `GET /jobs/{id}/model.ifc`; jobs keep-last-N + TTL, temp dir removed on shutdown. Run with
  `uv run --extra api kelpod serve` or `python -m kelpod.api`. 12 tests (1 slow). No auth yet;
  the size limit is checked while reading, so a reverse proxy should also cap request size.
- [ ] Final report in `docs/MVP_REPORT.md`: what works, what doesn't, what needs real plans.
- [ ] Continuous improvement (approved 2026-11-23; never checked, runs until a person stops the loop):
  see "Continuous improvement" below.

## Continuous improvement (after the MVP items)

The user wants Kelpod to keep getting more accurate, with experiments running in a loop until they
say stop. GPU compute on this cluster is available for it (shared: at most 4 GPUs at once), and any
online data may be used (credit it in `CREDITS.md`; note what the paid version could not use).

1. **Fixed benchmark first.** Freeze test splits that are never trained or tuned on: hard synthetic
   (degrade 0.5 and 1.0, non-rectangular rooms once the generator has them), rendered Swiss Dwellings
   / ResPlan, CubiCasa5k test, and real German plans as soon as any exist. Report every experiment on
   the same splits with `scripts/benchmark.py`. The headline metric is the share of plans that come
   out fully closed with correct rooms (closure, room IoU, wall F1, opening recall, area error).
2. **Pick the next bet** from `docs/research/V2_RESEARCH.md` (if present) and from what the last
   experiments showed: biggest expected gain per effort first. Ideas: learned segmentation
   (fine-tuned), junction detection plus optimisation, better OCR for Maßketten, larger and more
   realistic synthetic data (DIN-style renders of vector datasets, realistic scan damage),
   multi-hypothesis search scored by the graph check, multi-storey and heights from sections.
3. **Run experiments in parallel** (one subagent per experiment, GPU work via `sbatch`).
4. **Keep or revert:** merge a change only if it beats the current best on the benchmark without
   breaking tests; otherwise revert it and record why.
5. **Log** each experiment in `docs/EXPERIMENTS.md` (newest first): date, hypothesis, change, data,
   GPU hours, results vs. current best, kept or reverted. Update the README status block. Push.
6. Repeat. Every few cycles, re-check the state of the art online.

## Compute

The MVP runs on CPU. If a step needs a GPU (e.g. fine-tuning or running a learned segmenter, a
heavier OCR model), use the cluster's GPUs through Slurm from the login host; never run GPU jobs on
the login node itself. Partitions seen on 2026-11-23: `pascal` (GTX 1080 Ti), `volta` (V100),
`ampere` (RTX A6000, A100). Example: `srun -p ampere --gres=gpu:1 --time=02:00:00 uv run python …`
or an `sbatch` script under `scripts/slurm/`. Keep jobs small and release GPUs when done.
