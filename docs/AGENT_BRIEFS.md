# Module briefs for parallel subagents

The lead agent starts one subagent per unfinished module, all in parallel, each with the shared rules
plus its brief. A first attempt ran on the dev Mac and was stopped part-way, so a module folder may
already hold partial work: the subagent reads it first, keeps what is good, and finishes it.

## Shared rules (give these to every subagent)

- **Check before building.** Before writing code, read `docs/research/OPEN_SOURCE_SURVEY.md` and
  search (web, GitHub, PyPI) for an existing, maintained open-source solution to your
  step. Use it, wrap it or port the idea if it fits; write your own only when nothing fits, and
  say in your report what you checked and why you did or did not reuse it.
- Read `CLAUDE.md`, `docs/ARCHITECTURE.md` and `src/kelpod/contracts.py` first.
- Own ONLY `src/kelpod/<module>/` and `tests/<module>/`. Do not edit `contracts.py`,
  `pyproject.toml`, other modules or docs; report needed contract changes or dependencies instead.
- Do not commit, stash or push; the lead does that.
- `uv run python …`, `uv run pytest tests/<module>`. Installed: numpy, opencv-python-headless,
  scikit-image, pypdfium2, pillow, shapely, scipy, networkx, ifcopenshell, rapidocr-onnxruntime,
  pydantic.
- Any open-source licence is fine in this repo (GPL/AGPL code and non-commercial data such as
  CubiCasa5k included): follow its terms and add it to `CREDITS.md` with licence and link.
- No NVIDIA GPU is assumed. Keep each module's tests under ~20 s; mark slower ones
  `@pytest.mark.slow`.
- Other modules are built at the same time: build your own fixtures (draw with cv2/Pillow or construct
  contract objects); use `kelpod.evaluation.synth` once it exists.
- Finish with a short report: what was built, what was measured, limitations, needed changes.

## evaluation (`src/kelpod/evaluation/`, `tests/evaluation/`, `scripts/benchmark.py`)

1. `kelpod.evaluation.synth`: deterministic (seeded) generator of one-storey, rectilinear German-style
   plans (3–8 rooms) following the conventions in `docs/ARCHITECTURE.md` exactly: hatched 36.5 cm
   exterior walls, hatched interior walls, unhatched partitions, doors (gap + leaf + quarter arc),
   windows (gap + 2–3 lines), dimension chains on all four sides (ticks, extension lines, German
   numbers centred above the line, vertical chains rotated 90°), `BRH 0,90` near windows, room stamps
   (`Wohnen` / `14,52 m²`). 1:100 at 300 dpi; permissive font with fallback (DejaVu Sans on Linux,
   Hershey last). Optional degradation (±3° rotation, noise, blur, JPEG, stains, fold, uneven
   background). `generate(seed, degrade=0.5) -> SyntheticPlan` (image, dpi, plan_scale, ground truth
   `PlanGraph` in metres, y up, origin bottom-left of the undegraded image, scale_source
   `ground_truth`, and the list of drawn text annotations); `save(plan, dir)` writes `plan.png`,
   image-only `plan.pdf`, `ground_truth.json`; `python -m kelpod.evaluation.synth --out samples --n 3`.
2. `kelpod.evaluation.metrics`: `closure_metrics(graph)` (self-contained, networkx/shapely) and
   `compare_to_ground_truth(pred, gt)` (Hungarian-matched room IoU, room precision/recall, wall
   precision/recall/F1 at ~10 cm, opening recall by kind, area error; align by bounding-box min corner).
3. `scripts/benchmark.py`: N seeds → generate, save, run `kelpod.pipeline.convert` on the PDF, catch
   errors per plan, metrics + per-step timings, Markdown table + JSON.
4. Tests: determinism, ground truth closed and consistent with stamps, GT-vs-GT perfect, perturbed worse.

## clean (`src/kelpod/clean/`)

`run(path, *, dpi=200, plan_scale=100) -> list[RasterPage]`, plus `load(path, dpi)` and
`clean_image(gray, dpi)`. PDF via pypdfium2 (multi-page), PNG/JPG/TIFF (multi-page TIFF) with DPI
from metadata else 300; grayscale; resample to `dpi`. Deskew within ±10° to ~0.1° (straight plans
unchanged), record `deskew_deg`. Flatten background, suppress stains and faint fold lines, remove
speckles while keeping thin lines and text. Binarise (Otsu/Sauvola) to `ink`. Under ~3 s for A3 at
200 dpi. Tests: loaders, multi-page, deskew accuracy, thin lines kept, timing.

## segment (`src/kelpod/segment/`)

`run(page, *, backend="classical") -> Segmentation`. `Segmenter` protocol + registry, stub
`LearnedSegmenter` (RoomFormer / Detectron2 later, needs weights from real German plans). Classical
baseline on `page.ink`: wall mask (close hatching, open by ~half wall thickness, stroke-width filter,
drop text and dimension chains); vectorise to centreline segments with thickness, merge collinear;
classify exterior / interior / drywall by thickness and hatching; openings as gaps between collinear
walls bridged by thin lines (door if quarter arc, with hinge; window if parallel lines); rooms as
enclosed free-space components after closing openings (polygon, seed, area). A few seconds per A3
page. Tests on fixtures and on synth (scale GT with `m_to_px`, m_per_px = 0.0254/dpi*100).

## ocr (`src/kelpod/ocr/`)

`run(page, *, backend="rapidocr") -> OcrAnnotations`. `OcrEngine` protocol + registry with
`rapidocr` (default) and `none`; hook for docTR/PaddleOCR later. Read 90°-rotated text (run on the
rotated page too, map boxes back, de-duplicate, set `angle_deg`). Pure, well-tested parsers: German
numbers (`3,51`, `0,885`, `88,5`/`76` as cm, superscript mm `1,01⁵`, O→0, l→1), `BRH 0,90` /
`BRH 87,5` / `Brh.=90`, room stamps (room-name list, umlaut-tolerant, pair with nearby
`14,52 m²` / `qm` / `m2`). Other numbers → `DimensionLabel`; find its dimension line and the ticks
around the text and fill `line`. Under ~30 s per A3 page. Tests: parsers, rapidocr on rendered
horizontal and rotated numbers (slow), tick association.

## geometry (`src/kelpod/geometry/`)

`run(page, seg, ocr) -> PlanGraph` and `check(graph) -> ConsistencyReport`. (1) Scale from plan scale,
then robustly from dimensions with `line` spans (`scale_source="dimensions"`). (2) Snap to 0/90°
(45° if close), merge collinear pieces, cluster parallel wall coordinates, extend/trim to junctions,
split at T-junctions, drop spurs, merge nodes; bridge opening gaps and attach openings with
`offset_m`/`width_m`. (3) Least-squares fit of clustered axis coordinates to dimensions (faces or
centrelines, pick the better fit), residual into the report. (4) Rooms by polygonize on inner faces,
CCW; names/areas from stamps; sill/height from BRH or defaults (window sill 0.90, height 1.385; door
0 / 2.01). (5) NetworkX checks: components, dangling nodes, node/room closure, plain-language issues,
stamp-area mismatches > 10 %. Robust to missing OCR and spurious segments. Tests: noisy rectangle house
closes exactly, T-junction split, gap bridging, scale from dimensions, 3 % length error fixed to 1 cm,
check flags dangling walls and a second component.

## export (`src/kelpod/export/`)

`build_model(graph, *, storey_height_m=2.60, name)` → one storey `EG`, defaults for opening heights.
`write_ifc(model, path, *, schema="IFC4"|"IFC2X3") -> ExportResult` with `ifcopenshell.api`:
Project → Site → Building → Storey, metre units, Body + Axis contexts, walls as swept solids
(IFC2X3: IfcWallStandardCase), opening elements voiding walls and filled by IfcDoor / IfcWindow, IfcSpace
per room with LongName and Qto areas, optional floor slab, `Pset_WallCommon.IsExternal`, closed
corners (extend by half the meeting wall's thickness or IfcRelConnectsPathElements). Re-open, count
entities, `ifcopenshell.validate`, build all geometry with `ifcopenshell.geom`; problems → warnings.
Tests for both schemas. List open questions for the HottCAD import (e.g. IfcRelSpaceBoundary).

## research (`docs/research/OPEN_SOURCE_SURVEY.md` only; no code)

A researcher subagent with web access. Survey existing open-source solutions for each pipeline step
and for the whole problem (raster floor plan → vector → BIM/IFC): floor-plan recognition models and
repos (RoomFormer, FloorplanTransformation, Raster-to-Graph, HEAT, PolyRoom, FRI-Net, CubiCasa
tools, DeepFloorPlan and its forks, plan2scene, …), datasets (CubiCasa5k, R2V, RPLAN, Structured3D,
MSD, ResPlan, any German/European plans), OCR for rotated dimension text, vectorisation and
constraint-fitting tools, IFC writers and validators, and plan-to-BIM products or services.
For each: what it does, input/output, licence of code **and** weights **and** training data (verify
on the repo, so we can credit it correctly), maturity (last commit, stars, maintained?), whether it
runs without an NVIDIA GPU, and how Kelpod could use it (use as is, fine-tune on German plans,
borrow ideas only, avoid). End with a ranked shortlist of what to adopt next and what to avoid,
with reasons. Cite URLs. Mark anything not verified as unverified.

## finetune (`training/`, `src/kelpod/segment/learned.py`, `tests/training/`; starts after integration)

Goal: replace the classical segmentation baseline with a fine-tuned state-of-the-art model, plugged
in through the `Segmenter` protocol as `backend="learned"`, and show with numbers whether it beats
the baseline. Licences are no obstacle in this repo (see `CLAUDE.md`); credit everything in
`CREDITS.md`.

1. **Check first:** read `docs/research/OPEN_SOURCE_SURVEY.md` and confirm the current state of the
   art for raster floor plan → walls / openings / room polygons (e.g. Raster2Seq, SIGGRAPH 2026, MIT
   code, CubiCasa5k weights on Hugging Face; fpvec-lab's detection readout; Detectron2 Mask R-CNN or
   RF-DETR as a detection baseline). Pick the candidate with released weights, working code and the
   best reported wall/room scores; write the choice and reasons in `training/README.md`.
2. **Zero-shot baseline:** run the pretrained weights on 50 synthetic plans from
   `kelpod.evaluation.synth` and score them with `kelpod.evaluation.metrics` against the ground
   truth and against the classical baseline.
3. **Data:** (a) our generator (unlimited, exact labels, German conventions); (b) Modified Swiss
   Dwellings / Swiss Dwellings (CC BY 4.0) re-rendered in our DIN 1356-1 style by the generator's
   renderer; (c) CubiCasa5k (CC BY-NC-4.0, credit it); (d) ResPlan / ResPlan-FP (CC BY 4.0, 17k
   vector plans, re-rendered like (b)); later (e) the pilot's real plans. Use all of the datasets found
   in the survey and in this brief, and credit each in `CREDITS.md`. Keep a
   fixed held-out test split of synthetic and, if possible, SFS-A68 (German/Austrian/Swiss plans,
   room-function labels only) for a reality check.
4. **Fine-tune** on the cluster with Slurm (`sbatch`, scripts in `training/slurm/`; ampere A100 40 GB
   / RTX A6000 48 GB, max 2 days; volta V100 32 GB, max 4 days; interactive sessions are limited to
   2 h). Log metrics per epoch; keep checkpoints out of git (`/models/`, or a path under `~/`).
5. **Integrate:** `LearnedSegmenter` loads the checkpoint (path from an env var or argument), runs on
   CPU too (slowly), and returns a `Segmentation`. The geometry step stays deterministic.
6. **Report** in `training/README.md`: zero-shot vs fine-tuned vs classical on the same test split
   (wall F1, room IoU, opening recall, closure after geometry, runtime), GPU hours used, and what
   real German plans would add. Never present targets as results.
