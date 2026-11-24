# Hand-over: MVP milestone 1 (plan to closed IFC on 3 sample plans)

Read `KELPOD_CONTEXT.md`, `CLAUDE.md` and `docs/ARCHITECTURE.md` first. Work on branch
`claude/kelpod-mvp-fa1abc`; commit and push about once an hour; `main` only by promotion (`docs/EVAL_PROTOCOL.md`).

## Definition of done

The pipeline runs end to end on at least 3 synthetic sample plans, writes IFC files that IfcOpenShell
re-opens, the closure metric and runtime are reported honestly, and `uv run pytest` passes.

## Next up, right after the experiment in progress (user, 2026-11-24 13:35 CET)

The user wants these three set up and run by the agents here on pascal (not on the Mac). Finish
the experiment that is in progress (e.g. `exp/ocr-chainsum`), then do these, in parallel where
possible (Sonnet subagents, one per task):

### A. Make Kelpod work on real DIN 1356-1 plans from the web ("do the max you can")
The user tested a real, modern DIN-style plan from Google Images in the sandbox and the result was
bad. Files: `~/kelpod-data/real/din-web-2026-11-24/` (licence unchecked: keep out of the repo).
Worktree `~/kelpod-wt/din-web` on branch `exp/din-web-plans` (from main) is already created for it.
Measured with main on 2026-11-24 (default settings):
- `efh-erdgeschoss-550.jpg` (550 x 550 px, single-family house, ground floor): 23 walls, **0 rooms**,
  13 dangling ends, 4 components; the 16 dimension labels "disagree", so the scale was guessed.
- `positionsplan-fig134.png` (685 x 666 px, structural position plan): 18 walls, **0 rooms**.
Diagnosis (overlay of the classical segmenter on the first plan):
1. Walls are drawn as two thin parallel lines with an empty or lightly filled interior; the wall
   mask misses them and grabs furniture, stairs, the hatched terrace and text instead.
2. Very low resolution (a 24 cm wall is about 2 px). Upscale (Lanczos/bicubic) and sharpen (unsharp
   mask) before segmentation and OCR; consider super-resolution if justified.
3. OCR misreads centimetre chains with raised millimetre digits: "178⁵" read as "2865", "124⁵" as
   "1245". Detect the unit per plan (cm integers vs. m with comma), use chain sums (total = sum of
   parts) to correct digits, and read stamps like "20,91 m²" that came out as "2091".
4. 45-degree walls, bay shapes, lots of furniture and stair lines.
Do the most you can: wall detection from parallel line pairs (any angle), upscale + sharpen + OCR,
cm/superscript parsing with chain-sum checks, clutter suppression. Measure per plan before/after and
keep the synthetic benchmark from regressing; promote via the protocol. Report honestly in README.

### B. Keep the Streamlit sandbox in sync with main
The sandbox (https://kelpod.streamlit.app, code in `deploy/streamlit/`) currently installs kelpod
from a pinned commit (`requirements.txt`). Change it so the app imports kelpod from the repo
checkout it is deployed from (`sys.path` to `src/`, also for the worker subprocess via PYTHONPATH),
and `requirements.txt` lists only the third-party dependencies. Merge that into main (deploy-only
change, no pipeline change). After that, every push to main redeploys the sandbox automatically.
Note for the user in README: the Streamlit app must be switched to deploy from `main` once (the
user or the Mac session does that in the Streamlit UI; agents here cannot log in there).

### C. Collector for real DIN-style plan images
A draft is in `~/kelpod-drafts/collector/` (unfinished, untested; written on the Mac and stopped).
Finish it as `scripts/collect_plans.py` + `tests/collect/`: official APIs only, never HTML-scraping
Google or any CAPTCHA workaround: Openverse and Wikimedia Commons (no key, licences included) and
the Google Programmable Search JSON API (image search) only when `GOOGLE_CSE_KEY` and
`GOOGLE_CSE_CX` are set (100 free queries/day; the user would create the key). German queries
("Grundriss Erdgeschoss Bemaßung", "Bestandsplan Grundriss", "Grundriss DIN 1356", ...), polite
rate limits, robots.txt, a manifest with source, licence and URL per image, sha256 + dHash dedupe,
a cheap "is this a plan" filter. Run it as a small Slurm job (1 CPU) into `~/kelpod-data/web/`
(never into the repo), then use the plans that pass as a real dev set for task A (no ground truth:
report closure and visual checks; hand-label a few if useful).

## Queued, not urgent (user, 2026-11-24 13:15 CET): make Kelpod work on real historic plans

**Not a priority.** The user asked to keep working on what was in progress before (fine-tuning,
evaluation, continuous improvement) and to take this up afterwards, as one of the normal
improvement items. The measurements below stay valid.

The user uploaded three real German plans to the sandbox and the results were bad. When the current
work is done, improve Kelpod for this type of plan. Files (licence unchecked, keep them out of the
repo): `~/kelpod-data/real/user-feedback-2026-11-24/` (README.txt describes each).

Measured with the current pipeline (160 dpi, classical segmenter) on 2026-11-24:
- villa-garten-vorhof.jpg: 81 walls, 5 rooms, 3 components, 25 dangling ends, 25 loose pieces dropped.
- bauernhaeuser-oberbayern-1900-tafel71.jpg: 7 walls, 0 rooms, 7 openings not placed on any wall.
- villa-gundel-drempelgeschoss.jpg: 6 walls, 0 rooms, no dimension read.
- Scale fell back to "1:100 at the file's dpi" on all three: meaningless for web JPEGs.

What these plans have that our synthetic DIN-style plans do not (each is a work item):
1. Walls drawn solid black (poché) or as thin double lines with black pillars; no hatching.
2. Scale from a graphic scale bar (plans 1, 2) or from dimensions written INSIDE rooms with arrowheads
   and dotted lines (plan 3: "4,35", "3,00"); no outside Maßketten; unknown scan dpi.
3. Several plans on one sheet (plan 2: 7 floor plans, captions "ERDGESCHOSS", "I. STOCK", Fraktur
   text): detect and split plans, convert each (or let the user pick one).
4. Non-orthogonal walls: bay windows, octagonal rooms, curved stairs.
5. Clutter that is not walls: stairs, bathtubs, stoves, furniture, columns, garden/courtyard outlines.
6. Room names in capitals split across lines with hyphens ("SPEISE-/S.", "ANRICHTE-/Z."), old
   abbreviations (Schlafz., Fremdenz., Frühstz.); exterior spaces (Garten, Vorhof, Altan) are not rooms.
7. Low resolution (540-1024 px wide JPEGs).

How: research first (what handles solid-wall historic plans: CubiCasa5k-trained models, scale-bar
detection, plan-region detection). Then (a) add a "historic" style to the synthetic generator
(poché walls, scale bars, multi-plan sheets, interior dimension arrows, low res) and train/fine-tune
on it; (b) label these 3 plans by hand-assisted annotation (walls + rooms) as a tiny real-dev set
and report per-plan numbers before/after; (c) collect more openly licensed historic German plans
(Wikimedia Commons, LABW, Wien Museum) for real-dev. Use Sonnet subagents in parallel, GPUs via
Slurm as justified. Report honestly: these are 3 plans, not a benchmark. When something improves,
tell the user in README status which of the 3 plans now work.

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
- [x] Closure fixes in classical segment (2026-11-24, commit after a270ff2): held-out synth seeds
  100-129 (not tuned on), before -> after: degrade 0.5 closed 0.733 -> 0.800, room IoU 0.936 ->
  0.962; degrade 1.0 closed 0.067 -> 0.533, room IoU 0.774 -> 0.869 (docs/results/closurefix_*.md).
  Synthetic only; dev numbers, not a locked evaluation.
- [x] Fine-tune a state-of-the-art segmentation model (done 2026-11-24, e19fb69): ResNet-34 U-Net
  (smp) + deterministic vectoriser, `backend="learned"` / `kelpod convert --segment-backend learned`
  (checkpoint `~/kelpod-models/kelpod-seg/r34/model.pt` via KELPOD_SEG_CHECKPOINT; needs torch, the
  training env has it). 1.01 GPU h. Dev/val numbers, synthetic and re-rendered only: val-synth
  correct-closed classical 0.720 -> learned 0.980 (edit cost 4.14 -> 0.45); synth test seeds closed
  0.88/0.70 -> 1.00/1.00 at degrade 0.5/1.0; Swiss/ResPlan/CubiCasa5k room IoU 0.97/0.98/0.86.
  Zero-shot CubiCasa5k model clearly worse on our style. Classical stays default; switching is the
  first exp/* experiment (needs 2 training seeds, test-synth held-out style unmeasured, and
  locked_eval must be able to pass a checkpoint / torch env to the candidate). See training/README.md.
- [x] (step 0) `eval/protocol-v1` built and merged into this branch (cc4518a, no pipeline change):
  held-out generator style (default output byte-identical), frozen val-synth (200, train style,
  degrade 0-1.2) and test-synth (300, held-out style) in ~/kelpod-data/eval/ with SHA-256 manifests
  in `eval/manifests/`, fresh split per promotion, `scripts/locked_eval.py` (input-only temp dir,
  panel, paired bootstrap, promotion rule, 2-run budget, TEST_LOG), overlap/pHash checks; the gate
  now commits the TEST_LOG line. Smoke run of the current baseline on val-synth (200, synthetic):
  correct-closed 0.525, closed 0.585, room IoU 0.901, wall F1 0.974, doors P/R 0.93/0.94, 0 crashes.
  Pending: test-ood splits (need a committed renderer writing full GT graphs), real-dev / test-real
  (no files in ~/kelpod-data/real yet), OCR panel metrics (pipeline result exposes no OCR output).
  Open protocol points are listed in eval/README.md.
- [x] Minimal FastAPI endpoint (`kelpod.api`): `GET /health`, `POST /convert` (multipart upload,
  extension + magic-byte check, 415/413/422, pipeline in a worker thread), `GET /jobs/{id}` and
  `GET /jobs/{id}/model.ifc`; jobs keep-last-N + TTL, temp dir removed on shutdown. Run with
  `uv run --extra api kelpod serve` or `python -m kelpod.api`. 12 tests (1 slow). No auth yet;
  the size limit is checked while reading, so a reverse proxy should also cap request size.
- [x] Final report in `docs/MVP_REPORT.md`: what works, what doesn't, what needs real plans (2026-11-24).
- [x] baseline-v0 on main (40546a9, 2026-11-24): MVP + eval protocol v1 merged, tagged, pushed.
- [ ] Continuous improvement (approved 2026-11-23; never checked, runs until a person stops the loop):
  2026-11-24 14:10: eval/learned-runner and a load-aware OCR timing test merged into main.
  Historic plans (top priority) measured: classical 9/1/2 rooms, learned (r34) 4/1/0 rooms, none
  closed, scale guessed. Now: exp/historic-plans (labels for the 3 plans, scale bar, interior
  dimensions, multi-plan sheets, poché walls), exp/learned-v1 redirected to a historic training
  style, real-scan collection. exp/ocr-chainsum paused (WIP pushed).
  Earlier, 2026-11-24 12:50: exp/learned-v1 (style-randomised U-Net, 2 seeds, learned default),
  exp/ocr-chainsum (joint Maßketten decoding), eval/learned-runner (locked_eval for learned
  checkpoints), real-scan collection (~/kelpod-data/real, not yet written). Worktrees ~/kelpod-wt/.
  see "Continuous improvement" below.

## Continuous improvement (after the MVP items)

The user wants Kelpod to keep getting more accurate, with experiments running in a loop until they
say stop. The whole pascal cluster may be used when justified (justification and GPU hours per job in
`docs/EXPERIMENTS.md`), and any online data may be used (credit it in `CREDITS.md`; note what the
paid version could not use). Research and ranked bets: `docs/research/V2_RESEARCH.md`.

**Evaluation and promotion follow `docs/EVAL_PROTOCOL.md`.** Order of work:
0. On an `eval/protocol-v1` branch: build the frozen splits and manifests (`eval/`), the held-out
   test style in the generator, `scripts/locked_eval.py` (panel metrics, paired bootstrap, the
   promotion rule, `--verify-manifests`, appends to `eval/TEST_LOG.md`, runs the pipeline on an
   input-only temp dir), and its tests. Merge it into `main` together with the finished MVP: that is
   `baseline-v0`. Start the real-scan benchmark (RealBench-DE) as soon as real plans are available.
   **Real scans:** `~/kelpod-data/real/dev/` (pages + `sources.csv` with licences; outside the repo,
   never commit them) is the real dev set; `~/kelpod-data/sealed/real-test/` is held out (EVAL_PROTOCOL
   test-real candidates: never open per item). If `~/kelpod-data/real/sources.csv` does not exist,
   collect 40-80 real floor-plan pages first (German/DACH Bestandspläne preferred; licence per file).
   A proxy-metric runner (closure, scale from dimensions, OCR stamp area vs. computed room area,
   overlays) is in `~/kelpod-jobs/run_real.py`; first results of the MVP snapshot will be in
   `~/kelpod-runs/real-dev-bcd01f1/`. Baseline synthetic numbers (200 plans each at degrade 0,
   0.5, 1.0): `~/kelpod-runs/baseline-bcd01f1/`.
1. Then each experiment on its own `exp/<name>` branch and worktree, from `main`; promote with
   `scripts/promotion_check.sh exp/<name>`, merge `--no-ff`, tag `baseline-vN`, push; the next
   experiments start from the new `main`.

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

**Resource audit (2026-11-24, 12:40 CET).** Measured over 14 h: the agent job held 16 CPUs / 48 GB
but used ~1.1 CPUs on average and 17.5 GB at peak, so it now asks for 8 CPUs / 24 GB (it switches
itself to the new size at its next usage-limit pause). Cancelled: `kelpod-real` (held 4 CPUs / 8 GB
for 13.5 h only waiting for `sources.csv`; submit `~/kelpod-jobs/real-scans.sbatch` once real/dev
exists, it no longer waits) and the `kelpod-cockroach` watchdog job (2 CPUs idle; the cron entry
covers it). Rule in `CLAUDE.md` ("Cluster resources").

The MVP runs on CPU. If a step needs a GPU (e.g. fine-tuning or running a learned segmenter, a
heavier OCR model), use the cluster's GPUs through Slurm from the login host; never run GPU jobs on
the login node itself. Partitions seen on 2026-11-23: `pascal` (GTX 1080 Ti), `volta` (V100),
`ampere` (RTX A6000, A100). Example: `srun -p ampere --gres=gpu:1 --time=02:00:00 uv run python …`
or an `sbatch` script under `scripts/slurm/`. Keep jobs small and release GPUs when done.
