# Experiments

Hypothesis, change, data, GPU hours (with justification), results, decision. Append only.

## din-web-v2 (2026-11-25, branch exp/din-web-v2, locked run 1 of 2: FAIL on door recall only)

- **Change vs din-web-plans:** footprint guard fixed (in the partial-footprint case only wall and
  window networks that border the footprint and total >= 3 m are kept; loose stubs such as
  dimension extension lines dropped). Root cause found on one val plan (s200009, open outline).
- **Val-synth (200, synthetic), main -> branch:** correct-closed 0.720 -> 0.725, room IoU 0.9401 ->
  0.9443, edit cost -3.7 %, ext wall length error +4.9 % (not flagged), door recall 0.961 -> 0.959.
- **Locked run (test-synth 300 + fresh 200, synthetic):** correct-closed +2.0 pp [-0.3, +4.3]
  (test-synth), +3.0 pp [+1.0, +5.5] (fresh, primary): rules 2 and 4 pass. Rule 3 fails: door
  recall Δ CI [-3.3, -1.1] pp (test-synth) and [-4.5, -1.3] pp (fresh). **Not promoted.**
- **Fix (4be979b):** opening-to-wall attach tolerance 0.3 m (GeometryParams.opening_wall_tol_m);
  the chain-decoded dimension fit can move a thin wall >0.2 m from its detected door gap. Val-synth
  door recall -0.1 pp [-0.3, 0]. Disclosure (agent): while looking for its run folder it listed the
  per-plan ids of the locked run directory once; not used for any decision.
- **Locked run 2 of 2 (2026-11-25):** rule 3 now passes (door recall test-synth -2.0 pp
  [-3.2, -0.9], fresh -1.9 pp [-3.9, 0.0]; runtime +5-7 %). Rule 2 FAILS: fresh correct-closed
  +1.5 pp [-1.0, +4.0] (test-synth +2.3 pp [0.0, +5.0]); room IoU +0.009 [-0.004, +0.021].
  **Not promoted; branch budget used, branch closed.** The gain is real but small; the run-to-run
  difference on fresh (+3.0 vs +1.5 pp) shows its noise. Parts (upscaling, footprint guard,
  attach tolerance, chain OCR) can return inside a branch with a new idea.
- **CPU job for exp/learned-fast:** Slurm 128900, 9 CPUs / 24 GB, <= 4 h, val-synth smoke for both
  seeds vs main (needs its own node so runtime is not distorted by the din-web locked run).

## historic-plans (2026-11-24, branch exp/historic-plans @ d4ef811, includes din-web-plans; not gated)

- **Change:** scale-bar detection (`geometry/scalebar.py`), interior dimension arrows
  (`ocr/interior.py`), sheet splitting (`clean/sheets.py`, `pipeline.convert_all`), historic room
  names, exterior spaces (Garten, Vorhof, Altan, ...) flagged and left out of the IFC.
- **Labels:** the 3 user plans, labelled by the agent from overlays (+-5 px, not an expert),
  outside git. The scale-bar tick value (1 m) was assumed with these plans in view.
- **Real, 3 plans (room IoU / wall F1), main+din-web -> branch, classical:** villa 0.02/0.16 ->
  0.07/0.32; Gundel 0.31/0.59 -> 0.30/0.55; Tafel 71 plan 2 0.23/0.27 -> 0.18/0.41, plan 3
  0.14/0.23 -> 0.29/0.71. Scale error 59 % -> 1-2 % on villa and Tafel.
- **Same branch with the learned v1 segmenter (seed 1, ONNX), measured by the lead:** wall F1
  villa 0.70, Gundel 0.46, Tafel 0.97 / 0.84; room IoU 0.19 / 0.12 / 0.46 / 0.50. The learned model
  finds the walls; rooms stay open (dangling ends at door gaps in solid walls).
- **Synthetic guard:** benchmark seeds 300-319 at degrade 0.5 unchanged; val-synth smoke (40) flags
  ext_wall_length_error (inherited from din-web, being fixed on exp/din-web-v2).
- **Next:** rebase on exp/din-web-v2 once fixed; learned segmenter + door-gap closing in solid
  walls; then gate.

## learned-v1 (2026-11-24, branch exp/learned-v1, not yet run through the gate)

- **Change:** U-Net ResNet-34 retrained with train-only style and degradation randomisation (held-out
  options guarded by a test), synth_v1 / swiss_v1 / resplan_v1 renders, CubiCasa5k; learned is the
  default backend on the branch (no silent fallback); ONNX export.
- **GPU:** kseg-v1-s1 1.86 GPU-h, kseg-v1-s2 1.65 GPU-h (ampere; expected ~4 GPU-h total).
- **Val-synth (200, synthetic):** both seeds correct-closed 0.985 (classical 0.720), edit cost
  0.435, room IoU 0.982, wall F1 0.999. Runtime about 81-84 s per plan under load in the eval job,
  against about 49 s for classical on the same node.
- **Blocker:** protocol rule 3 tolerates at most +25 % runtime, so the current model cannot be
  promoted however accurate it is. Next: `exp/learned-fast` (faster CPU inference: int8 ONNX,
  lower inference resolution, lighter encoder) before the locked run. The runtime rule itself is a
  decision for the founders (asked in README), not changed by the agent.

## din-web-plans (2026-11-24, branch exp/din-web-plans @ 2430913, includes ocr-chainsum; FAIL)

- **Change:** effective-resolution estimate from glyph height + Lanczos/unsharp upscaling of
  low-res inputs, footprint-filter guard, thin double-line wall detector (`segment/pairs.py`,
  used only when classical does not close), plus the ocr-chainsum OCR changes.
- **Real web plans (dev, no labels):** efh 0 -> 5 rooms (not the real rooms), 1 component, scale
  from dimensions; Positionsplan 0 -> 5 rooms (one merged), 1 component. Neither plan correct.
- **Locked run 1 of 2 (2026-11-24 18:09, test-synth 300 + fresh 200, synthetic):** primary (fresh)
  correct-closed +0.005 [-0.010, +0.020]; room IoU +0.047 [+0.025, +0.071] (rule 2 passes via the
  IoU alternative, rule 4 passes); rule 3 FAILS: door precision -3 to -6 pp, edit cost +11 to +24 %,
  exterior wall length error +16 to +69 % on both splits. **Not promoted.**
- **Next:** a new branch with the regression found and fixed on dev/val data only (which component
  costs doors and envelope length), keeping the room IoU gain.

## ocr-chainsum (2026-11-24, branch exp/ocr-chainsum @ 9fe1219, not promoted)

- **Hypothesis:** joint decoding of Maßketten (n-best readings from RapidOCR's CTC output, chain
  sums, pixel spans, shared ticks; CP-SAT) raises dimension recall/precision and, through the
  dimension fit, closure and area accuracy.
- **Data:** synthetic dev seeds 200000-200049 (tuning), 200050-200099 (held-out dev), frozen
  val-synth (200). No test or sealed data looked at. CPU only, 0 GPU-h.
- **Results (val-synth, synthetic, main -> branch):** dimension precision 0.803 -> 0.878, recall
  0.708 -> 0.793, linked to the right span 0.530 -> 0.775; correct-closed 0.720 -> 0.725, room IoU
  0.9401 -> 0.9443, edit cost 4.14 -> 4.00. Runtime +20-26 % per plan (OR-Tools import + solve).
  Real web plan (550 px): area stamps fixed (`2091mF` -> 20,91 m²), chain decoding falls back
  (too few consistent labels at that resolution).
- **Decision:** not run through the locked evaluation. The end-to-end gain on val is far below the
  promotion bar (+3 pp correct-closed, or edit cost -10 %, or room IoU +0.02), so a locked run
  would only spend budget. Kept as the OCR base for `exp/din-web-plans` (cm chains with raised mm
  digits, low-resolution web plans), which will be promoted together if it clears the bar.

## finetune-unet-v0 (2026-11-23, branch claude/kelpod-mvp-fa1abc, training/)

- **Hypothesis:** a 5-class U-Net (background, wall, window, door, room interior; ImageNet
  ResNet-34 encoder, segmentation_models_pytorch) fine-tuned on synth + Swiss Dwellings / ResPlan
  re-rendered in DIN style + CubiCasa5k, followed by the deterministic vectoriser in
  `kelpod.segment.learned`, beats the classical segmenter at heavy degradation (classical at
  degrade 1.0: wall recall 0.94, door recall 0.81, room recall 0.72, synthetic) and the zero-shot
  CubiCasa5k model.
- **Data:** synth default style seeds 1,000,000-1,003,999 (4000 pages); Swiss Dwellings train
  sites (held out by site ID, crc32(site) % 10 in {0, 1} = test/val), up to 3000 floors; ResPlan
  train split, up to 3000; CubiCasa5k official train split (4200). Manifests in
  `training/manifests/`.
- **GPU jobs:**
  - `kseg-r34` (ampere, 1 GPU, 30k iterations x batch 16 at 512 px): expected 2-4 GPU hours.
    Justification: the one fine-tuning run the brief asks for; a ResNet-34 U-Net is small enough
    to also run on the CPU in the pipeline.
  - Evaluation runs on CPUs (pascal partition for free cores; GPU not used, runtime measured on
    CPU): 0 GPU hours of compute, ~1-3 h allocation each.
- **GPU hours used:** training job 128281, 1.01 GPU h (RTX A6000); evaluation jobs 128282,
  128297, 128312, 128313 CPU-only (GPU hidden), roughly 3-4 h of GPU allocation, no GPU compute.
- **Results** (dev/synthetic numbers, not a locked evaluation; details in `training/README.md`):
  - Synth test seeds 100000-100049, after geometry, closed share classical / zero-shot CubiCasa5k
    / fine-tuned: degrade 0.5: 0.88 / 0.36 / 1.00; degrade 1.0: 0.70 / 0.16 / 1.00. Room IoU at
    degrade 1.0: 0.937 / 0.892 / 0.980; door recall 0.950 / 0.993 / 1.000.
  - val-synth (frozen, 200, eval-branch panel), classical vs fine-tuned: correct-closed 0.720 vs
    0.980, edit cost 4.14 vs 0.45, wall F1 0.988 vs 0.999, room IoU 0.940 vs 0.982.
  - Held-out dataset pages (mask level), room IoU classical / zero-shot / fine-tuned: Swiss (by
    site) 0.31 / 0.68 / 0.97; ResPlan test 0.60 / 0.54 / 0.98; CubiCasa5k test (100, real Finnish
    raster plans, in-domain for the zero-shot model) 0.20 / 0.84 / 0.86.
  - CPU runtime of the segmentation step: fine-tuned 3.4-5.4 s per page (single process,
    8 threads) vs classical 0.3-0.5 s.
  - Found and fixed on the way: a vectoriser sliver bug (door next to a perpendicular wall face);
    with oracle class maps, 12/12 dev plans now close after geometry.
- **Caveats:** no real German plans; Swiss/ResPlan are re-rendered in our default style; the
  test-synth held-out style (other fonts, hatching, line weights, damage) was never seen in
  training, so a drop there is likely; one training seed only (rule 5 needs two before promotion).
- **Decision:** the learned backend is available as `backend="learned"` (checkpoint via
  `KELPOD_SEG_CHECKPOINT`). The default stays `classical`. Making `learned` the default is a
  separate experiment (`exp/<name>` branch, second training seed), promoted only via
  `docs/EVAL_PROTOCOL.md`.

## learned-v1 (2026-11-24, branch exp/learned-v1)

- **Hypothesis:** the learned backend (U-Net r34; val-synth correct-closed 0.980 vs classical
  0.720, dev numbers) generalises much better to unseen drawing styles (test-synth's held-out
  style) and to real scans if it is trained with strong style and degradation randomisation; with
  that it can replace classical as the default segmentation backend.
- **Change:** training-only renderer/degrader `training/kelpod_train/{style_v1,render_v1,degrade_v1}.py`:
  hatching variants (45° at 1.3-3.0 mm, 30°, 60°, 0°, double 45°, dashed 45°, stipple, none),
  wall outline on/off, per-element line weights (scale 0.85-1.35), door symbols (arc, leaf only,
  double, sliding, rectangle leaf, dashed arc, gap), window symbols (1-4 lines, frame, sill),
  13 train fonts, clutter (furniture, stairs, axis grid, notes, stamps, title block, north
  arrow), own damage profiles (pencil, ink bleed, bleed-through, paper texture, letterpress),
  selected Augraphy effects (MIT), scan geometry (wobble, page curl, perspective), scan dpi
  150-600, JPEG Q 20-95. **Held out and never sampled:** every option flagged held-out in
  `kelpod.evaluation.synth.styles` (fonts URW Gothic, Nimbus Sans Narrow, Nimbus Mono, Hershey
  duplex/complex; hatching 135° dense 1.0 mm, cross, solid; line weights 0.7 and 1.6; damage
  diazo, photocopy, faded), plus their families/look-alikes (see `style_v1` docstring);
  enforced by `check_not_heldout` at sampling time and by `tests/training/test_style_v1.py`.
  Pipeline: `learned` becomes the default backend (checkpoint from `kelpod-run.toml`, ONNX via
  onnxruntime, no torch in the main env), failing clearly if no checkpoint is available.
- **Data:** existing r34 sources (synth default style 4000, Swiss 1859, ResPlan 2000,
  CubiCasa5k 4200) plus style-randomised re-renders: synth geometry seeds 1,100,000-1,105,999
  (6000), Swiss Dwellings train sites (up to 3000 floors, same site hold-out), ResPlan train
  (3000). Dev-val: synth_v1 seeds 910,000-910,059, Swiss val sites re-rendered (47). Data is
  CPU-built (this node + pascal CPU jobs, no GPU requested).
- **GPU jobs (planned):**
  - `kseg-v1-s1`, `kseg-v1-s2` (ampere, 1 GPU each, ResNet-34 U-Net, 40k iterations x batch 16
    at 512 px, seeds 1 and 2): expected ~2 GPU-h each (r34: 24k iterations = 1.01 GPU-h, plus
    5 val sets every 2000 iterations), time limit 6 h, checkpoint + resume every 2000
    iterations. Justification: rule 5 requires two training seeds; one model of the size that
    already runs on the CPU in the pipeline. Total expected ~4 GPU-h.
- **Selection:** val only (dev-val sets above, frozen val-synth via `kelpod_train.val_synth`).
  test-synth / test-ood / sealed data are not looked at.

## learned-fast (2026-11-25, branch exp/learned-fast = exp/learned-v1 + main)

- **Hypothesis:** the learned-v1 pipeline fails the +25 % runtime rule (EVAL_PROTOCOL §3) only
  because of how the network is run; whole-page inference and a lower inference resolution bring it
  under the rule without retraining and without losing correct-closed.
- **Change (`src/kelpod/segment/learned.py`):** `LearnedSegmenter` runs the whole page in one pass
  (`tile=2048`, was 1024 px tiles with 128 px overlap = about 2.3x the page area at 1300x1400 px)
  and shrinks the page by `infer_scale=0.65` (INTER_AREA) before the network, bilinear logits back
  to page size. Same weights (v1-s1, v1-s2 ONNX), no retraining: **0 GPU-h**, no Slurm job.
  Final `kelpod-run.toml` lists both seeds (sha256, all `training/manifests/*.json`);
  `locked_eval.py --verify-manifests`: 7 manifests, 0 overlaps.
- **Network only, 1 core (4 pages, fp32 ONNX, IoU vs fp32 full scale, wall/window/door/room):**
  whole page 1.0: 6.8 s; 0.75: 4.5 s (0.991/0.970/0.961/0.997); 0.65: 3.1 s (0.980/0.951/0.938/
  0.996); 0.5: 1.9 s (0.927/0.918/0.874/0.988). 1024 px tiles at 1.0: 35 s on the same loaded node.
  Dynamic int8 quantisation (`quantize_dynamic`): no result after more than 10 minutes on 1 page
  set, killed (ORT's int8 convolutions are far slower on this CPU); static int8 not tried.
- **Per-step runtime, 1 core, 5 val-synth plans, mean s/plan (node load was 14-20, so absolute
  numbers vary 2x between sessions; compare only within one session):** classical total 13.1
  (clean 0.4, segment 0.5, OCR 11.5, export 0.7); learned before (tiles 1024, 1.0, earlier loaded
  session) segment 28.0 of 58.4 total vs classical 23.9; learned whole page 1.0: segment 7.4, total
  20.7 (1.58x classical). OCR was **not** slower in the learned pipeline: 12.2 s vs 11.5 s in the
  same session; the earlier 27.9 vs 21.2 s came from the node load (jobs run concurrently).
- **Dev val-synth, first 40 plans, eval-branch panel via `kelpod_train.val_synth` (8 workers x 1
  thread, synthetic):** correct-closed 1.000 for v1-s1 and v1-s2 at both 0.75 and 0.65.
- **Smoke run (`locked_eval.py --splits val-synth --smoke --limit 40`, 8 workers x 1 core, run
  `20261124-233408_exp-learned-fast`, synthetic):** correct-closed baseline 0.900, cand0 (seed 1)
  0.950, cand1 (seed 2) 0.950 (one failed plan s200001, 4 of 5 rooms; the dev panel scored it
  correct, a difference between the two panels not yet explained); median runtime baseline 42.7 s,
  cand0 50.1 s (+17.3 %), cand1 51.1 s (+19.6 %, CI +9.9 to +36.4 %); p95 59.3 s vs 66.7 / 67.2 s.
  Smoke has no promotion decision. The +25 % rule is met on the point estimate with ~6 pp margin
  only; the 200-plan val number for both seeds at scale 0.65 was **not** run (40 plans only).
- **Decision:** keep scale 0.65. Next: val-synth 200 plans for both seeds at 0.65 before the lead's
  locked run. A lighter encoder (ResNet-18) was not trained; it is the fallback if the real
  locked-run runtime exceeds +25 %.

## ocr-chainsum (2026-11-24, branch exp/ocr-chainsum)

- **Hypothesis:** dimension chains say every length more than once (parts sum to the
  Gesamtmaß, nested chains share ticks, opposite facades repeat the overall, every value matches
  its pixel span at one px/m). Decoding all labels jointly with n-best readings raises dimension
  precision and recall over one greedy guess per label, and with it the dimension fit.
- **Change** (`src/kelpod/ocr/`): `nbest.py` (CTC prefix beam search over RapidOCR's softmax,
  top 5 per digit-bearing box, no extra recogniser pass); `chains.py` (chains from shared
  dimension lines; ticks kept only with a 45° stroke plus an extension line; CP-SAT Boolean
  model over tick coordinates: one value per tick node, one option per label =
  (tick interval, n-best value, lost-superscript-⁵ variant, raised-mm-digit variant) or drop;
  unread intervals whose ends are tied by the chosen labels get an exact inferred value;
  corrections and inferences are emitted only if no unchecked label they depend on has a cheap
  alternative; pages where under half of the chain labels are consistent keep the greedy result).
  `parse_area` reads a lost comma (`2091mF` = 20,91 m²). Contracts: optional
  `TextItem.alternatives`, `DimensionLabel.origin` (`read` / `corrected` / `inferred`).
  New dependency: OR-Tools (Apache-2.0). Geometry unchanged.
- **Data:** dev synth (train style) seeds 200000-200049 for tuning, 200050-200099 held out, degrade
  0.5 and 1.0; frozen val-synth (200 plans). Script: `tests/ocr/chain_eval.py` (cached reads).
  No test-synth, no sealed data. CPU only, 0 GPU hours (all runs on the agent node).
- **Results** (synthetic, dev/val, not a locked evaluation), greedy (v0) → chain:

  | set | dim precision | dim recall | linked to right span | correct-closed | room IoU | total area err | edit cost |
  |---|---|---|---|---|---|---|---|
  | dev held-out, degrade 0.5 (50) | 0.803 → 0.882 | 0.717 → 0.807 | 0.544 → 0.793 | 0.74 → 0.78 | 0.9405 → 0.9444 | 0.0133 → 0.0134 | 3.38 → 2.58 |
  | dev held-out, degrade 1.0 (50) | 0.799 → 0.874 | 0.693 → 0.772 | 0.498 → 0.733 | 0.56 → 0.60 | 0.8804 → 0.8861 | 0.0479 → 0.0476 | 9.12 → 8.60 |
  | val-synth (200) | 0.803 → 0.878 | 0.708 → 0.793 | 0.530 → 0.775 | 0.720 → 0.725 | 0.9401 → 0.9443 | 0.0128 → 0.0124 | 4.14 → 4.00 |

  Of the labels the decoder changed on val-synth, corrected ones are right 78 % (501/641) and
  inferred ones 57 % (420/731); most wrong ones have the right span but are ±5 mm off (lost
  superscript ⁵ ambiguity).
- **Runtime:** `scripts/benchmark.py --n 6 --seed 200000` on the shared node, main vs branch,
  two rounds: 3.64 / 3.66 s → 4.60 / 4.40 s per plan (+20-26 %). Steady state +0.5-0.8 s per plan
  (decoding +0.2-0.3 s mean, p95 +0.5 s; beam search 0.25 s); a fresh process also pays the
  OR-Tools import (~0.5 s). Borderline against the +25 % rule; the locked run decides.
- **Real plan check** (one web JPEG of a single-family-house plan, 550 px, dev only, not committed):
  room areas `2091mF` / `2142m1` now read as 20,91 / 21,42 m²; `1245` gets the 124⁵ cm = 1.245 m
  option, but at this resolution the page-wide scale and most reads are too poor, so the guard
  keeps the greedy result. `2865` for 178⁵ is a digit misread that only chain arithmetic could
  fix, which needs a readable chain. A higher-resolution scan is needed to judge real plans.
- **Decision:** candidate for promotion by the lead (`scripts/promotion_check.sh`, not run here).
  Ideas not built: margin by re-solving, page-level length style (cm / ⁵ / m) as a prior, the room
  stamp area as a further constraint, using `origin` as a weight in the geometry fit.

## din-web-plans (2026-11-24, branch exp/din-web-plans)

- **Hypothesis:** real DIN plans from the web fail for three separable reasons: (1) tiny images
  were *shrunk* (file dpi 96/300 assumed, resampled to 200 dpi), (2) the classical footprint filter
  deletes every wall outside a small enclosed region, (3) walls drawn as two thin lines are
  invisible to the solid/hatched wall mask.
- **Change:** `clean/io.py`: effective dpi from the modal glyph height (2.5 mm digits); images
  below 130 dpi are upscaled (Lanczos + unsharp mask, factor 200/eff, max 3x) instead of using the
  file dpi. `segment/classical.py`: the footprint filter is skipped when the enclosed footprint
  touches under 65 % of the wall length (synthetic plans: min 70 %, positionsplan 60 %).
  `segment/pairs.py` (new backend `pairs`): line components per direction (0/90 and, in 45-degree
  rotated copies, 45/135), parallel pairs at the plan's dominant (length-weighted) spacing, fill
  test, widest-pair dedupe, heavy single strokes as walls, network filter (isolated furniture,
  chains, terrace grids drop out), door-gap closing, rooms from the barrier. `pipeline.convert`
  tries `pairs` only if the classical graph is not closed and under half its wall length is
  hatched/solid, and takes it only if it scores better (rooms, dangling ends, groups).
  Script `scripts/web_plan_check.py` (real plans, no ground truth). Tests: `tests/segment/test_pairs.py`.
- **Data:** two web plans in `~/kelpod-data/real/din-web-2026-11-24/` (dev only, licence
  unchecked, not committed; no ground truth, no hand labels); synthetic seeds 300-319, degrade
  0.5 and 1.0. The web collector had written no plan images when this ran. CPU only, 0 GPU hours.
- **Results (real, main -> branch):**

  | plan | before | after |
  |---|---|---|
  | efh-erdgeschoss-550.jpg | 23 walls, 0 rooms, 13 dangling, 4 groups, scale guessed | 48 walls, 5 rooms (0.7-9.2 m2, not the real rooms), 12 dangling, 1 group, scale from dims (src=dimensions) |
  | positionsplan-fig134.png | 18 walls, 0 rooms | 37 walls, 5 rooms (one 77 m2 merged), 9 dangling, 1 group |

  Not solved: neither plan is closed. Segmenter overlay checked by eye (Read tool).
- **Synthetic guard** (20 plans, seed 300): identical to main on all means at degrade 0.5
  (closed 0.85, room IoU 0.990, wall F1 0.995) and 1.0 (closed 0.75, IoU 0.985, F1 0.992).
  Runtime 4.9 s/plan (0.5) and 4.9 s (1.0); the main run was taken while other jobs ran
  (6.7 / 7.9 s), so no clean ratio; against the ocr-chainsum figure (4.4-4.6 s) it is within +25 %.
  `uv run pytest`: 337 passed, 1 skipped (existing).
- **What still fails and why:** (a) efh: the pairs path finds the axis-aligned double lines (see
  overlays) but misses walls whose faces fuse into one line at this resolution, the 45-degree
  hall/bay walls (diagonals are chopped by aliasing; only ~3 short pieces found) and curved
  walls, so no closed outline; the classical result (garbage rooms) is therefore still chosen.
  Closing with 5 px line bridging raised clutter and was reverted. (b) Only two dimensions match
  at the nominal scale, so the scale rests on few labels; OCR of 178^5-style cm chains was not
  improved here. (c) positionsplan: partial hatched walls, circled position numbers cross wall
  lines, open outline at window gaps.
- **Decision:** the upscale and footprint-guard changes are safe candidates (no synthetic change);
  the `pairs` backend is an experimental fallback, not a result. Not promoted, no locked run.


## historic-plans (2026-11-24, CPU only, no GPU used)

Branch `exp/historic-plans` (on top of main + `exp/din-web-plans`). Labels: `~/kelpod-data/real/user-feedback-2026-11-24/labels/`
(outside git; hand-made by the agent with `scripts/dev/annotate_assist.py`, +-5 px, scales of the villa and the sheet ASSUMED
from unnumbered scale bars). Scorer: `scripts/dev/realdev_score.py`. Real, 3 plans, not a benchmark; scale_err in the table is
partly circular for the two bar plans (label scale uses the same bar).

| plan | before: rooms / wall F1 / room IoU / scale err | after |
|---|---|---|
| villa-garten-vorhof (19 GT rooms) | 5 / 0.16 / 0.02 / 59 % | 3 / 0.32 / 0.07 / 1 % |
| villa-gundel (7) | 12 / 0.59 / 0.31 / 4 % | 13 / 0.55 / 0.30 / 5 % (source: dimensions) |
| tafel71 plan 2 (7) | 12 / 0.27 / 0.23 / 59 % (whole sheet) | 3 / 0.41 / 0.18 / 2 % |
| tafel71 plan 3 (6) | 12 / 0.23 / 0.14 / 59 % | 4 / 0.71 / 0.29 / 2 % |

Synthetic (`benchmark.py --n 20 --seed 300 --degrade 0.5`): before closed 0.850, IoU 0.991, wall F1 0.996; after 0.850, 0.990, 0.995.
Tried and dropped: snapping partitions across 1.2 m door gaps in massive walls (made tafel rooms worse).

## din-web-v2 (2026-11-24, branch exp/din-web-v2 = exp/din-web-plans + main)

- **Why:** the locked run of exp/din-web-plans gained room IoU (+0.047) but regressed door precision
  (-3 to -6 pp), edit cost (+11 to +24 %) and exterior wall length error (+16 to +69 %).
- **Method:** val-synth only (never test splits). Smoke runner, first 30 val plans, main vs branch
  (before the fix): ext_wall_length_error 0.0009 -> 0.0310, door precision 0.996 -> 0.979, edit
  cost 1.97 -> 3.37. The per-plan table showed the whole regression in one plan (s200009: ext wall
  error 0.02 -> 0.92, door/window precision 1.0 -> 0.5/0.4, edit cost 17 -> 59, room IoU
  0.58 -> 0.92). Component switch on that plan (segmenter wall count): footprint guard off -> 15
  walls (as main), guard on -> 31 walls; the upscale (no effect on a 200 dpi plan) and the `pairs`
  fallback (not triggered: hatched walls) were not involved. The OCR chain decoder was not ablated
  separately (it does not change walls/openings; no per-plan difference was seen outside s200009).
  The full 5-way ablation was cut short by a heavily loaded node (jobs 2 min/plan).
- **Root cause:** the footprint guard (skip the "delete walls outside the enclosed footprint" filter
  when the footprint touches under 65 % of the wall length) kept *everything* when the outline was
  open. On a skewed, shaded plan the outline is open, so dimension-chain artefacts (extension lines,
  ticks, outline-only "E" runs 29-55 px thick) stayed as walls, windows and door gaps: wrong
  exterior walls, openings and rooms (the IoU gain on that plan was the rooms merging).
- **Fix:** in the partial case, keep only the networks (pieces touching within 6 px) that border
  the footprint and are at least 3 m long in total (`_connected_to_footprint`); loose stubs go.
  Full footprint case unchanged. Test: `tests/segment/test_pairs.py::test_partial_footprint_*`.
- **Result (val-synth, n=200, main -> branch, CPU, ~1300 s wall for the run on 8 cores):**
  correct_closed 0.720 -> 0.725, room IoU 0.9401 -> 0.9443 [+0.0025, +0.0062], wall F1
  0.9878 -> 0.9889, door precision 0.9748 -> 0.9738 [-0.0030, 0.0], door recall 0.9609 -> 0.9587,
  window precision 0.9963 -> 0.9963, edit cost 4.135 -> 3.980 (-3.7 %, CI [-11.5, +4.1]),
  ext_wall_length_error 0.0058 -> 0.0061 (+4.9 %, CI [0, +22.4], not flagged), runtime median
  within +25 % (loaded node). No rule-3 regression on val-synth.
- **Web plans (real, dev only):** unchanged by the fix, identical before and after: efh 48 walls,
  5 rooms, 12 dangling, 1 group; positionsplan 37 walls, 5 rooms, 9 dangling, 1 group; neither
  closed.
- **Caveat:** val-synth did not show the test regression at the 200-plan level before the fix (only
  the 30-plan subset was run before). The test splits use a held-out style; the mechanism
  (partial footprint + dimension artefacts) is style independent, but whether it explains all of the
  test regression is unverified. Not tested on test splits by design.

### din-web-v2, door recall follow-up (2026-11-25)

- **Trigger:** locked run 1 failed rule 3 only on door recall (test-synth, fresh).
- **Mechanism found on val-synth (2 of 200 plans lose door recall: s200009, s200048):** in s200048 the
  chain decoder gives a correct 2,135 span (168 px instead of 173), so the wall fit moves the thin
  wall (0.115 m) to x = 8.92 m (RMS before the fit 0.5 cm instead of 4.3 cm). The detected door gap
  lies at x = 8.70 m, 0.22 m from the wall, and `attach_openings` only accepts
  `max(0.6 x thickness, 0.2 m)`: "1 detected opening could not be placed on any wall". With the greedy
  reading the wall happened to sit 0.18 m from the gap, so the door survived by luck. Leave-one-out
  over the dimension labels does not restore it, `pairs` off and OCR `greedy` give the same
  geometry otherwise: it is the opening-to-wall tolerance, not one label. The upscale does not act on
  vector PDFs.
- **Fix:** `GeometryParams.opening_wall_tol_m = 0.3` (was 0.2 inline). Tests:
  `tests/geometry/test_opening_tol.py` (22 cm off is placed, 60 cm off is not). `uv run pytest`:
  350 passed, 2 skipped (fastapi, torch missing, as on main).
- **Train-style check:** 60 plans, seeds 2000-2059, degrade 1.2, benchmark runner, CPU: door recall
  identical (0.9532) for main, branch, branch with greedy OCR, without `pairs`, and with the old
  footprint guard; so no component of the branch costs door recall on train style.
- **val-synth, n=200, main (4df2fea) -> branch (4be979b):** door recall 0.9609 -> 0.9599 [-0.0030, 0.0000],
  door precision 0.9748 -> 0.9738 [-0.0030, 0.0000], room IoU 0.9401 -> 0.9443, edit cost
  4.135 -> 3.975, ext wall error 0.0058 -> 0.0061, all rule-3 checks ok. The remaining door-recall loss
  is s200009 (one door of five, same count of doors, position match).
- **Caveat:** val-synth shows only a -0.1 pp door recall change, the test splits -1 to -4 pp. The
  mechanism above (tolerance between pixel gaps and dimension-fitted walls) is style independent
  and can be larger with noisier gap positions, but it is not proven to explain the test drop.
  Incident: while looking for the run directory I listed the per-plan results of the coordinator's
  latest locked run (test-synth / fresh ids) once, before noticing; I did not use them for any
  decision, the fix was derived from val-synth s200048 only.

## historic-v2 (2026-11-25, branch exp/historic-v2 = main/baseline-v1 + historic-plans + din-web-v2 + door gaps)

- **Hypothesis:** with the learned segmenter the walls of the historic plans are mostly found, but
  rooms stay open because doors and segmentation dropouts are plain gaps in solid walls. Bridging
  a dangling wall end across a gap, only on evidence, closes the rooms without hurting the
  synthetic splits.
- **Merge:** `origin/exp/historic-plans` (scale bars, interior dimensions, sheet splitting,
  historic room names, `scripts/dev/realdev_score.py`), then `origin/exp/din-web-v2` (fixed
  footprint guard, attach tolerance 0.3 m, chain OCR, upscaling). `pipeline.convert_all` resolves
  the default backend from `kelpod-run.toml` (learned) as on main. `uv run pytest`: 397 passed, 2
  skipped (skips pre-existing).
- **Change (`src/kelpod/geometry/doorgap.py`, switch `GeometryParams.close_door_gaps`):** for each
  dangling wall end (touches no other wall) look for (a) the collinear dangling end of another
  wall pointing back at it, same thickness (+-6 cm), centrelines within a few cm, or (b) a
  perpendicular wall behind it. The gap must be 0.55-1.3 m (door: added as an opening of kind
  door) or up to 2.6 m (segmentation dropout: bridged as plain wall, no opening). A gap towards
  open air has no target and is left alone. Gate: all candidates together must raise the number of
  closed rooms, and then every bridge that is not needed for that gain is dropped (leave-one-out).
  A first version with gaps limited to 1.3 m barely changed the plans (real, villa 7 -> 9 rooms,
  Tafel unchanged), because most gaps are dropouts, not doors.
- **Synthetic guard, run 1 (val-synth 200, smoke, job 128911):** seed 1 correct-closed 0.945 ->
  0.960, seed 2 0.945 -> 0.930 (CIs include 0), but rule 3 flagged `ext_wall_length_error` and
  `gross_volume_error` (absolute 0.0001 -> 0.0002, i.e. 0.01 % -> 0.02 % of the wall length).
  Ablation on val plans: door gaps are not the cause (identical with the switch off); the chain
  decoding of the dimension labels (from din-web-v2 / ocr-chainsum) moves the exterior wall length by
  ~0.05 %. Fix: `ocr.interpret.annotate` defaults to `decode="greedy"` (chain decoding stays
  available as an option); the real-plan numbers below are identical with either setting.
- **Real, 3 hand-labelled plans (dev only, agent-made labels +-5 px, not a benchmark).**
  Wall F1 / room IoU with the label scale ("oracle": geometry only) / scale error;
  main = baseline-v1, branch = this branch:

| plan | main F1 / IoU / scale err | branch F1 / IoU / scale err | branch before door gaps |
|---|---|---|---|
| tafel71 plan 2 (7 rooms) | 0.18 / 0.00 / 91 % (whole sheet) | 0.61 / 0.72 / 1.9 % | 0.62 / 0.72 |
| tafel71 plan 3 (6) | 0.19 / 0.00 / 91 % | 0.73 / 0.55 / 1.9 % | 0.72 / 0.37 |
| villa-garten-vorhof (19) | 0.86 / 0.17 / 37 % | 0.81 / 0.41 / 50 % | 0.81 / 0.24 |
| villa-gundel (7) | 0.36 / 0.00 / 68 % | 0.71 / 0.25 / 5 % | 0.56 / 0.00 |

  (F1 and IoU are the oracle numbers; at the pipeline's own scale: Tafel 0.73 / 0.55 on plan 3, villa
  IoU 0.06 because its scale is wrong, Gundel 0.22.)
- **Still failing:** the villa scale bar has no numbers; the tick value is guessed from the wall
  thickness, and the learned segmenter's thickness (22 px, true about 11 px) gives 0.5 m per tick
  instead of 1 m (scale error 50 %, worse than main's 37 %). Gundel: the learned wall mask is
  patchy on the thick outer walls (large dropouts), thin partitions are missed. Rooms at
  IoU < 0.6 everywhere: **no plan works yet.**
