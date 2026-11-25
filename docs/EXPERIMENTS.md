# Experiments

Hypothesis, change, data, GPU hours (with justification), results, decision. Append only.

## learned-v2 (2026-11-25, branch exp/learned-v2, not yet run through the gate)

- **Hypothesis:** baseline-v1 reaches correct-closed 0.945 on val-synth (training style) but only about
  0.54-0.60 on the held-out-style splits (synthetic). Broader train-only style randomisation, more and
  more diverse data and a longer schedule narrow that gap (and help on real scans). Same network
  (U-Net ResNet-34, whole-page ONNX at 0.65x): runtime unchanged by construction.
- **Change (training only; `training/kelpod_train/{style_v2,render_v1,degrade_v1,build,train}.py`):**
  new train options on top of learned-v1 (`style_v2.py`, guard + test `tests/training/test_style_v2.py`):
  18 more open fonts (SIL OFL Google Fonts: sans, serif, hand lettering; none narrow, monospace or
  geometric-gothic; see CREDITS.md), hatching at 15/75/90 degrees, damage kinds `aged_paper`, `defocus`,
  `scan_bands`, plus page rescale jitter 0.72-1.40 (was 0.8-1.25). Held-out options are not used;
  poche/solid walls and cross-hatching are held out in `kelpod.evaluation.synth.styles`, so they are
  **not** used (deviation from the brief, on purpose). More data: synth_v2 12,000 pages (new geometry
  seeds 1,200,000+), swiss_v2 ~4,500 and resplan_v2 ~4,500 re-renders of the train floors/plans
  with new style draws, on top of v0/v1 data and CubiCasa5k. Schedule 64k iterations (v1: 40k).
- **Dev proxy (style generalisation, tuned on):** 200 synthetic plans (geometry seeds 930,000-930,199,
  not in any eval split) drawn in our renderer with options **reserved from every training run and
  never seen by baseline-v1 either**: 8 fonts (`DEV_FONTS`), hatching 20 degrees, 70 degrees and sparse
  45 degrees (3.3-4.5 mm), damage `uneven_light`, `sensor_noise`, `halftone`. Scored at mask level
  (per-class IoU at the deployment setting: ONNX, whole page, 0.65x) for baseline-v1 seeds 1/2 and
  learned-v2 seeds 1/2. Using v1's own options as the held-out part would favour baseline-v1 (it trained
  on them), hence new options.
- **Hard dev proxy (added after the first one scored 0.94 mean IoU for baseline-v1, too mild):**
  (a) mask level, 200 pages (`build dev_hard`, seeds 960,000+): unseen font + hatch + damage together,
  full scan degradation, 150-220 dpi resampling, JPEG Q 20-55; (b) end to end, 100 plans
  (`kelpod_train.dev_hard`, seeds 950,000+): generator layout and renderer with an unseen font, 45 degree
  hatching at 3.4-4.5 mm, unseen damage, degrade 0.6-1.2, extra low resolution / JPEG / blur, written as
  plan.pdf + ground_truth.json and scored by the real pipeline and the eval panel (correct-closed etc.).
  Only options in no train and no held-out list. Baseline-v1 numbers are measured by CPU jobs 128936-128937.
- **CPU jobs (no GPU; planned):** page builds on pascal (not node02), 8 CPUs each, about 3.5 CPU-s per
  page measured on 8 pages: synth_v2 12,000 pages (3 jobs), swiss_v2 4,500, resplan_v2 4,500, val
  and dev sets about 400 pages: about 21,400 pages = ~21 CPU-h in total, wall about 40-60 min per job.
- **GPU jobs (planned):** `kseg-v2-s1`, `kseg-v2-s2` (ampere, 1 GPU each, 64k iterations x batch 16
  at 512 px). Expected 2.8 GPU-h each (v1: 40k iterations = 1.86 / 1.65 GPU-h, so 64k = 1.6 x
  = 2.6-3.0 GPU-h, plus 6 val sets every 2000 iterations), time limit 6 h, checkpoint + resume every
  2000 iterations; ~5.6 GPU-h in total. Justification: rule 5 needs two seeds; the longer schedule
  because more varied data needs more iterations (v1 val score was still rising at 40k: to be checked
  in `log.jsonl`), same network, no bigger encoder.
- **Selection:** dev proxy, val-synth (smoke) and the dev-val sets only. test-synth / fresh / test-ood /
  sealed data are not looked at.

## ocr-fast (2026-11-25, branch exp/ocr-fast, negative result, closed)

- **Hypothesis:** OCR (~85 % of per-plan runtime on 1 core) can be made >= 40 % faster without
  accuracy loss, buying headroom under the +25 % runtime rule.
- **Profile (1 core, val-synth s200003):** 11.9 s OCR = detector 8.2 s (8 calls, ~0.65-0.9 s/MP),
  recognition 2.3 s, dimension parsing 0.8 s, line removal 0.2 s.
- **Tried (val-synth dev, 12-30 plans):** detect on a downscaled page (896-1800 px): 30-60 % faster
  but dimension recall -3 to -12 pp (superscript mm digits lost); one whole-page detector call: no
  gain; skipping boxes before recognition: no gain; 90-degree pass only on leftover ink: recall
  0.50. No change kept; 0 GPU-h.
- **Untried ideas:** per-plan detector scale from measured text height; downscaled detection plus
  full-resolution re-detection around superscripts; a text-line detector trained on our plan style.

## historic-v2 (2026-11-25, branch exp/historic-v2 @ b8d363c, not gated)

- **Change (on baseline-v1):** historic-plans (scale bars, interior dimensions, sheet splitting,
  historic room names, exterior spaces) + din-web-v2 fixes + door-gap bridging in solid walls
  (`geometry/doorgap.py`: collinear same-thickness ends or a perpendicular wall behind; 0.55-1.3 m
  gaps become doors, up to 2.6 m bridged as wall; kept only if the closed-room count rises).
  Chain OCR decoding switched back to greedy by default (it shifted exterior lengths ~0.05 %).
- **Real, 3 plans (agent-made labels; room IoU at label scale), main -> branch:** Tafel 71 plan 2
  0.00 -> 0.72, plan 3 0.00 -> 0.55, villa 0.17 -> 0.41, Gundel 0.00 -> 0.25. Scale error Tafel
  91 % -> 1.9 %, Gundel 68 % -> 5 %, villa 37 % -> 50 % (scale bar without numbers; learned walls
  read ~2x too thick).
- **Val-synth smoke (200, synthetic):** seed 1 unchanged (correct-closed 0.945, no rule-3 flag);
  seed 2 flags that also appear for baseline-v1 itself as candidate (control job 128920), i.e.
  caused by comparing candidate seed 2 with the baseline's seed 1 (eval flaw, see below).
- **Promotion problem:** the gains are on real plans; the protocol has no real-plan split yet, and
  the synthetic splits are neutral by design, so rule 2 cannot pass. Asked the founders (README).
- **Eval flaw found:** locked_eval runs the baseline with its first declared seed only; with a
  2-seed baseline, candidate seed k must be paired with baseline seed k. Fix on an eval/* branch.

## learned-fast -> PROMOTED as baseline-v1 (2026-11-25)

- **Change:** learned-v1 U-Net (2 seeds) as the default segmenter; whole-page ONNX inference at
  0.65x scale instead of 1024 px tiles (network ~35 s -> ~3 s per page on 1 core). 0 GPU-h (no
  retraining). CPU jobs: Slurm 128900 (val smoke, 9 CPUs, ~1 h), 128901 (gate, 9 CPUs, ~2.5 h).
- **Val-synth (200, synthetic):** correct-closed 0.720 -> 0.945 (s1) / 0.920 (s2); edit cost
  -81 % / -79 %; runtime +21 % / +20 %.
- **Locked run 1 (test-synth 300 held-out style + fresh 200, synthetic), worse seed s2:**
  correct-closed fresh 0.305 -> 0.600 (+29.5 pp [+21.5, +37.5]), test-synth 0.267 -> 0.537;
  room IoU +0.135; door recall +0.13; edit cost -62 to -72 %; runtime median +18 to +28 %
  (s1 fresh +27.7 % [+21.3, +35.3]: point estimate above +25 %, not flagged because the CI is not
  entirely above). Rules 1-6 pass. **Promoted: main = baseline-v1.**
- **Caveat:** the checkpoints live outside git (~/kelpod-models); the sandbox falls back to the
  classical segmenter until the weights are published (needs the founders' OK; training data
  includes CubiCasa5k, CC BY-NC).

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
