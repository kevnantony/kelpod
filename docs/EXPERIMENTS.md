# Experiments

Hypothesis, change, data, GPU hours (with justification), results, decision. Append only.

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
