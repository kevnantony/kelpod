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
