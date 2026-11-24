# Experiments

Hypothesis, change, data, GPU hours (with justification), results, decision. Append only.

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
