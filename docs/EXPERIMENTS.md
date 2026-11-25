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
- **Locked run 2 of 2:** started 2026-11-25 on the agent node (8 pinned workers).
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
