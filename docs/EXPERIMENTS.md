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
