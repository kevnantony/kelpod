# Kelpod learned segmentation (fine-tuning)

Work log and results for the "finetune" brief (`docs/AGENT_BRIEFS.md`). Synthetic and re-rendered
dataset plans only, no real German *Bestandspläne* yet. Status lines are dated; numbers are only
written here once measured.

## Environment

Separate uv project (heavy ML deps stay out of the main `pyproject.toml`):

    cd training && uv sync          # torch 2.6.0+cu124, segmentation-models-pytorch, timm
    uv run python -m kelpod_train.<module> ...

Cluster driver 550 (CUDA 12.4). GPU work only via `sbatch training/slurm/*.sbatch`.
Data in `~/kelpod-data/` (not in git), checkpoints in `~/kelpod-models/` (not in git).

## 1. Model choice (2026-11-23)

Checked (see `docs/research/OPEN_SOURCE_SURVEY.md`):

- **Raster2Seq** (SIGGRAPH 2026, MIT code, CC BY-NC weights on CubiCasa5k): room polygons as token
  sequences; newest, but autoregressive decoding, research code, and fpvec-lab reports detection
  readouts beat sequence readouts on real scans. Not chosen for the time budget.
- **RoomFormer / PolyRoom / FRI-Net**: trained on point-cloud density maps, wrong input type.
- **CubiCasa5k model** (hourglass multi-task net, released weights, CC BY-NC 4.0): walls, room
  types, door/window icons as pixel maps on raster plans. Released weights + simple code, runs on
  CPU: used as the **zero-shot baseline**.
- **Fine-tuned model**: a U-Net with an ImageNet-pretrained encoder (segmentation_models_pytorch,
  MIT; timm encoders) predicting 5 classes (background, wall, window, door, room interior),
  trained on synth + re-rendered dataset plans + CubiCasa5k. Pixel maps are then vectorised by a
  deterministic post-processor (`kelpod.segment.learned.masks_to_segmentation`), so the geometry
  step stays deterministic. Chosen because it is the option we can surely get running and train
  within the budget; it matches the "detect, then assemble geometry" readout fpvec-lab found best.

Post-processor sanity check (oracle = ground-truth class maps, synth seeds 100000-100003, degrade
0.5 and 1.0): wall recall 0.995-0.999, wall precision 0.998-0.999, door/window recall 1.00,
room recall 1.00, room IoU 0.98. So the vectoriser loses little; errors of the learned backend
come from the network.

## 3. Data (built 2026-11-23/24)

All pages are cleaned 200-dpi pages (``kelpod.clean``) with a 5-class label map; dataset plans are
re-rendered in our DIN style by `kelpod_train/render.py` and degraded with the synth degradation.

| Source | Licence | Train | Val | Test (held out) |
|---|---|---|---|---|
| `kelpod.evaluation.synth`, default style | own | 4000 (seeds 1,000,000-1,003,999) | 60 (seeds 900,000+) | seeds 100000-100049 at degrade 0.5 and 1.0 |
| Swiss Dwellings v3.0.0, per floor | CC BY 4.0 | 1859 floors (1323 buildings, 763 sites) | 47 | 96 floors, 87 buildings; held out by site ID, 0 overlapping buildings/sites |
| ResPlan | CC BY 4.0 | building (up to 2000) | 50 | 100 (official test split) |
| CubiCasa5k, original images rescaled to ~20 px walls | CC BY-NC 4.0 | 4200 (official train) | 400 | 400 (official test) |

Manifests with SHA-256 of every training file: `training/manifests/<source>.json` (Swiss: floor,
building and site IDs). Synthetic training never uses the generator's held-out test style.

## 4. Fine-tuning run r34 (2026-11-24, measured)

- Slurm job 128281, ampere partition, 1x RTX A6000, **1.01 GPU hours**.
- U-Net, ResNet-34 ImageNet encoder, 1-channel input, 5 classes; 24,000 iterations x batch 16,
  512 px crops; AdamW 1e-3, cosine schedule; loss = weighted cross-entropy (1, 2, 4, 4, 1) + Dice;
  bf16 autocast. Mix: synth 0.35, Swiss 0.25, ResPlan 0.2, CubiCasa5k 0.2.
- Best checkpoint at iteration 20,000 (score = mean IoU of wall/window/door over the val sets =
  0.894). Exported as TorchScript `~/kelpod-models/kelpod-seg/r34/model.pt` (98 MB, not in git).
- Validation IoU at 20k (val pages, dev numbers):

| val set | wall | window | door | room |
|---|---|---|---|---|
| synth (60 pages) | 0.985 | 0.973 | 0.960 | 0.997 |
| Swiss Dwellings (40) | 0.963 | 0.944 | 0.919 | 0.955 |
| CubiCasa5k (40) | 0.792 | 0.806 | 0.699 | 0.717 |

The last evaluation (24k) dipped to 0.853; most likely ResPlan pages joining the mix late (the file
list is rescanned every 15 min while the build was running). The 20k checkpoint is used.

## 5. Vectoriser fix found during evaluation

With oracle (ground-truth) class maps, the first vectoriser left a short wall sliver between a door
and the face of a perpendicular wall, which the geometry step turned into a dangling wall (3/3
dev plans not closed). Fix in `masks_to_segmentation`: slivers under 0.30 m next to an opening are
absorbed, and opening ends run up to a perpendicular wall face within 0.30 m (as the classical
segmenter does). After the fix, oracle class maps on 12 dev seeds (90000-90011, degrade 1.0) give
12/12 closed plans after geometry (classical on the same pages: 5/12). All learned and zero-shot
numbers below use the fixed vectoriser.

## Files

- `kelpod_train/labels.py`: class scheme, synth samples, page warps.
- `kelpod_train/render.py`: DIN-style renderer for vector plans (reuses the synth canvas).
- `kelpod_train/datasets/`: Swiss Dwellings, ResPlan, CubiCasa5k converters.
- `kelpod_train/build.py`: writes PNG pairs.
- `kelpod_train/third_party/cubicasa_hg.py`: vendored CubiCasa5k network (CC BY-NC 4.0).
