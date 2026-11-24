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

## 6. Results: classical vs zero-shot vs fine-tuned (2026-11-24, measured)

**All numbers are dev/synthetic numbers** from `kelpod_train.evaluate` and `kelpod_train.val_synth`,
not a locked evaluation under `docs/EVAL_PROTOCOL.md`. No real German plans were tested. Classical =
committed classical segmenter (fe5a361, closure fixes). Zero-shot = released CubiCasa5k model, page
scaled by 0.7 (chosen on dev seeds 90000-90002, not on test), same vectoriser. Learned = r34 at
20k iterations. The same geometry step and the same cached OCR are used for all three.
Raw rows: `training/runs/{eval_base,eval_base_resplan,eval_ft,val_synth}/rows.json` (git-ignored).

### Synthetic test seeds 100000-100049 (50 plans each), after geometry

| degrade | backend | closed | wall F1 | room IoU | room recall | door recall | window recall |
|---|---|---|---|---|---|---|---|
| 0.5 | classical | 0.88 | 0.994 | 0.970 | 0.987 | 0.989 | 0.991 |
| 0.5 | zero-shot CubiCasa5k | 0.36 | 0.998 | 0.883 | 0.929 | 0.993 | 0.960 |
| 0.5 | **fine-tuned** | **1.00** | 0.997 | **0.983** | **1.000** | **1.000** | **1.000** |
| 1.0 | classical | 0.70 | 0.981 | 0.937 | 0.964 | 0.950 | 0.989 |
| 1.0 | zero-shot CubiCasa5k | 0.16 | 0.983 | 0.892 | 0.932 | 0.993 | 0.964 |
| 1.0 | **fine-tuned** | **1.00** | **0.998** | **0.980** | **1.000** | **1.000** | **1.000** |

Segmentation-only (page px, before geometry) at degrade 1.0: wall recall 0.982 / 0.997 / 0.994,
door recall 0.956 / 0.993 / 1.000, room recall 0.938 / 0.974 / 1.000 (classical / zero-shot /
fine-tuned).

### val-synth (frozen, 200 plans, train style, degrade 0-1.2), eval-branch panel

| metric | classical | fine-tuned |
|---|---|---|
| correct-closed (headline) | 0.720 | **0.980** |
| closed | 0.760 | **0.995** |
| room IoU | 0.940 | **0.982** |
| room precision / recall | 0.983 / 0.962 | **1.000 / 1.000** |
| wall F1 @10 cm | 0.988 | **0.999** |
| door P / R | 0.975 / 0.961 | **0.998 / 0.998** |
| window P / R | 0.996 / 0.991 | **1.000 / 1.000** |
| edit cost | 4.14 | **0.45** |
| total area error | 0.013 | **0.001** |
| thickness-class accuracy | 0.995 | 1.000 |
| crashes / IFC failures | 0 / 0 | 0 / 0 |
| runtime, whole pipeline, mean s* | 48.8 | 62.7 |

\* 8 plans in parallel with 2 threads each on one 16-core node (OCR included), so absolute times are
inflated; see the runtime notes below. val is for model selection, so these are not test numbers.

### Held-out dataset pages (mask-level; rooms vs rooms extracted from the GT class map)

| set | backend | wall IoU | wall px F1 | door recall | window recall | opening precision | room IoU | room recall |
|---|---|---|---|---|---|---|---|---|
| Swiss Dwellings test (96 floors, held out by site) | classical | 0.618 | 0.751 | 0.502 | 0.490 | 0.660 | 0.314 | 0.332 |
| | zero-shot | 0.751 | 0.856 | 0.860 | 0.670 | 0.847 | 0.681 | 0.719 |
| | **fine-tuned** | **0.892** | **0.942** | **0.952** | **0.978** | **0.958** | **0.970** | **0.985** |
| ResPlan test (100) | classical | 0.717 | 0.827 | 0.673 | 0.622 | 0.900 | 0.600 | 0.636 |
| | zero-shot | 0.817 | 0.899 | 0.889 | 0.690 | 0.810 | 0.540 | 0.578 |
| | **fine-tuned** | **0.903** | **0.949** | **0.969** | **0.975** | **0.981** | **0.975** | **0.988** |
| CubiCasa5k test (first 100 of 400) | classical | 0.459 | 0.615 | 0.095 | 0.138 | 0.223 | 0.197 | 0.197 |
| | zero-shot | 0.760 | 0.860 | **0.936** | 0.787 | 0.892 | 0.838 | 0.874 |
| | **fine-tuned** | **0.815** | **0.896** | 0.928 | **0.809** | **0.893** | **0.863** | **0.897** |

Caveats: Swiss and ResPlan pages are re-rendered in our DIN style, so they test geometry
diversity, not real drawing styles. CubiCasa5k is the only set with real (Finnish) raster plans.
The zero-shot model was trained on CubiCasa5k's train split, so its CubiCasa numbers are in-domain
for it. The classical segmenter assumes DIN conventions and is not expected to work there.
Classical rows come from the baseline run (classical code unchanged since); zero-shot rows are
from the re-run with the fixed vectoriser.

### Runtime (CPU, segmentation step only)

- Single process, 8 torch threads, shared 16-core node, pages ~1.3-1.7 MP: fine-tuned **3.4-5.4 s**
  (4 threads: 5.5-8.3 s); classical 0.3-0.5 s.
- In the evaluation runs (8 parallel workers x 2 threads, contended): median 59 s fine-tuned,
  34 s zero-shot, 1.5 s classical. Large Swiss floors took up to ~100 s there.
- Either way well under the 3-minute-per-plan target, but the learned step adds seconds per page;
  an ONNX export (supported by `load_model`) or a GPU would cut this.

### GPU hours

Training r34: 1.01 GPU h (RTX A6000). Evaluation jobs ran on the pascal partition with the GPU
hidden (CPU-only runtime numbers): ~0 GPU compute, roughly 3-4 h of GPU allocation in total. No other GPU
use.

### What real German plans would add

- A real test: everything above is synthetic or re-rendered; RealBench-DE (test-real) decides.
- Real drawing variety that the renderer does not cover: hand-drawn and pencil plans, other
  hatchings (e.g. cross-hatching, solid poché, concrete dots), stamps, handwritten corrections,
  yellowed copies, multi-plan sheets, sections and elevations on the same page.
- Fine-tuning data (a few hundred annotated pages) for the remaining domain gap, and
  self-training on unlabelled scans (`docs/research/V2_RESEARCH.md`, bet 6).
- Measured expert correction time with the pilot, which outranks every automatic metric.

### Expected risk on test-synth (held-out style)

The held-out style changes fonts, hatching patterns, line weights and scan damage. The network
was trained only on the default style (plus CubiCasa5k photos of real plans and Swiss/ResPlan
renders in the same default style). Hatching and line weights are exactly what it relies on to
see walls, so a drop is likely on test-synth; how large is unknown until a locked run. CubiCasa5k
training (other wall styles) should limit it. Mitigation, if needed: style randomisation in
training data (allowed train-style options only), never the held-out options.

## How to run

    # use the learned backend in the pipeline (CPU; torch needed for .pt, onnxruntime for .onnx)
    export KELPOD_SEG_CHECKPOINT=~/kelpod-models/kelpod-seg/r34/model.pt
    cd training && uv run python -c "from kelpod.pipeline import convert; \
        convert('plan.pdf', 'out.ifc', segment_backend='learned')"
    # (the `kelpod convert` CLI has no backend option yet)
    # data
    uv run python -m kelpod_train.build synth --out ~/kelpod-data/train/synth --start 1000000 --n 4000
    uv run python -m kelpod_train.build swiss|resplan --split train --n 2000 --out ...
    uv run python -m kelpod_train.datasets.cubicasa --split train --out ~/kelpod-data/train/cubicasa
    uv run python -m kelpod_train.manifest           # training/manifests/*.json (SHA-256, pHash, IDs)
    # train (GPU, Slurm)
    sbatch --job-name=kseg-r34 training/slurm/train.sbatch ~/kelpod-models/kelpod-seg/r34 \
        --encoder resnet34 --iters 24000 --batch 16 --val synth,cubicasa,swiss
    # evaluate (CPU)
    sbatch training/slurm/eval.sbatch --backends classical zeroshot@0.7 learned \
        --checkpoint ~/kelpod-models/kelpod-seg/r34/model.pt --splits synth swiss cubicasa resplan \
        --limit 100 --out training/runs/eval
    KELPOD_SEG_CHECKPOINT=... uv run python -m kelpod_train.val_synth --out training/runs/val_synth

## Files

- `kelpod_train/labels.py`: class scheme, synth samples, page warps.
- `kelpod_train/render.py`: DIN-style renderer for vector plans (reuses the synth canvas).
- `kelpod_train/datasets/`: Swiss Dwellings, ResPlan, CubiCasa5k converters.
- `kelpod_train/build.py`: writes PNG pairs; `manifest.py`: training manifests.
- `kelpod_train/train.py`: U-Net training + TorchScript export; `slurm/*.sbatch`.
- `kelpod_train/zeroshot.py`: CubiCasa5k zero-shot predictor; `third_party/cubicasa_hg.py`:
  vendored CubiCasa5k network (CC BY-NC 4.0).
- `kelpod_train/evaluate.py`, `kelpod_train/val_synth.py`: comparisons above.
- `../src/kelpod/segment/learned.py`: `LearnedSegmenter` + deterministic vectoriser;
  `../tests/training/test_learned.py`.
