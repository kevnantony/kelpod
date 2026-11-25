#!/bin/bash
# learned-v2: fine-tune the U-Net on style-randomised data, one seed per job. Resumes from
# <out>/ckpt.pth if the job is restarted (checkpoint every --eval-every iterations). Usage:
#   sbatch --job-name=kseg-v2-s1 training/slurm/train_v2.sbatch <out-dir> --seed 1 [train.py args]
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/home/kevin.antony/kelpod-wt/learned-v2/training/logs/%x-%j.out
set -euo pipefail
cd /home/kevin.antony/kelpod-wt/learned-v2/training
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
OUT="$1"; shift
export OMP_NUM_THREADS=1 PYTHONPATH=.
uv run python -m kelpod_train.train --out "$OUT" --workers 14 --encoder resnet34 \
    --iters 64000 --batch 16 \
    --mix synth_v2:0.28,swiss_v2:0.14,resplan_v2:0.14,synth_v1:0.08,swiss_v1:0.04,resplan_v1:0.04,synth:0.02,cubicasa:0.2 \
    --scale-range 0.72,1.4 \
    --val synth,synth_v1,synth_v2,swiss_v1,swiss_v2,cubicasa "$@"
