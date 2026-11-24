#!/bin/bash
# learned-v1: fine-tune the U-Net on style-randomised data, one seed per job. Resumes from
# <out>/ckpt.pth if the job is restarted (checkpoint every --eval-every iterations). Usage:
#   sbatch --job-name=kseg-v1-s1 training/slurm/train_v1.sbatch <out-dir> --seed 1 [train.py args]
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/home/kevin.antony/kelpod-wt/learned-v1/training/logs/%x-%j.out
set -euo pipefail
cd /home/kevin.antony/kelpod-wt/learned-v1/training
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
OUT="$1"; shift
export OMP_NUM_THREADS=1 PYTHONPATH=.
uv run python -m kelpod_train.train --out "$OUT" --workers 14 --encoder resnet34 \
    --iters 40000 --batch 16 \
    --mix synth:0.08,synth_v1:0.25,swiss:0.05,swiss_v1:0.2,resplan:0.04,resplan_v1:0.18,cubicasa:0.2 \
    --val synth,synth_v1,swiss,swiss_v1,cubicasa "$@"
