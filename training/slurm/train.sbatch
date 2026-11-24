#!/bin/bash
# Fine-tune the Kelpod U-Net on one GPU. Usage:
#   sbatch --job-name=kseg-r34 training/slurm/train.sbatch <out-dir> [extra train.py args]
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/kevin.antony/kelpod/training/logs/%x-%j.out
set -euo pipefail
cd /home/kevin.antony/kelpod/training
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
OUT="$1"; shift
export OMP_NUM_THREADS=1
uv run python -m kelpod_train.train --out "$OUT" --workers 14 "$@"
