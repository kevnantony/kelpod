#!/bin/bash
# CPU evaluation of segmentation backends (GPU partition only because it has free CPUs; the GPU
# is not used, segment runtimes are CPU numbers). Usage:
#   sbatch --job-name=kseg-eval training/slurm/eval.sbatch <evaluate.py args>
#SBATCH --partition=pascal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/kevin.antony/kelpod/training/logs/%x-%j.out
set -euo pipefail
cd /home/kevin.antony/kelpod/training
export CUDA_VISIBLE_DEVICES=""
uv run python -m kelpod_train.evaluate "$@"
