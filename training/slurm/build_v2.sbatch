#!/bin/bash
# CPU-only build of learned-v2 pages (no GPU). Usage:
#   sbatch --job-name=kbuild-v2-x training/slurm/build_v2.sbatch <source> <out> [build.py args]
#SBATCH --partition=pascal
#SBATCH --exclude=pascal-node02.l3s.intra
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/kevin.antony/kelpod-wt/learned-v2/training/logs/%x-%j.out
set -euo pipefail
cd /home/kevin.antony/kelpod-wt/learned-v2/training
SRC="$1"; OUT="$2"; shift 2
export OMP_NUM_THREADS=1 PYTHONPATH=.
uv run python -m kelpod_train.build "$SRC" --out "$OUT" --workers ${SLURM_CPUS_PER_TASK:-8} "$@"
