#!/bin/bash
# CPU-only build of style-randomised training pages (learned-v1). No GPU requested. Usage:
#   sbatch --job-name=kbuild-v1 training/slurm/build_v1.sbatch <source> <out> [build.py args]
#SBATCH --partition=pascal
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/kevin.antony/kelpod-wt/learned-v1/training/logs/%x-%j.out
set -euo pipefail
cd /home/kevin.antony/kelpod-wt/learned-v1/training
ls /usr/share/fonts/opentype/urw-base35 | wc -l
SRC="$1"; OUT="$2"; shift 2
export OMP_NUM_THREADS=1 PYTHONPATH=.
uv run python -m kelpod_train.build "$SRC" --out "$OUT" --workers ${SLURM_CPUS_PER_TASK:-6} "$@"
