#!/bin/bash
# learned-v1 evaluation, CPU only (no GPU requested): ONNX export of both seeds, dev-val IoU
# (r34 vs seeds), frozen val-synth panel per seed, historic-plan overlays. Submitted with
# --dependency=afterok:<both training jobs>. Usage: sbatch training/slurm/eval_v1.sbatch
#SBATCH --job-name=kseg-v1-eval
#SBATCH --partition=pascal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/home/kevin.antony/kelpod-wt/learned-v1/training/logs/%x-%j.out
set -uo pipefail
cd /home/kevin.antony/kelpod-wt/learned-v1/training
export PYTHONPATH=. OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
M=/home/kevin.antony/kelpod-models/kelpod-seg
R=runs/v1; mkdir -p $R
for s in 1 2; do uv run python -m kelpod_train.export_onnx $M/v1-s$s 2>&1 | grep -v Warn | tail -2; done
for m in r34 v1-s1 v1-s2; do
  uv run python -m kelpod_train.val_iou $M/$m/model.pt --threads 16 > $R/val_iou_$m.txt 2>&1; tail -6 $R/val_iou_$m.txt
done
uv run python -m kelpod_train.historic --images ~/kelpod-data/real/user-feedback-2026-11-24 \
  --models r34=$M/r34-onnx/model.onnx v1-s1=$M/v1-s1/model.onnx v1-s2=$M/v1-s2/model.onnx \
  --out ~/kelpod-runs/historic 2>&1 | grep '^{'
for s in 1 2; do
  KELPOD_SEG_CHECKPOINT=$M/v1-s$s/model.onnx OMP_NUM_THREADS=2 uv run python -m kelpod_train.val_synth \
    --backends learned --workers 8 --threads 2 --out $R/val_synth_s$s 2>&1 | tail -25
done
echo EVALDONE
