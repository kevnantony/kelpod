#!/bin/bash
# Runs the Kelpod agent loop as a long CPU job (survives logouts). Submit from the repo root:
#   sbatch scripts/slurm/agent-loop.sbatch
# The Claude account is chosen by CLAUDE_CONFIG_DIR (log in once: CLAUDE_CONFIG_DIR=... claude).
#SBATCH -J kelpod-agent
#SBATCH -p pascal
#SBATCH --exclude=pascal-node03.l3s.intra
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH --time=7-00:00:00
#SBATCH -o logs/slurm-agent-%j.out
export CLAUDE_CONFIG_DIR=${CLAUDE_CONFIG_DIR:-$HOME/.claude-kelpod}
export KELPOD_MODEL=${KELPOD_MODEL:-claude-opus-5-5}
cd "${SLURM_SUBMIT_DIR:-$HOME/kelpod}"
bash scripts/agent-loop.sh
