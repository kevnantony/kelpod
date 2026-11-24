#!/bin/bash
# Runs the Kelpod agent loop as a long CPU job (survives logouts). Submit from the repo root:
#   sbatch scripts/slurm/agent-loop.sbatch
# The Claude account is chosen by CLAUDE_CONFIG_DIR (log in once: CLAUDE_CONFIG_DIR=... claude).
# Sized from measurements (2026-11-24): ~1.1 CPUs average, 17.5 GB peak over 14 h. Heavy
# evaluation or training runs go to their own sbatch jobs, sized to the work.
#SBATCH -J kelpod-agent
#SBATCH -p pascal
#SBATCH --exclude=pascal-node03.l3s.intra
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH --time=7-00:00:00
#SBATCH -o logs/slurm-agent-%j.out
export CLAUDE_CONFIG_DIR=${CLAUDE_CONFIG_DIR:-$HOME/.claude-kelpod}
export KELPOD_MODEL=${KELPOD_MODEL:-claude-opus-5-5}
# Spend the usage limit wisely: Opus leads (plans, reviews, integrates, decides promotions);
# subagents run on Sonnet by default and may pick Haiku for mechanical work.
export CLAUDE_CODE_SUBAGENT_MODEL=${CLAUDE_CODE_SUBAGENT_MODEL:-claude-sonnet-5-5}
cd "${SLURM_SUBMIT_DIR:-$HOME/kelpod}"
bash scripts/agent-loop.sh
