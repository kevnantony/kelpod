#!/usr/bin/env bash
# Keeps a Claude Code lead agent working through docs/HANDOFF.md until everything is done:
# resumes after usage limits (waits until the stated reset time), pushes the branch after every
# run, and every hour pushes committed work plus a snapshot of uncommitted work to
# "<branch>-autosave" (never touches main).
#
# Run on the cluster as a Slurm job (survives logouts): sbatch scripts/slurm/agent-loop.sbatch
# Needs: `claude` logged in (CLAUDE_CONFIG_DIR selects the account) and `gh auth login`.
set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
BRANCH=$(git rev-parse --abbrev-ref HEAD)  # re-read every iteration below
MODEL=${KELPOD_MODEL:-claude-opus-5-5}
CONFIG_DIR=${CLAUDE_CONFIG_DIR:-$HOME/.claude}
mkdir -p logs
# In -p mode Claude Code stops background subagents 10 min after the lead's last turn; that killed
# training and fix subagents mid-work. Wait for them instead.
export CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0

PROMPT='You are the lead engineer of Kelpod, running continuously on the cluster until a person
stops you. Read CLAUDE.md, docs/HANDOFF.md and docs/AGENT_BRIEFS.md.
Phase 1: work through every unchecked MVP item in docs/HANDOFF.md, in order.
Phase 2 (once the MVP items are done; never ends): keep improving accuracy on real German
Bestandspläne, following "Continuous improvement" in docs/HANDOFF.md. Run experiments, keep what
beats the benchmark, revert what does not, log every experiment in docs/EXPERIMENTS.md, repeat.
- Work multi-agent: whenever two or more pieces of work are independent (modules, experiments,
  training runs, evaluation, research, docs), start one subagent per piece IN PARALLEL with the
  shared rules and a clear brief. You review, test, integrate.
- Check whether something already exists (open source, papers) before building it.
- Commit and push about once an hour (bundle the hour's work; also before a promotion and before
  stopping for a usage limit). No need to commit after every small change.
- Branches: main is the baseline. Each experiment runs on its own exp/<name> branch in its own git
  worktree (~/kelpod-wt/<name>). Merge into main ONLY by promotion: scripts/promotion_check.sh must
  pass under docs/EVAL_PROTOCOL.md; then merge --no-ff, tag baseline-vN, push; new experiments
  branch from the new main. Never weaken or game the evaluation; eval changes go on eval/* branches.
- Keep docs/HANDOFF.md current, and keep the Status block in README.md current (Done / Doing now /
  Next, timestamp; see CLAUDE.md), pushed after every step so the team can follow on GitHub.
- GPUs: use Slurm (sbatch) as described in docs/HANDOFF.md; never run GPU work on the login node.
  You may use as much of the pascal cluster as is justified: write the justification and expected
  GPU hours per job in docs/EXPERIMENTS.md, checkpoint long jobs, cancel idle ones.
- Never present targets as results. If a usage limit stops you, the loop resumes you later: pick
  up from docs/HANDOFF.md, docs/EXPERIMENTS.md and git log. Never write .agent-done; only a person
  stops this loop (scripts/cockroach.sh stop).'

snapshot() { # push uncommitted work to <branch>-autosave without touching the index or tree
  local idx tree c
  idx=$(mktemp)
  GIT_INDEX_FILE=$idx git read-tree HEAD && GIT_INDEX_FILE=$idx git add -A . 2>/dev/null
  tree=$(GIT_INDEX_FILE=$idx git write-tree); rm -f "$idx"
  if [ -n "$tree" ] && [ "$tree" != "$(git rev-parse 'HEAD^{tree}')" ]; then
    c=$(git commit-tree "$tree" -p HEAD -m "autosave $(date -Is)") &&
      git push -q -f origin "$c:refs/heads/$(git rev-parse --abbrev-ref HEAD)-autosave"
  fi
}
autosave() { while true; do sleep 3600; git push -q origin HEAD 2>/dev/null; snapshot 2>/dev/null; done; }
autosave & AUTOSAVE_PID=$!
trap 'kill $AUTOSAVE_PID 2>/dev/null' EXIT

wait_for_reset() { # sleep until the reset time in the last run's output ("resets 8:50pm",
  # "resets Nov 30, 8pm"), else retry in 15 minutes
  local reset t now
  reset=$(tail -n 20 logs/last-run.log 2>/dev/null | grep -oE 'resets [^(·]*[0-9]{1,2}(:[0-9]{2})?[ap]m' | tail -1 | sed 's/^resets //; s/,//')
  if [ -n "$reset" ] && t=$(date -d "$reset" +%s 2>/dev/null); then
    now=$(date +%s); [ "$t" -le "$now" ] && t=$((t + 86400))
    echo "[$(date -Is)] usage limit, sleeping until $reset" | tee -a logs/agent-loop.log
    sleep $((t - now + 120))
  else
    sleep 900
  fi
}

# Only one agent loop at a time: if an older kelpod-agent job is already running, leave.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  older=$(squeue -u "$USER" -h -n kelpod-agent -t RUNNING -o %i | awk -v me="$SLURM_JOB_ID" '$1 < me' | head -1)
  [ -n "$older" ] && { echo "[$(date -Is)] job $older already runs the loop; exiting" | tee -a logs/agent-loop.log; exit 0; }
fi
bash scripts/cockroach.sh check  # make sure the watchdog is up

while [ ! -f .agent-stop ]; do
  BRANCH=$(git rev-parse --abbrev-ref HEAD)
  git pull -q --rebase --autostash origin "$BRANCH" 2>>logs/agent-loop.log || true
  CONTINUE=""
  ls "$CONFIG_DIR"/projects/*kelpod*/*.jsonl >/dev/null 2>&1 && CONTINUE="--continue"
  echo "[$(date -Is)] run (model $MODEL, account dir $CONFIG_DIR) $CONTINUE" | tee -a logs/agent-loop.log
  claude -p "$PROMPT" $CONTINUE --model "$MODEL" --permission-mode acceptEdits \
    --allowedTools "Bash Read Edit Write Glob Grep Agent Task WebSearch WebFetch" \
    >logs/last-run.log 2>&1
  status=$?
  cat logs/last-run.log >>logs/agent-loop.log
  git push -q origin "$BRANCH" 2>>logs/agent-loop.log || true
  snapshot 2>/dev/null || true
  if [ $status -ne 0 ] || grep -q "hit your .*limit" logs/last-run.log; then wait_for_reset; else sleep 30; fi
done
echo "[$(date -Is)] stop flag found, loop ends" | tee -a logs/agent-loop.log
