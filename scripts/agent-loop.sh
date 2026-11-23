#!/usr/bin/env bash
# Keeps a Claude Code lead agent working through docs/HANDOFF.md until everything is done:
# resumes after usage limits (waits until the stated reset time), pushes the branch after every
# run, and every 10 minutes pushes committed work plus a snapshot of uncommitted work to
# "<branch>-autosave" (never touches main).
#
# Run on the cluster as a Slurm job (survives logouts): sbatch scripts/slurm/agent-loop.sbatch
# Needs: `claude` logged in (CLAUDE_CONFIG_DIR selects the account) and `gh auth login`.
set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
BRANCH=claude/kelpod-mvp-fa1abc
MODEL=${KELPOD_MODEL:-claude-opus-5-5}
CONFIG_DIR=${CLAUDE_CONFIG_DIR:-$HOME/.claude}
mkdir -p logs

PROMPT='You are the lead engineer of the Kelpod MVP. Read CLAUDE.md, docs/HANDOFF.md and
docs/AGENT_BRIEFS.md, then work through every unchecked item in docs/HANDOFF.md, in order, and do
not stop until all are checked.
- Work multi-agent: whenever two or more pieces of work are independent (modules, the benchmark,
  the demo, training experiments, the API, the report), start one subagent per piece IN PARALLEL
  with the shared rules and its brief from docs/AGENT_BRIEFS.md. You review, test, integrate.
- Check whether something already exists (open source) before building it.
- Commit and git push origin '"$BRANCH"' after EVERY meaningful change (a module or fix
  landing, a passing integration step, a doc update): small commits, pushed as often as possible.
  Never push main.
- Keep the status in docs/HANDOFF.md current (newest facts, numbers measured, what is not done).
- GPUs: use Slurm (sbatch) as described in docs/HANDOFF.md; never run GPU work on the login node.
- Never present targets as results. If a usage limit stops you, the loop resumes you later: pick
  up from docs/HANDOFF.md and git log.
When every item is checked, write DONE as the only line of .agent-done and stop.'

snapshot() { # push uncommitted work to <branch>-autosave without touching the index or tree
  local idx tree c
  idx=$(mktemp)
  GIT_INDEX_FILE=$idx git read-tree HEAD && GIT_INDEX_FILE=$idx git add -A . 2>/dev/null
  tree=$(GIT_INDEX_FILE=$idx git write-tree); rm -f "$idx"
  if [ -n "$tree" ] && [ "$tree" != "$(git rev-parse 'HEAD^{tree}')" ]; then
    c=$(git commit-tree "$tree" -p HEAD -m "autosave $(date -Is)") &&
      git push -q -f origin "$c:refs/heads/$BRANCH-autosave"
  fi
}
autosave() { while true; do sleep 600; git push -q origin "$BRANCH" 2>/dev/null; snapshot 2>/dev/null; done; }
autosave & AUTOSAVE_PID=$!
trap 'kill $AUTOSAVE_PID 2>/dev/null' EXIT

wait_for_reset() { # sleep until "resets 8:50pm" from the last run's output, else 15 minutes
  local reset t now
  reset=$(tail -n 20 logs/last-run.log 2>/dev/null | grep -oE 'resets [0-9]{1,2}(:[0-9]{2})?[ap]m' | tail -1 | cut -d' ' -f2)
  if [ -n "$reset" ] && t=$(date -d "$reset" +%s 2>/dev/null); then
    now=$(date +%s); [ "$t" -le "$now" ] && t=$((t + 86400))
    echo "[$(date -Is)] usage limit, sleeping until $reset" | tee -a logs/agent-loop.log
    sleep $((t - now + 120))
  else
    sleep 900
  fi
}

while [ ! -f .agent-done ]; do
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
echo "[$(date -Is)] done" | tee -a logs/agent-loop.log
