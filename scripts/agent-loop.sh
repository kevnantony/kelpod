#!/usr/bin/env bash
# Keeps a Claude Code agent working on docs/HANDOFF.md, resuming after usage limits.
# Run on a server inside tmux:  tmux new -s kelpod 'bash scripts/agent-loop.sh'
# Needs: `claude` logged in, and `gh auth login` (or an SSH key) so it can push the branch.
set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
BRANCH=claude/kelpod-mvp-fa1abc
PROMPT='You are the lead engineer of the Kelpod MVP. Read docs/HANDOFF.md, CLAUDE.md and
docs/AGENT_BRIEFS.md. Work through the unchecked items in docs/HANDOFF.md. Rule: check whether
something has already been done (open source) before building it. So FIRST run the research
subagent from docs/AGENT_BRIEFS.md until docs/research/OPEN_SOURCE_SURVEY.md exists and commit it.
THEN, for modules that are not finished, start one subagent per module IN PARALLEL with the shared rules and its brief from
docs/AGENT_BRIEFS.md. When they
report back, review their work, run uv run pytest, integrate, update the status in
docs/HANDOFF.md, commit, and git push origin '"$BRANCH"' after every finished step (never main).
When every item is checked, write DONE as the only line of .agent-done and stop.'
mkdir -p logs
while [ ! -f .agent-done ]; do
  git pull -q --rebase origin "$BRANCH" || true
  echo "[$(date -Is)] run" | tee -a logs/agent-loop.log
  claude -p "$PROMPT" --continue --permission-mode acceptEdits \
    --allowedTools "Bash Read Edit Write Glob Grep Agent Task WebSearch WebFetch" \
    >>logs/agent-loop.log 2>&1
  status=$?
  git push -q origin "$BRANCH" 2>>logs/agent-loop.log || true
  # Non-zero usually means a usage limit or a crash: wait and retry.
  if [ $status -ne 0 ]; then sleep 1800; else sleep 60; fi
done
echo "[$(date -Is)] done" | tee -a logs/agent-loop.log
