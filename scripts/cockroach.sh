#!/usr/bin/env bash
# cockroach: keeps the Kelpod agent alive on the cluster until you say stop.
#
# Three layers, each able to bring the others back:
#   1. the agent job (scripts/slurm/agent-loop.sbatch) waits out usage limits itself and, when it
#      starts, makes sure a watchdog job exists;
#   2. a tiny watchdog Slurm job (kelpod-cockroach, 1 CPU) checks every 5 minutes and resubmits the
#      agent job if it is gone (time limit, node failure, scancel, crash);
#   3. a cron entry on the login node runs `check` every 10 minutes and revives both.
#
# Usage (from anywhere):
#   scripts/cockroach.sh start    remove the stop flag, install cron, bring everything up
#   scripts/cockroach.sh stop     set the stop flag, cancel agent + watchdog, remove cron entry
#   scripts/cockroach.sh status   show jobs, flag and the last loop log lines
#   scripts/cockroach.sh check    one pass (used by cron, the watchdog and the agent job)
#   scripts/cockroach.sh watch    the watchdog loop (run inside the kelpod-cockroach job)
set -u
USER=${USER:-$(id -un)}; export USER
REPO=$(cd "$(dirname "$0")/.." && pwd)
STOP="$REPO/.agent-stop"
LOG="$REPO/logs/cockroach.log"
AGENT=kelpod-agent
DOG=kelpod-cockroach
CRON_LINE="*/10 * * * * bash $REPO/scripts/cockroach.sh check >/dev/null 2>&1"
mkdir -p "$REPO/logs"

log() { echo "[$(date -Is)] $*" >>"$LOG"; }
alive() { squeue -u "$USER" -h -n "$1" -t PENDING,RUNNING,CONFIGURING,COMPLETING -o %i 2>/dev/null | grep -q .; }

check() {
  [ -f "$STOP" ] && return 0
  if ! alive "$AGENT"; then
    out=$(cd "$REPO" && sbatch scripts/slurm/agent-loop.sbatch 2>&1); log "agent missing -> $out"
  fi
  if [ "${SLURM_JOB_NAME:-}" != "$DOG" ] && ! alive "$DOG"; then
    out=$(cd "$REPO" && sbatch -J "$DOG" -p pascal --exclude=pascal-node03.l3s.intra -c 1 --mem=512M \
      --time=30-00:00:00 -o logs/slurm-cockroach-%j.out --wrap "bash scripts/cockroach.sh watch" 2>&1)
    log "watchdog missing -> $out"
  fi
}

case "${1:-status}" in
  check) check ;;
  watch)
    log "watchdog up (job ${SLURM_JOB_ID:-?})"
    while [ ! -f "$STOP" ]; do check; sleep 300; done
    log "watchdog: stop flag found, exiting" ;;
  start)
    rm -f "$STOP"
    (crontab -l 2>/dev/null | grep -v 'cockroach.sh check'; echo "$CRON_LINE") | crontab -
    check; log "start"; "$0" status ;;
  stop)
    touch "$STOP"
    crontab -l 2>/dev/null | grep -v 'cockroach.sh check' | crontab -
    scancel -u "$USER" -n "$DOG" 2>/dev/null; scancel -u "$USER" -n "$AGENT" 2>/dev/null
    log "stop"
    echo "Stopped agent and watchdog. GPU jobs the agent submitted keep running; see: squeue -u $USER" ;;
  status)
    echo "stop flag: $([ -f "$STOP" ] && echo SET || echo not set)"
    echo "cron:      $(crontab -l 2>/dev/null | grep -c 'cockroach.sh check') entry"
    squeue -u "$USER" -o "%.8i %.16j %.2t %.10M %R" | grep -E "JOBID|$AGENT|$DOG|kelpod" || true
    echo "--- agent loop log"; tail -n 5 "$REPO/logs/agent-loop.log" 2>/dev/null
    echo "--- cockroach log"; tail -n 5 "$LOG" 2>/dev/null ;;
  *) echo "usage: $0 start|stop|status|check|watch" >&2; exit 2 ;;
esac
