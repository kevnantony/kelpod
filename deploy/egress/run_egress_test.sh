#!/usr/bin/env bash
# ZLLF egress test (NFR-SEC-01, DEVELOPMENT_PLAN P1.10.4/5). See docs/runbooks/egress-test.md.
#
# usage: deploy/egress/run_egress_test.sh [options]
#   --mode MODE     auto (default) | linux | docker | audit-only
#                   auto = linux on Linux, docker elsewhere (macOS)
#   --cmd CMD       command to run (default: $ZLLF_EGRESS_CMD or the fake-profile pipeline)
#   --out DIR       evidence directory (default: egress-out/<UTC timestamp>)
#   --image IMG     image for --mode docker (default: $ZLLF_IMAGE or zllf-app:dev)
#   --expect-egress invert the verdict (harness self-test: the command MUST be caught)
#
# PASS iff: command exits 0 AND 0 firewall-counted egress attempts AND 0 DNS attempts AND
# 0 non-loopback packets in the pcap AND 0 non-loopback attempts in the Python audit log.
# Exit: 0 pass, 1 fail, 2 harness/setup error. Evidence + egress-report.json in --out.
set -euo pipefail

EGRESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EGRESS_DIR/../.." && pwd)"
DEFAULT_CMD="zllf run --corpus tests/fixtures/corpus --hardware fake --auto-approve --runs-dir /tmp/runs"

mode=auto
cmd="${ZLLF_EGRESS_CMD:-$DEFAULT_CMD}"
out=""
image="${ZLLF_IMAGE:-zllf-app:dev}"
expect_egress=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) mode="$2"; shift ;;
    --cmd) cmd="$2"; shift ;;
    --out) out="$2"; shift ;;
    --image) image="$2"; shift ;;
    --expect-egress) expect_egress=1 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

if [[ $mode == auto ]]; then
  if [[ "$(uname -s)" == Linux ]]; then mode=linux; else mode=docker; fi
fi
out="${out:-$REPO_ROOT/egress-out/$(date -u +%Y%m%dT%H%M%SZ)-$mode}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"
: > "$out/audit.jsonl"
rm -f "$out/command.exit" "$out/counters.env"

export EGRESS_DIR EGRESS_OUT="$out" EGRESS_CMD="$cmd" EGRESS_IMAGE="$image"
# Python-level attempt recorder (all modes; inside Docker it is mounted separately).
export PYTHONPATH="$EGRESS_DIR/audit${PYTHONPATH:+:$PYTHONPATH}"
export ZLLF_EGRESS_AUDIT_LOG="$out/audit.jsonl"
# Make the project's venv (zllf entry point) visible for host modes.
[[ -d "$REPO_ROOT/.venv/bin" ]] && export PATH="$REPO_ROOT/.venv/bin:$PATH"

echo "[egress] mode=$mode out=$out"
echo "[egress] cmd: $cmd"
cd "$REPO_ROOT"
case "$mode" in
  linux)
    [[ "$(uname -s)" == Linux ]] || { echo "--mode linux needs Linux" >&2; exit 2; }
    # shellcheck source=SCRIPTDIR/lib/egress_linux.sh
    source "$EGRESS_DIR/lib/egress_linux.sh"
    egress_linux_run || exit 2
    ;;
  docker)
    # shellcheck source=SCRIPTDIR/lib/egress_docker.sh
    source "$EGRESS_DIR/lib/egress_docker.sh"
    egress_docker_run || exit 2
    ;;
  audit-only)
    echo "[egress] WARNING: audit-only mode is NOT an isolation test (host network is up);" \
         "it only records Python-level attempts." >&2
    set +e
    bash -c "$cmd" 2>&1 | tee "$out/command.log"
    echo "${PIPESTATUS[0]}" > "$out/command.exit"
    set -e
    echo "FIREWALL=none" > "$out/counters.env"
    ;;
  *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac

[[ -f "$out/command.exit" ]] || { echo "[egress] harness error: no command.exit" >&2; exit 2; }

python3 - "$out" "$mode" "$cmd" "$expect_egress" <<'PY'
import json, shlex, sys
from pathlib import Path

out, mode, cmd, expect = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4] == "1"
env = {}
for line in (out / "counters.env").read_text().splitlines():
    if "=" in line:
        k, v = line.split("=", 1)
        env[k] = shlex.split(v)[0] if v.strip() else ""
def num(key):
    try:
        return int(env.get(key, 0) or 0)
    except ValueError:
        return 0
audit = [json.loads(l) for l in (out / "audit.jsonl").read_text().splitlines() if l.strip()]
rc = int((out / "command.exit").read_text().strip() or 1)
ifaces = env.get("CONTAINER_IFACES", "").split()
checks = {
    "command_exit_zero": rc == 0,
    "firewall_egress_attempts_zero": num("EGRESS_ATTEMPTS") == 0,
    "dns_attempts_zero": num("DNS_ATTEMPTS") == 0 and num("DNS_PACKETS") == 0,
    "pcap_non_loopback_packets_zero": num("NONLO_PACKETS") == 0,
    "python_audit_attempts_zero": len(audit) == 0,
}
if mode == "docker":
    checks["container_only_loopback"] = bool(ifaces) and set(ifaces) <= {"lo"}
egress_seen = not all(v for k, v in checks.items() if k != "command_exit_zero")
passed = all(checks.values())
if expect:  # self-test: an offending command must be caught by at least one detector
    verdict = "PASS(caught)" if egress_seen else "FAIL(missed)"
    ok = egress_seen
else:
    verdict = "PASS" if passed else "FAIL"
    ok = passed
report = {
    "test": "zllf-egress", "mode": mode, "command": cmd, "command_exit": rc,
    "expect_egress": expect, "verdict": verdict,
    "firewall": env.get("FIREWALL"), "dummy_uplink": env.get("UPLINK"),
    "pcap_captured": env.get("PCAP") == "1",
    "counters": {"egress_attempts": num("EGRESS_ATTEMPTS"), "dns_attempts": num("DNS_ATTEMPTS"),
                 "non_loopback_packets": num("NONLO_PACKETS"), "dns_packets": num("DNS_PACKETS"),
                 "python_audit_attempts": len(audit)},
    "container_interfaces": ifaces or None,
    "audit_sample": audit[:20],
    "checks": checks,
}
(out / "egress-report.json").write_text(json.dumps(report, indent=2) + "\n")
print(f"[egress] {verdict}  mode={mode} exit={rc} " + " ".join(
    f"{k}={v}" for k, v in report["counters"].items()))
for k, v in checks.items():
    print(f"[egress]   {'ok  ' if v else 'FAIL'} {k}")
sys.exit(0 if ok else 1)
PY
