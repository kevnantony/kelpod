#!/usr/bin/env bash
# Linux variant: run EGRESS_CMD in a throw-away network namespace that has only loopback plus a
# dummy uplink whose traffic is logged + counted + dropped by nftables (iptables fallback), with
# tcpdump on all interfaces. Sourced by run_egress_test.sh (needs: EGRESS_CMD, EGRESS_OUT,
# EGRESS_DIR). Uses sudo (CI runners, appliance hosts) or falls back to rootless `unshare -rnm`.
set -euo pipefail

egress_linux_run() {
  command -v unshare >/dev/null || { echo "unshare (util-linux) not found" >&2; return 2; }
  command -v ip >/dev/null || { echo "ip (iproute2) not found" >&2; return 2; }
  local -a pre
  local run_uid="" run_gid=""
  if [[ "$(id -u)" == 0 ]]; then
    pre=()
  elif sudo -n true 2>/dev/null; then
    pre=(sudo -E)
    run_uid="$(id -u)"; run_gid="$(id -g)"
  else
    pre=()
    echo "[linux] no passwordless sudo: using rootless 'unshare -rnm' (user namespace)" >&2
  fi
  ${pre[@]+"${pre[@]}"} modprobe dummy 2>/dev/null || true

  local -a ns_cmd
  if [[ ${#pre[@]} -gt 0 || "$(id -u)" == 0 ]]; then
    ns_cmd=(unshare --net --mount --propagation private)
  else
    ns_cmd=(unshare --map-root-user --net --mount --propagation private)
  fi
  ${pre[@]+"${pre[@]}"} env "PATH=$PATH" "HOME=$HOME" \
    "EGRESS_CMD=$EGRESS_CMD" "EGRESS_OUT=$EGRESS_OUT" "EGRESS_DIR=$EGRESS_DIR" \
    "RUN_AS_UID=$run_uid" "RUN_AS_GID=$run_gid" \
    "PYTHONPATH=${PYTHONPATH:-}" "ZLLF_EGRESS_AUDIT_LOG=${ZLLF_EGRESS_AUDIT_LOG:-}" \
    "${ns_cmd[@]}" -- bash "$EGRESS_DIR/lib/netns_inner.sh"
}
