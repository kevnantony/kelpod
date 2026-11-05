#!/usr/bin/env bash
# Docker variant (macOS / any Docker host): run EGRESS_CMD in the app image with
# `--network none` (the container gets only `lo`). Sourced by run_egress_test.sh.
# Needs: EGRESS_CMD, EGRESS_OUT, EGRESS_DIR, EGRESS_IMAGE; optional EGRESS_DOCKER_ARGS.
set -euo pipefail

egress_docker_run() {
  command -v docker >/dev/null || { echo "docker not found" >&2; return 2; }
  docker image inspect "$EGRESS_IMAGE" >/dev/null 2>&1 || {
    echo "image $EGRESS_IMAGE not found; build it: docker build -f deploy/docker/Dockerfile.app -t $EGRESS_IMAGE ." >&2
    return 2
  }
  chmod a+rwx "$EGRESS_OUT"
  local -a extra=()
  # shellcheck disable=SC2206  # intentional word splitting of user-supplied docker args
  [[ -n "${EGRESS_DOCKER_ARGS:-}" ]] && extra=(${EGRESS_DOCKER_ARGS})
  local inner
  # Record the container's interfaces first (evidence: only lo is UP), then run the command.
  # Some kernels (e.g. Docker Desktop's) create tunnel fallback devices (tunl0, gre0, sit0, ...)
  # in every namespace; they are DOWN with no address, so we check the IFF_UP flag.
  # shellcheck disable=SC2016  # expanded by the container's shell, not here
  inner='for i in /sys/class/net/*; do [ -d "$i" ] || continue; echo "${i##*/} $(cat "$i/operstate") $(cat "$i/flags")"; done > /egress-out/container-ifaces.txt; cat /proc/net/dev > /egress-out/container-netdev.txt; '"$EGRESS_CMD"
  set +e
  docker run --rm --network none \
    --read-only --tmpfs /tmp:rw,size=2g,mode=1777 \
    --cap-drop ALL --security-opt no-new-privileges \
    -v "$EGRESS_DIR/audit:/opt/zllf-egress-audit:ro" \
    -v "$EGRESS_OUT:/egress-out" \
    -e PYTHONPATH=/opt/zllf-egress-audit \
    -e ZLLF_EGRESS_AUDIT_LOG=/egress-out/audit.jsonl \
    ${extra[@]+"${extra[@]}"} \
    --entrypoint /bin/sh "$EGRESS_IMAGE" -c "$inner" 2>&1 | tee "$EGRESS_OUT/command.log"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "$rc" > "$EGRESS_OUT/command.exit"
  local ifaces
  # interfaces with IFF_UP (0x1) set
  ifaces=$(python3 -c 'import sys
for line in open(sys.argv[1]):
    parts = line.split()
    if len(parts) == 3 and int(parts[2], 16) & 1:
        print(parts[0], end=" ")' "$EGRESS_OUT/container-ifaces.txt" 2>/dev/null || true)
  cat > "$EGRESS_OUT/counters.env" <<VARS
FIREWALL=docker-network-none
CONTAINER_IFACES="${ifaces% }"
VARS
}
