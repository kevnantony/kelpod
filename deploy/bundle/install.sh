#!/usr/bin/env bash
# Appliance installer (P3.5.3) - SKELETON. Verifies first, then loads. Without --yes it is a
# dry run: prerequisite checks + bundle verification only, nothing on the host changes.
#
# usage: sudo deploy/bundle/install.sh --dir DIR --pubkey PEM [--yes] [--gpu] [--apply-firewall]
#                                      [--min-free-gb N] [--allow-unsigned]
# Steps: prerequisites -> verify_bundle.sh -> docker load -> models into the volume
#        -> host egress firewall (opt.) -> compose up -> health check -> egress self-test.
# TODO(P3.5.3/3.5.4): LUKS volume, TLS cert + admin bootstrap token, upgrade.sh with rollback.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=SCRIPTDIR/lib.sh
source "$here/lib.sh"

dir=""; pubkey=""; yes=0; gpu=0; firewall=0; min_free_gb=100; verify_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir) dir="$2"; shift ;;
    --pubkey) pubkey="$2"; shift ;;
    --yes) yes=1 ;;
    --gpu) gpu=1 ;;
    --apply-firewall) firewall=1 ;;
    --min-free-gb) min_free_gb="$2"; shift ;;
    --allow-unsigned) verify_args+=(--allow-unsigned) ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done
[[ -n "$dir" ]] || die "--dir DIR required"
image="${ZLLF_IMAGE:-zllf-app:dev}"

# ---- prerequisites
fail=0
check() {  # description, command...
  local desc=$1; shift
  if "$@" >/dev/null 2>&1; then log "ok    $desc"; else log "FAIL  $desc"; fail=1; fi
}
free_gb() { df -Pk "${ZLLF_DOCKER_ROOT:-/var/lib/docker}" 2>/dev/null | awk 'NR==2 {print int($4/1048576)}'; }
has_free() { [[ "$(free_gb)" -ge "$min_free_gb" ]]; }
nvidia_runtime() { docker info --format '{{json .Runtimes}}' | grep -q nvidia; }

check "Linux host" test "$(uname -s)" = Linux
check "docker CLI" command -v docker
check "docker daemon reachable" docker info
check "docker compose v2" docker compose version
check "python3 (verification)" command -v python3
check "nft (host egress firewall)" command -v nft
check "free disk >= ${min_free_gb} GB under ${ZLLF_DOCKER_ROOT:-/var/lib/docker}" has_free
if [[ $gpu == 1 ]]; then
  check "NVIDIA driver (nvidia-smi)" nvidia-smi
  check "NVIDIA container toolkit runtime" nvidia_runtime
fi
[[ $fail == 0 ]] || die "prerequisites not met"

# ---- verify before loading anything
bundle="$("$here/verify_bundle.sh" --dir "$dir" ${pubkey:+--pubkey "$pubkey"} \
  ${verify_args[@]+"${verify_args[@]}"} --extract-to "${ZLLF_STAGING:-/var/tmp/zllf-install}")"
log "verified bundle at $bundle"
if [[ $yes == 0 ]]; then
  log "dry run complete (prerequisites + verification). Re-run with --yes to install."
  exit 0
fi

# ---- install
for img in "$bundle"/images/*.tar.gz; do
  [[ -e "$img" ]] || continue
  log "docker load $(basename "$img")"
  gunzip -c "$img" | docker load
done
if [[ -d "$bundle/models/hub" ]]; then
  log "copying models into volume zllf_zllf-models"
  docker volume create zllf_zllf-models >/dev/null
  docker run --rm --network none --user 0:0 -v zllf_zllf-models:/dst \
    -v "$bundle/models:/src:ro" --entrypoint /bin/sh "$image" -c 'cp -R /src/. /dst/'
fi
if [[ $firewall == 1 ]]; then
  log "applying host egress firewall"
  nft -f "$bundle/deploy/egress/nftables-appliance.nft"
fi
ZLLF_IMAGE="$image" ZLLF_BIND="${ZLLF_BIND:-0.0.0.0}" \
  docker compose -f "$bundle/deploy/compose/compose.yaml" up -d
healthy=0
for _ in $(seq 1 60); do
  if docker compose -f "$bundle/deploy/compose/compose.yaml" exec -T api python -c \
     "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/system/health', timeout=3)" \
     >/dev/null 2>&1; then healthy=1; break; fi
  sleep 2
done
[[ $healthy == 1 ]] || die "api did not become healthy"
log "api healthy; running egress self-test (docker variant)"
mkdir -p /var/log/zllf
ZLLF_IMAGE="$image" "$bundle/deploy/egress/run_egress_test.sh" --mode docker \
  --image "$image" --out "/var/log/zllf/egress-install-$(date -u +%Y%m%dT%H%M%SZ)" \
  || die "egress self-test FAILED"
log "installed. UI: http://<host>:${ZLLF_PORT:-8080}/"
