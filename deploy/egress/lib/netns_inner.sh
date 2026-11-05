#!/usr/bin/env bash
# Runs INSIDE a fresh network + mount namespace (created by lib/egress_linux.sh with either
# `sudo unshare --net --mount` or rootless `unshare -rnm`). Not meant to be called directly.
#
# env in:  EGRESS_CMD, EGRESS_OUT, EGRESS_DIR, RUN_AS_UID/RUN_AS_GID (optional, real-root case)
# files out (in $EGRESS_OUT): command.log, command.exit, egress.pcap, pcap-summary.txt,
#           counters.env (EGRESS_ATTEMPTS, DNS_ATTEMPTS, FIREWALL, NONLO_PACKETS, DNS_PACKETS)
set -uo pipefail
out="$EGRESS_OUT"
log() { echo "[netns] $*" >&2; }

ip link set lo up

# 1) dummy uplink so outbound attempts are routed (and counted) instead of ENETUNREACH.
uplink=0
if ip link add egress0 type dummy 2>/dev/null; then
  ip link set egress0 addrgenmode none 2>/dev/null || true   # no IPv6 link-local chatter
  ip addr add 10.255.255.1/24 dev egress0
  ip link set egress0 up
  ip route add default via 10.255.255.254 dev egress0 onlink 2>/dev/null || ip route add default dev egress0
  ip -6 addr add fd00:255::1/64 dev egress0 nodad 2>/dev/null && ip -6 route add default dev egress0 2>/dev/null
  uplink=1
else
  log "WARN: cannot create dummy interface; attempts fail with ENETUNREACH and are only seen by the audit hook"
fi
ip -br addr > "$out/netns-interfaces.txt" 2>&1

# 2) resolver pointing at the dummy net, so DNS lookups are attempts we can count.
resolv="$(mktemp)"
echo "nameserver 10.255.255.53" > "$resolv"
mount --bind "$resolv" /etc/resolv.conf 2>/dev/null || log "WARN: could not override /etc/resolv.conf"

# 3) firewall: log + counter + drop for everything not on lo.
firewall=none
if command -v nft >/dev/null 2>&1 && nft -f "$EGRESS_DIR/nftables-netns.nft" 2>"$out/nft.err"; then
  firewall=nft
elif command -v iptables >/dev/null 2>&1; then
  for ipt in iptables ip6tables; do
    command -v $ipt >/dev/null 2>&1 || continue
    $ipt -A OUTPUT -o lo -j ACCEPT
    $ipt -A OUTPUT -p udp --dport 53 -m comment --comment zllf-dns
    $ipt -A OUTPUT -p tcp --dport 53 -m comment --comment zllf-dns
    $ipt -A OUTPUT -j LOG --log-prefix "ZLLF-EGRESS " 2>/dev/null || true
    $ipt -A OUTPUT -j DROP -m comment --comment zllf-egress
  done
  firewall=iptables
else
  log "WARN: neither nft nor iptables usable; relying on pcap + audit hook"
fi

# 4) packet capture on every interface in the namespace.
tcpdump_pid=""
if command -v tcpdump >/dev/null 2>&1; then
  # -Z root keeps tcpdump from switching to the 'tcpdump' user (which cannot write $out);
  # inside a rootless user namespace setgroups() is denied, so skip it there.
  zarg=(-Z root)
  [[ "$(awk 'NR==1 {print $2}' /proc/self/uid_map)" != 0 ]] && zarg=()
  tcpdump -i any -n -U --immediate-mode "${zarg[@]}" -w "$out/egress.pcap" >/dev/null 2>"$out/tcpdump.err" &
  tcpdump_pid=$!
  for _ in $(seq 1 50); do grep -q "listening on" "$out/tcpdump.err" 2>/dev/null && break; sleep 0.1; done
else
  log "WARN: tcpdump not installed; no pcap"
fi

# 5) the pipeline (as the invoking user when we are real root).
log "running: $EGRESS_CMD"
set +e
if [[ -n "${RUN_AS_UID:-}" && "$(id -u)" == 0 && "${RUN_AS_UID}" != 0 ]]; then
  # shellcheck disable=SC2153  # RUN_AS_GID comes from the environment
  setpriv --reuid "$RUN_AS_UID" --regid "$RUN_AS_GID" --init-groups \
    bash -c "$EGRESS_CMD" > >(tee "$out/command.log") 2>&1
else
  bash -c "$EGRESS_CMD" > >(tee "$out/command.log") 2>&1
fi
rc=$?
sleep 0.5
echo "$rc" > "$out/command.exit"

# 6) evidence.
if [[ -n "$tcpdump_pid" ]]; then
  kill -INT "$tcpdump_pid" 2>/dev/null; wait "$tcpdump_pid" 2>/dev/null
fi
egress=0; dns=0
case "$firewall" in
  nft)
    nft list counter inet zllf_egress egress_attempts > "$out/firewall-counters.txt"
    nft list counter inet zllf_egress dns_attempts >> "$out/firewall-counters.txt"
    nft list counter inet zllf_egress link_housekeeping >> "$out/firewall-counters.txt"
    egress=$(nft list counter inet zllf_egress egress_attempts | grep -oE 'packets [0-9]+' | awk '{print $2}')
    dns=$(nft list counter inet zllf_egress dns_attempts | grep -oE 'packets [0-9]+' | awk '{print $2}')
    ;;
  iptables)
    : > "$out/firewall-counters.txt"
    for ipt in iptables ip6tables; do
      command -v $ipt >/dev/null 2>&1 || continue
      $ipt -L OUTPUT -v -x -n >> "$out/firewall-counters.txt"
      e=$($ipt -L OUTPUT -v -x -n | awk '/zllf-egress/ {s+=$1} END {print s+0}')
      d=$($ipt -L OUTPUT -v -x -n | awk '/zllf-dns/ {s+=$1} END {print s+0}')
      egress=$((egress + e)); dns=$((dns + d))
    done
    ;;
esac
nonlo=0; dnspk=0
if [[ -s "$out/egress.pcap" ]]; then
  # -i any => LINUX_SLL2 frames; tcpdump prints the interface name as the 2nd field.
  tcpdump -n -r "$out/egress.pcap" 2>/dev/null > "$out/pcap-all.txt" || true
  awk '$2 != "lo"' "$out/pcap-all.txt" > "$out/pcap-nonloopback.txt"
  nonlo=$(wc -l < "$out/pcap-nonloopback.txt" | tr -d ' ')
  dnspk=$(tcpdump -n -r "$out/egress.pcap" 'port 53' 2>/dev/null | wc -l | tr -d ' ')
  {
    echo "total_packets=$(wc -l < "$out/pcap-all.txt" | tr -d ' ')"
    echo "non_loopback_packets=$nonlo"
    echo "dns_packets=$dnspk"
    echo "--- non-loopback (first 50) ---"
    head -50 "$out/pcap-nonloopback.txt"
  } > "$out/pcap-summary.txt"
  rm -f "$out/pcap-all.txt"
fi
cat > "$out/counters.env" <<VARS
EGRESS_ATTEMPTS=${egress:-0}
DNS_ATTEMPTS=${dns:-0}
FIREWALL=$firewall
UPLINK=$uplink
NONLO_PACKETS=$nonlo
DNS_PACKETS=$dnspk
PCAP=$([[ -n "$tcpdump_pid" ]] && echo 1 || echo 0)
VARS
chmod -R a+rX "$out" 2>/dev/null || true
exit 0
