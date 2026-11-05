#!/usr/bin/env bash
# Harness self-test: proves the egress test can FAIL. Runs stand-in commands instead of the
# pipeline (works before `zllf run` exists):
#   clean    - only loopback traffic (a local TCP echo)            -> harness must PASS
#   connect  - TCP connect to TEST-NET-1 192.0.2.1, error swallowed, exit 0  -> harness must catch it
#   dns      - DNS lookup of a public name, error swallowed, exit 0 -> harness must catch it
#   failing  - clean but exits 1                                    -> harness must FAIL
# usage: deploy/egress/self_test.sh [--mode linux|docker|audit-only] [--image IMG]
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
args=("$@")
py="${PYTHON:-python3}"
[[ " ${args[*]} " == *" --mode docker "* ]] && py=python

# shellcheck disable=SC2034  # read indirectly via ${!cmdvar}
clean="$py -c \"import socket,threading
s=socket.socket(); s.bind(('127.0.0.1',0)); s.listen(1); p=s.getsockname()[1]
threading.Thread(target=lambda: s.accept()[0].sendall(b'ok'), daemon=True).start()
c=socket.create_connection(('127.0.0.1',p), timeout=2); assert c.recv(2)==b'ok'; print('loopback ok')\""
# shellcheck disable=SC2034
connect="$py -c \"import socket
try: socket.create_connection(('192.0.2.1',443), timeout=3); print('CONNECTED')
except OSError as e: print('connect failed (swallowed):', type(e).__name__)\""
# shellcheck disable=SC2034
dns="$py -c \"import socket
try: print(socket.getaddrinfo('huggingface.co',443)[0][4])
except OSError as e: print('dns failed (swallowed):', type(e).__name__)\""
# shellcheck disable=SC2034
failing="$py -c \"import sys; sys.exit(1)\""

base="${EGRESS_SELFTEST_OUT:-$(mktemp -d)}"
mkdir -p "$base"
fails=0
run() {  # label command-var expected_rc [extra args]
  local name=$1 cmdvar=$2 want=$3; shift 3
  "$here/run_egress_test.sh" "${args[@]}" --out "$base/$name" --cmd "${!cmdvar}" "$@" >"$base/$name.log" 2>&1
  local rc=$?
  local verdict
  verdict=$(grep -m1 -oE '\[egress\] (PASS|FAIL)[^ ]*' "$base/$name.log" || echo "?")
  if [[ $rc == "$want" ]]; then
    echo "ok   $name: rc=$rc $verdict"
  else
    echo "FAIL $name: rc=$rc (wanted $want) $verdict -- see $base/$name.log"; fails=$((fails+1))
  fi
}
run clean clean 0
run connect-caught connect 0 --expect-egress
run dns-caught dns 0 --expect-egress
run connect-fails connect 1   # same offender as a normal run must make the test fail
run failing failing 1
echo "evidence: $base"
exit $fails
