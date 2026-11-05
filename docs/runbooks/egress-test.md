# Runbook: egress test (NFR-SEC-01)

Goal: show that the pipeline finishes with **zero outbound network traffic**. The test runs a
command (the fake-profile pipeline by default) in isolation, collects evidence, and returns
PASS only if the command succeeded **and** nothing tried to leave the machine.

```bash
deploy/egress/run_egress_test.sh                      # auto: linux on Linux, docker on macOS
deploy/egress/run_egress_test.sh --mode linux         # network namespace + nftables + tcpdump
deploy/egress/run_egress_test.sh --mode docker        # docker run --network none (needs image)
deploy/egress/self_test.sh --mode linux|docker        # proves the harness can FAIL
make egress-test
```

Default command (change with `--cmd` or `ZLLF_EGRESS_CMD`):

```
zllf run --corpus tests/fixtures/corpus --hardware fake --auto-approve --runs-dir /tmp/runs
```

For a real-model run on a GPU/Linux host use for example
`--cmd "zllf run --corpus bench/corpora/data/ref-500 --hardware <gpu-profile> --auto-approve --runs-dir /tmp/runs"`.

Exit codes: `0` PASS, `1` FAIL (egress seen or command failed), `2` setup error. Evidence goes to
`--out` (default `egress-out/<UTC time>-<mode>/`); CI uploads it as the `egress-evidence`
artefact.

| File | Content |
|---|---|
| `egress-report.json` | verdict, every check, counters, first 20 audit records |
| `command.log`, `command.exit` | pipeline output and exit code |
| `firewall-counters.txt` | nft counters `egress_attempts`, `dns_attempts`, `link_housekeeping` (linux) |
| `egress.pcap`, `pcap-summary.txt` | capture on all interfaces in the namespace (linux) |
| `netns-interfaces.txt` / `container-ifaces.txt` | interfaces seen by the command |
| `audit.jsonl` | Python-level network attempts (host + port only, never payloads) |

## PASS criteria

All must hold: command exit 0; nft `egress_attempts` = 0; DNS attempts and DNS packets = 0;
non-loopback packets in the pcap = 0; Python audit records = 0; in docker mode, only `lo` is UP
inside the container.

## What each variant proves

### `--mode linux` (CI, appliance hosts) — the strongest one
`unshare --net --mount` (via sudo, or rootless `unshare -rnm` without sudo) gives the command a
fresh network namespace with:

- `lo`, plus a **dummy uplink `egress0`** with a default route (IPv4 and IPv6) and
  `/etc/resolv.conf` bind-mounted to `nameserver 10.255.255.53`. Outbound connects and DNS
  lookups are therefore *routed* (not refused with ENETUNREACH) and reach the firewall.
- `deploy/egress/nftables-netns.nft`: on `output`, everything not on `lo` is
  **logged + counted + dropped**; DNS gets its own counter. Kernel IPv6 ND/MLD on the dummy link
  is counted separately as `link_housekeeping` and does not fail the test.
- `tcpdump -i any` writes `egress.pcap`.

Proves: the process tree had no path to any network, **and** it did not even try (any attempt,
including a DNS lookup or a connect whose error a library swallowed, shows up in the counters).
Covers every language, not just Python. Does **not** prove anything about the real host
network config of a production machine: that is what `nftables-appliance.nft` is for.

With sudo the namespace is created as root and the command runs as the invoking user (via
`setpriv`); without sudo the command runs as mapped root inside a user namespace. On Ubuntu
24.04 unprivileged user namespaces are restricted by AppArmor, so use sudo there (GitHub
runners have passwordless sudo). Needs `iproute2`, `util-linux`, `nftables` (or `iptables`
fallback) and `tcpdump`; without `tcpdump`/nft the test still runs but reports weaker evidence
(`pcap_captured: false`, `firewall: none`).

### `--mode docker` (macOS dev machine, P1.10.5 second line of evidence)
`docker run --network none --read-only --cap-drop ALL --security-opt no-new-privileges` with the
app image (`--image`, default `zllf-app:dev`, build it with
`docker build -f deploy/docker/Dockerfile.app -t zllf-app:dev .`). Runs as the image user
(uid 10001).

Proves: the pipeline completes inside the **shipped image** with only loopback available, as
non-root, on a read-only root filesystem, with the offline env from DEVELOPMENT_PLAN Appendix A.
Attempts are caught by the Python audit hook (they cannot leave: there is no interface).
Does **not** count non-Python attempts (a C library's failed connect is invisible apart from a
possible error), and there is no packet capture. Docker Desktop's kernel creates DOWN tunnel
devices (`tunl0`, `gre0`, `sit0`, ...) in every namespace; the check is "only `lo` is UP".

### `--mode audit-only` (anywhere, smoke test)
Runs the command on the host network with only the audit hook. **Not an isolation test** — use
it to find which code path tries to connect while developing.

### The Python audit hook (all modes)
`deploy/egress/audit/sitecustomize.py` is put on `PYTHONPATH`; it registers a PEP 578
`sys.addaudithook` that records non-loopback `socket.connect/sendto/sendmsg/getaddrinfo/
gethostbyname*` events to `$ZLLF_EGRESS_AUDIT_LOG`. It never blocks (the OS layer does that); it
exists so attempts that fail *silently* still fail the test.

## Harness self-test

`deploy/egress/self_test.sh --mode <m>` runs stand-in commands, no pipeline needed:

| Case | Command | Expected |
|---|---|---|
| clean | local TCP echo on 127.0.0.1 | PASS |
| connect-caught | TCP connect to 192.0.2.1:443, error swallowed, exit 0 | caught (`--expect-egress`) |
| dns-caught | `getaddrinfo("huggingface.co")`, error swallowed, exit 0 | caught |
| connect-fails | same offender as a normal test | FAIL (rc 1) |
| failing | `sys.exit(1)` | FAIL (rc 1) |

Verified 2026-11-05/06: linux mode (as root, via sudo + setpriv, and rootless) inside a
privileged Debian container; docker mode on the M4 Pro with Docker Desktop; and the real
`zllf run ... --hardware fake` pipeline PASSed in linux mode (netns inside a container), docker
mode (`--network none`) and audit-only mode.

## Appliance hosts

`deploy/egress/nftables-appliance.nft` is the host policy: allow loopback, established replies
and the LAN, **log + count + drop** everything else leaving the host or forwarded from Docker
bridges. Apply with `sudo nft -f deploy/egress/nftables-appliance.nft` (or
`install.sh --apply-firewall`), check with
`sudo nft list counter inet zllf_appliance egress_dropped`. Narrow `lan4` to the real subnet.
Release checklist: run the linux variant on the GPU machine with the `ref-500` corpus and
archive `egress-out/`.
