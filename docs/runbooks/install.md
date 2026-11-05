# Runbook: build, verify and install the offline bundle

Status: **skeleton** (DEVELOPMENT_PLAN P3.5). Building, signing and verifying work; `install.sh`
does the prerequisite checks and verification and has the load/start steps, but LUKS, TLS and
the admin bootstrap token are TODO. Today only the CPU image exists (`Dockerfile.app`);
`Dockerfile.cuda` is a placeholder until a Linux/NVIDIA machine is available.

Two machines: an **online build host** and the **air-gapped appliance**. Data moves one way,
build host → appliance, as files (USB/media or a data diode). No handshake is needed.

## 1. Build host (online)

```bash
uv sync
ZLLF_ALLOW_ONLINE=1 uv run --with huggingface_hub python scripts/download_models.py --hardware toy
docker build -f deploy/docker/Dockerfile.app -t zllf-app:0.0.1 .

# one-time signing key (keep the private key offline; publish the .pub out-of-band)
ZLLF_PYTHON=.venv/bin/python bash -c 'source deploy/bundle/lib.sh; ed25519_keygen zllf-sign.pem zllf-sign.pub'

ZLLF_PYTHON=.venv/bin/python deploy/bundle/build_bundle.sh \
  --image zllf-app:0.0.1 --hardware toy --sign-key zllf-sign.pem --part-size 4000m --out dist/bundle
```

Output in `dist/bundle/`:

| File | Purpose |
|---|---|
| `zllf-bundle-<v>.tar.part-aaa …` | payload, split into fixed-size parts for media/diodes |
| `zllf-bundle-<v>.SHA256SUMS` + `.sig` | sha256 of every part and of the reassembled tar, Ed25519-signed |
| `bundle-manifest.json` + `.sig` | every file in the bundle with size + sha256, model licences |

Inside the tar (`bundle/`): `images/*.tar.gz` (`docker save`), `wheelhouse/` (locked deps as
CPU wheels for `--platform`, plus the zllf wheel), `models/hub/models--*/snapshots/<pinned rev>/`
(exactly the files in `models/manifest.json`, re-verified before packing), `models/manifest.json`,
`deploy/{compose,egress,bundle}`, `signing-key.pub` (reference only).

Options: `--skip-images/--skip-wheels/--skip-models` for quick bundles; `--platform` for the
wheelhouse target (`manylinux_2_28_x86_64` default). The tar is deterministic (sorted names,
mtime 0, uid/gid 0) so identical inputs give identical parts.

**Signatures.** Ed25519, detached, over the raw file bytes, PEM keys. `deploy/bundle/lib.sh`
uses OpenSSL 3 (`openssl pkeyutl -rawin`) when present, otherwise Python `cryptography`
(macOS ships LibreSSL without Ed25519, so set `ZLLF_PYTHON=.venv/bin/python` there). Both give
the same 64-byte signature. Trained adapters are signed separately by the package stage
(`zllf.stages.package.signing`, checked with `zllf verify-package`); keep a separate release
key for bundles.

## 2. Appliance (offline)

Prerequisites: Linux x86_64, Docker Engine + compose v2, python3 (stdlib only, plus
`cryptography` *or* OpenSSL 3 for signatures), `nft`, ≥ 100 GB free under `/var/lib/docker`;
GPU tier later: NVIDIA driver + container toolkit (`--gpu`).

```bash
# 1. verify only (no changes); refuses on any bad signature, hash, missing or extra file
deploy/bundle/verify_bundle.sh --dir /media/zllf --pubkey /etc/zllf/zllf-sign.pub --extract-to /var/tmp/zllf-verify

# 2. dry run of the installer: prerequisite checks + verification
sudo bundle/deploy/bundle/install.sh --dir /media/zllf --pubkey /etc/zllf/zllf-sign.pub

# 3. install: docker load, models into the zllf_zllf-models volume, compose up, health, egress self-test
sudo ZLLF_IMAGE=zllf-app:0.0.1 bundle/deploy/bundle/install.sh --dir /media/zllf \
     --pubkey /etc/zllf/zllf-sign.pub --apply-firewall --yes
```

The public key must reach the appliance out-of-band (printed fingerprint, separate medium);
the copy inside the bundle is not a trust anchor. `--allow-unsigned` exists for dev bundles only.

## 3. Runtime layout

`deploy/compose/compose.yaml`:

- `api` (`zllf serve-api --host 0.0.0.0 --port 8000`) and `worker` (`zllf worker`) share the
  `zllf-data` volume (`/data`: SQLite DB, workspaces, runs) and mount `zllf-models` read-only at
  `/data/models` (HF cache layout). Both are only on the `internal: true` network, run as uid
  10001 with a read-only root filesystem, all capabilities dropped and `no-new-privileges`.
- `gateway` (unprivileged nginx) is the only container on a published network; it proxies
  `${ZLLF_BIND:-127.0.0.1}:${ZLLF_PORT:-8080}` → `api:8000`. Docker does not publish ports of
  containers that are only on internal networks, hence the gateway. Its own egress is dropped
  by the host firewall on appliances.
- Env: `ZLLF_HARDWARE` (default `fake` in the CPU image), `ZLLF_IMAGE`, `ZLLF_BIND`,
  `ZLLF_PORT`. Appendix A offline variables are baked into the image.

Health: `GET /api/system/health`. Egress evidence: `/var/log/zllf/egress-install-*/`.
See [egress-test.md](egress-test.md).

## Not yet done (tracked in P3.5)

- `upgrade.sh` (verify, load, DB migration with backup, rollback).
- LUKS-encrypted data volume, TLS certificate, admin bootstrap token (`zllf create-admin`).
- CUDA image and GPU compose override; model import bundles (FR-17).
- Fresh-machine install test (P3.5.6).
