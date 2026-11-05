#!/usr/bin/env bash
# Build the offline install bundle on the ONLINE build host (DEVELOPMENT_PLAN P3.5.1,
# NFR-SEC-02). See docs/runbooks/install.md.
#
# usage: deploy/bundle/build_bundle.sh [options]
#   --out DIR          output dir (default: dist/bundle)
#   --version V        bundle version (default: from pyproject.toml)
#   --image IMG        image to `docker save` (repeatable; default: zllf-app:dev)
#   --hardware "A B"   only models used by these hardware profiles (default: whole manifest)
#   --platform P       wheelhouse platform tag (default: manylinux_2_28_x86_64)
#   --part-size SIZE   split size for media / data diodes (default: 4000m; `split -b` syntax)
#   --sign-key PEM     Ed25519 private key; signs bundle-manifest.json and the SHA256SUMS
#   --skip-images | --skip-wheels | --skip-models   (quick/dev bundles)
#
# Output (in --out):
#   zllf-bundle-<v>.tar.part-aaa ...    the payload split into fixed-size parts
#   zllf-bundle-<v>.SHA256SUMS[.sig]    sha256 of every part and of the reassembled tar
#   bundle-manifest.json[.sig]          copy of the in-tar manifest (every file + sha256 + licence)
# Inside the tar (bundle/): images/*.tar.gz  wheelhouse/*.whl  models/hub/...
#   models/manifest.json  deploy/{compose,egress,bundle}  signing-key.pub
#   bundle-manifest.json[.sig]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
# shellcheck source=SCRIPTDIR/lib.sh
source "$here/lib.sh"
py="${ZLLF_PYTHON:-python3}"

out="$repo/dist/bundle"; version=""; images=(); hardware=""; platform="manylinux_2_28_x86_64"
part_size="4000m"; sign_key=""; skip_images=0; skip_wheels=0; skip_models=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) out="$2"; shift ;;
    --version) version="$2"; shift ;;
    --image) images+=("$2"); shift ;;
    --hardware) hardware="$2"; shift ;;
    --platform) platform="$2"; shift ;;
    --part-size) part_size="$2"; shift ;;
    --sign-key) sign_key="$2"; shift ;;
    --skip-images) skip_images=1 ;;
    --skip-wheels) skip_wheels=1 ;;
    --skip-models) skip_models=1 ;;
    -h|--help) sed -n '2,23p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done
[[ ${#images[@]} -gt 0 ]] || images=(zllf-app:dev)
version="${version:-$(sed -nE 's/^version = "([^"]+)"/\1/p' "$repo/pyproject.toml" | head -1)}"
if [[ -n "$sign_key" && ! -f "$sign_key" ]]; then die "sign key not found: $sign_key"; fi
name="zllf-bundle-${version:?}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"
stage="$(mktemp -d "${TMPDIR:-/tmp}/zllf-bundle.XXXXXX")"
trap 'rm -rf "${stage:?}"' EXIT
b="$stage/bundle"
mkdir -p "$b"

# 1) container images
if [[ $skip_images == 0 ]]; then
  mkdir -p "$b/images"
  for img in "${images[@]}"; do
    docker image inspect "$img" >/dev/null || die "image $img not built"
    f="$b/images/$(echo "$img" | tr '/:' '__').tar.gz"
    log "docker save $img"
    docker save "$img" | gzip -1 > "$f"
  done
fi

# 2) wheelhouse (host-side tooling / upgrades; the image already contains its own venv)
if [[ $skip_wheels == 0 ]]; then
  mkdir -p "$b/wheelhouse"
  req="$stage/requirements.txt"
  # -o, not a pipe: with FORCE_COLOR set, uv colours stdout and corrupts the first line.
  uv export --project "$repo" --frozen --no-dev --no-emit-project --no-hashes \
     --format requirements.txt -o "$req.all" >/dev/null
  grep -viE '^(nvidia-|triton==|cuda-|mlx)' "$req.all" > "$req"
  rm -f "$req.all"
  log "wheelhouse for $platform ($(grep -cE '^[a-zA-Z0-9]' "$req") packages)"
  uvx pip download --quiet -r "$req" -d "$b/wheelhouse" --only-binary=:all: \
     --platform "$platform" --python-version 3.12 --implementation cp \
     --extra-index-url https://download.pytorch.org/whl/cpu
  uv build --project "$repo" --wheel --out-dir "$b/wheelhouse" >/dev/null
fi

# 3) models: pinned snapshots from the HF cache, symlinks dereferenced, re-verified first
if [[ $skip_models == 0 ]]; then
  manifest="$repo/models/manifest.json"
  [[ -f "$manifest" ]] || die "no models/manifest.json; run scripts/download_models.py first"
  # shellcheck disable=SC2086  # $hardware is a space-separated list on purpose
  "$py" "$repo/scripts/download_models.py" --verify ${hardware:+--hardware $hardware} \
    || die "model verification failed"
  mkdir -p "$b/models/hub"
  cp "$manifest" "$b/models/manifest.json"
  cache="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
  # Copy exactly the files recorded in the manifest (the cache snapshot may hold extras).
  "$py" - "$manifest" "$hardware" "$cache" "$b/models/hub" <<'PY'
import json, os, shutil, sys
manifest, hardware, cache, dest = sys.argv[1:5]
wanted = set(hardware.split()) if hardware else None
for m in json.load(open(manifest))["models"]:
    if wanted is not None and not wanted & set(m.get("profiles", [])):
        continue
    d = "models--" + m["name"].replace("/", "--")
    print(f"[build_bundle] model {m['name']}@{m['revision'][:12]} ({len(m['files'])} files)",
          file=sys.stderr)
    for f in m["files"]:
        src = os.path.join(cache, d, "snapshots", m["revision"], f["path"])
        dst = os.path.join(dest, d, "snapshots", m["revision"], f["path"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)  # follows the blob symlink
    os.makedirs(os.path.join(dest, d, "refs"), exist_ok=True)
    with open(os.path.join(dest, d, "refs", "main"), "w") as fh:
        fh.write(m["revision"])
PY
fi

# 4) deployment files (+ public key, for reference only: trust comes from the out-of-band copy)
mkdir -p "$b/deploy"
cp -R "$repo/deploy/compose" "$repo/deploy/egress" "$repo/deploy/bundle" "$b/deploy/"
if [[ -n "$sign_key" ]]; then ed25519_pubkey "$sign_key" "$b/signing-key.pub"; fi

# 5) bundle-manifest.json: every file + size + sha256 (+ model licences)
log "hashing files"
"$py" - "$b" "$version" <<'PY'
import datetime, hashlib, json, os, sys
root, version = sys.argv[1], sys.argv[2]
skip = {"bundle-manifest.json", "bundle-manifest.json.sig"}
files = []
for dirpath, dirnames, filenames in os.walk(root):
    dirnames.sort()
    for fn in sorted(filenames):
        p = os.path.join(dirpath, fn)
        rel = os.path.relpath(p, root)
        if rel in skip:
            continue
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        files.append({"path": rel, "size": os.path.getsize(p), "sha256": h.hexdigest()})
models = []
mp = os.path.join(root, "models", "manifest.json")
if os.path.exists(mp):
    models = [{"name": m["name"], "revision": m["revision"], "license": m.get("license")}
              for m in json.load(open(mp))["models"]]
manifest = {
    "schema_version": 1, "name": "zllf-bundle", "version": version,
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "models": models, "file_count": len(files), "total_size": sum(f["size"] for f in files),
    "files": files,
}
with open(os.path.join(root, "bundle-manifest.json"), "w") as fh:
    json.dump(manifest, fh, indent=1)
print(f"[build_bundle] {len(files)} files, {manifest['total_size'] / 2**20:.1f} MiB", file=sys.stderr)
PY
if [[ -n "$sign_key" ]]; then
  ed25519_sign "$sign_key" "$b/bundle-manifest.json" "$b/bundle-manifest.json.sig"
  log "signed bundle-manifest.json"
else
  log "WARNING: unsigned bundle (no --sign-key); verify_bundle.sh will require --allow-unsigned"
fi

# 6) deterministic tar (sorted, mtime 0, uid/gid 0) -> fixed-size parts + SHA256SUMS
log "tar + split ($part_size parts)"
tarball="$stage/$name.tar"
"$py" - "$stage" "bundle" "$tarball" <<'PY'
import os, sys, tarfile
stage, top, dest = sys.argv[1], sys.argv[2], sys.argv[3]
def norm(ti):
    ti.mtime = 0
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = "root"
    ti.mode = 0o755 if ti.isdir() or ti.mode & 0o111 else 0o644
    return ti
with tarfile.open(dest, "w", format=tarfile.PAX_FORMAT) as tar:
    tar.add(os.path.join(stage, top), arcname=top, recursive=False, filter=norm)
    for dirpath, dirnames, filenames in os.walk(os.path.join(stage, top)):
        dirnames.sort()
        for n in sorted(dirnames) + sorted(filenames):
            p = os.path.join(dirpath, n)
            tar.add(p, arcname=os.path.relpath(p, stage), recursive=False, filter=norm)
PY
find "$out" -maxdepth 1 -name "$name.tar.part-*" -delete
split -b "$part_size" -a 3 "$tarball" "$out/$name.tar.part-"
{
  for p in "$out/$name".tar.part-*; do echo "$(sha256_of "$p")  $(basename "$p")"; done
  echo "$(sha256_of "$tarball")  $name.tar"
} > "$out/$name.SHA256SUMS"
if [[ -n "$sign_key" ]]; then
  ed25519_sign "$sign_key" "$out/$name.SHA256SUMS" "$out/$name.SHA256SUMS.sig"
fi
cp "$b/bundle-manifest.json" "$out/"
if [[ -f "$b/bundle-manifest.json.sig" ]]; then cp "$b/bundle-manifest.json.sig" "$out/"; fi
n_parts=$(find "$out" -maxdepth 1 -name "$name.tar.part-*" | wc -l | tr -d ' ')
log "done: $out/$name.tar.part-* ($n_parts parts)"
