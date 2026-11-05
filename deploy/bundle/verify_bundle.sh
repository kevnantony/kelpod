#!/usr/bin/env bash
# Verify an offline bundle BEFORE anything is loaded (P3.5.2, NFR-SEC-02). Refuses on any
# mismatch. Runs on the air-gapped side: no network and no docker needed.
#
# usage: deploy/bundle/verify_bundle.sh --dir DIR --pubkey PEM [--extract-to DIR] [--allow-unsigned]
#   1. Ed25519 signature of <name>.SHA256SUMS (parts can be checked before reassembly: diodes)
#   2. sha256 of every part and part count, then of the reassembled tar
#   3. safe extract (no absolute paths, '..', links or devices), signature of bundle-manifest.json
#   4. every manifest file present with the right size + sha256, and NO extra files
# Exit 0 = verified (prints the extracted bundle path on stdout); non-zero = refuse.
# The public key must come out-of-band (not from the bundle itself).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=SCRIPTDIR/lib.sh
source "$here/lib.sh"
py="${ZLLF_PYTHON:-python3}"

dir=""; pubkey=""; extract=""; allow_unsigned=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir) dir="$2"; shift ;;
    --pubkey) pubkey="$2"; shift ;;
    --extract-to) extract="$2"; shift ;;
    --allow-unsigned) allow_unsigned=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done
[[ -d "$dir" ]] || die "--dir DIR required"
shopt -s nullglob
sums_files=("$dir"/zllf-bundle-*.SHA256SUMS)
shopt -u nullglob
[[ ${#sums_files[@]} == 1 ]] || die "expected exactly one zllf-bundle-*.SHA256SUMS in $dir"
sums="${sums_files[0]}"
name="$(basename "$sums" .SHA256SUMS)"

check_sig() {  # file
  if [[ -f "$1.sig" && -n "$pubkey" ]]; then
    ed25519_verify "$pubkey" "$1" "$1.sig" || die "BAD SIGNATURE: $(basename "$1")"
    log "signature OK: $(basename "$1")"
  elif [[ $allow_unsigned == 1 ]]; then
    log "WARNING: $(basename "$1") not signature-checked (--allow-unsigned)"
  else
    die "$(basename "$1"): missing .sig or --pubkey (--allow-unsigned is for dev bundles only)"
  fi
}

# 1+2) parts
check_sig "$sums"
tar_sha=""
while read -r sha f; do
  if [[ "$f" == "$name.tar" ]]; then tar_sha="$sha"; continue; fi
  [[ "$f" == "$name.tar.part-"* && "$f" != */* ]] || die "unexpected entry in SHA256SUMS: $f"
  [[ -f "$dir/$f" ]] || die "missing part: $f"
  [[ "$(sha256_of "$dir/$f")" == "$sha" ]] || die "HASH MISMATCH: $f"
done < "$sums"
[[ -n "$tar_sha" ]] || die "SHA256SUMS lacks the tar digest"
n_listed=$(grep -c "  $name.tar.part-" "$sums")
n_present=$(find "$dir" -maxdepth 1 -name "$name.tar.part-*" | wc -l | tr -d ' ')
[[ "$n_listed" == "$n_present" ]] || die "part count mismatch: listed $n_listed, present $n_present"
log "parts OK ($n_listed)"

extract="${extract:-$(mktemp -d "${TMPDIR:-/tmp}/zllf-verify.XXXXXX")}"
mkdir -p "$extract"
[[ ! -e "$extract/bundle" ]] || die "$extract/bundle already exists; use an empty --extract-to"
tarball="$extract/$name.tar"
cat "$dir/$name".tar.part-* > "$tarball"
[[ "$(sha256_of "$tarball")" == "$tar_sha" ]] || die "HASH MISMATCH: reassembled tar"

# 3) safe extract
"$py" - "$tarball" "$extract" <<'PY'
import os, sys, tarfile
tar_path, dest = sys.argv[1], os.path.realpath(sys.argv[2])
with tarfile.open(tar_path) as tar:
    members = tar.getmembers()
    for m in members:
        target = os.path.realpath(os.path.join(dest, m.name))
        if not target.startswith(dest + os.sep) or not (m.isfile() or m.isdir()):
            sys.exit(f"refusing unsafe tar member: {m.name}")
    try:
        tar.extractall(dest, members=members, filter="data")
    except TypeError:  # Python without PEP 706 tar filters; members were checked above
        tar.extractall(dest, members=members)
PY
rm -f "${tarball:?}"
b="$extract/bundle"
check_sig "$b/bundle-manifest.json"

# 4) every file, no extras
"$py" - "$b" <<'PY'
import hashlib, json, os, sys
root = sys.argv[1]
man = json.load(open(os.path.join(root, "bundle-manifest.json")))
listed = {f["path"]: f for f in man["files"]}
present = set()
for dirpath, _, filenames in os.walk(root):
    for fn in filenames:
        rel = os.path.relpath(os.path.join(dirpath, fn), root)
        if rel not in ("bundle-manifest.json", "bundle-manifest.json.sig"):
            present.add(rel)
bad = [f"extra file: {p}" for p in sorted(present - set(listed))]
bad += [f"missing: {p}" for p in sorted(set(listed) - present)]
for rel in sorted(present & set(listed)):
    f, p = listed[rel], os.path.join(root, rel)
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    if os.path.getsize(p) != f["size"] or h.hexdigest() != f["sha256"]:
        bad.append(f"HASH MISMATCH: {rel}")
for line in bad[:50]:
    print(f"[verify_bundle] {line}", file=sys.stderr)
if bad:
    sys.exit(1)
print(f"[verify_bundle] {len(listed)} files OK ({man['name']} {man['version']})", file=sys.stderr)
PY
echo "$b"
