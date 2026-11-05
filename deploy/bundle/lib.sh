#!/usr/bin/env bash
# Shared helpers for build_bundle.sh / verify_bundle.sh / install.sh (sourced, not executed).
#
# Ed25519 detached signatures over raw file bytes, PEM keys (PKCS#8 private / SPKI public):
#   1. OpenSSL >= 3 (`openssl pkeyutl -rawin`) when available (Ubuntu 22.04+, RHEL 9), else
#   2. Python + `cryptography` ($ZLLF_PYTHON, default python3; the project venv has it).
# Both produce/accept the same 64-byte raw signature, so either side can verify the other.
# (macOS ships LibreSSL without Ed25519, so the Python path is used there.)

sha256_of() {  # file -> hex digest
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}

_openssl_ed25519_ok() {
  command -v openssl >/dev/null 2>&1 || return 1
  openssl version 2>/dev/null | grep -qE '^OpenSSL [3-9]' || return 1
}

_py() { "${ZLLF_PYTHON:-python3}" "$@"; }

ed25519_keygen() {  # private.pem public.pem
  if _openssl_ed25519_ok; then
    openssl genpkey -algorithm ed25519 -out "$1" && openssl pkey -in "$1" -pubout -out "$2"
  else
    _py - "$1" "$2" <<'PY'
import sys
from cryptography.hazmat.primitives import serialization as s
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
k = Ed25519PrivateKey.generate()
priv = k.private_bytes(s.Encoding.PEM, s.PrivateFormat.PKCS8, s.NoEncryption())
pub = k.public_key().public_bytes(s.Encoding.PEM, s.PublicFormat.SubjectPublicKeyInfo)
open(sys.argv[1], "wb").write(priv)
open(sys.argv[2], "wb").write(pub)
PY
  fi
  chmod 600 "$1"
}

ed25519_pubkey() {  # private.pem public.pem
  if _openssl_ed25519_ok; then
    openssl pkey -in "$1" -pubout -out "$2"
  else
    _py - "$1" "$2" <<'PY'
import sys
from cryptography.hazmat.primitives import serialization as s
k = s.load_pem_private_key(open(sys.argv[1], "rb").read(), password=None)
pub = k.public_key().public_bytes(s.Encoding.PEM, s.PublicFormat.SubjectPublicKeyInfo)
open(sys.argv[2], "wb").write(pub)
PY
  fi
}

ed25519_sign() {  # private.pem file sigfile
  if _openssl_ed25519_ok; then
    openssl pkeyutl -sign -inkey "$1" -rawin -in "$2" -out "$3"
  else
    _py - "$1" "$2" "$3" <<'PY'
import sys
from cryptography.hazmat.primitives.serialization import load_pem_private_key
key = load_pem_private_key(open(sys.argv[1], "rb").read(), password=None)
open(sys.argv[3], "wb").write(key.sign(open(sys.argv[2], "rb").read()))
PY
  fi
}

ed25519_verify() {  # public.pem file sigfile -> exit 0 if valid
  if _openssl_ed25519_ok; then
    openssl pkeyutl -verify -pubin -inkey "$1" -rawin -in "$2" -sigfile "$3" >/dev/null 2>&1
  else
    _py - "$1" "$2" "$3" <<'PY'
import sys
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.serialization import load_pem_public_key
key = load_pem_public_key(open(sys.argv[1], "rb").read())
try:
    key.verify(open(sys.argv[3], "rb").read(), open(sys.argv[2], "rb").read())
except InvalidSignature:
    sys.exit(1)
PY
  fi
}

die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[$(basename "$0" .sh)] $*" >&2; }
