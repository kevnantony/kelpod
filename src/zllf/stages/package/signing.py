"""Appliance Ed25519 key, ``SHA256SUMS`` manifests and detached signatures (NFR-SEC-02).

Layout of a signed package directory::

    SHA256SUMS              "<sha256>  <relative/path>" per file, sorted, LF-terminated
    SHA256SUMS.sig          base64 Ed25519 signature over the exact bytes of SHA256SUMS
    appliance_ed25519.pub   PEM public key of the signing appliance (convenience copy)

Verification needs no network and no handshake (diode-friendly). The bundled public
key only proves integrity, not origin: pass the appliance's trusted key to
:func:`verify_package` (default: the local appliance key, when present).
"""

from __future__ import annotations

import base64
import binascii
import os
from dataclasses import dataclass, field
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from zllf.core.hashing import sha256_bytes, sha256_file

SUMS_NAME = "SHA256SUMS"
SIG_NAME = "SHA256SUMS.sig"
PRIVATE_KEY_NAME = "appliance_ed25519"
PUBLIC_KEY_NAME = "appliance_ed25519.pub"
_UNSIGNED = frozenset({SUMS_NAME, SIG_NAME})


# ---------------------------------------------------------------- key management
def default_key_dir() -> Path:
    """``$ZLLF_KEY_DIR`` or ``$ZLLF_DATA_DIR/keys`` (default ``~/.local/share/zllf/keys``)."""
    if override := os.environ.get("ZLLF_KEY_DIR"):
        return Path(override).expanduser()
    data_dir = os.environ.get("ZLLF_DATA_DIR") or "~/.local/share/zllf"
    return Path(data_dir).expanduser() / "keys"


def _public_pem(key: Ed25519PublicKey) -> bytes:
    return key.public_bytes(serialization.Encoding.PEM,
                            serialization.PublicFormat.SubjectPublicKeyInfo)


def key_id(key: Ed25519PublicKey) -> str:
    """Stable fingerprint: first 16 hex chars of sha256 over the raw 32-byte public key."""
    raw = key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return sha256_bytes(raw)[:16]


def load_or_create_key(key_dir: Path | None = None) -> Ed25519PrivateKey:
    """Load the appliance signing key, generating it (mode 0600) on first use."""
    key_dir = key_dir or default_key_dir()
    priv_path = key_dir / PRIVATE_KEY_NAME
    if not priv_path.exists():
        key_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        key = Ed25519PrivateKey.generate()
        pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption())
        try:
            fd = os.open(priv_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:  # created concurrently: use theirs
            return load_or_create_key(key_dir)
        with os.fdopen(fd, "wb") as f:
            f.write(pem)
        (key_dir / PUBLIC_KEY_NAME).write_bytes(_public_pem(key.public_key()))
        return key
    loaded = serialization.load_pem_private_key(priv_path.read_bytes(), password=None)
    if not isinstance(loaded, Ed25519PrivateKey):
        raise ValueError(f"{priv_path} is not an Ed25519 private key")
    pub_path = key_dir / PUBLIC_KEY_NAME
    if not pub_path.exists():
        pub_path.write_bytes(_public_pem(loaded.public_key()))
    return loaded


def load_public_key(source: Path | bytes | Ed25519PublicKey) -> Ed25519PublicKey:
    """Accept a PEM file/bytes, raw 32 bytes, or a key object."""
    if isinstance(source, Ed25519PublicKey):
        return source
    data = source.read_bytes() if isinstance(source, Path) else source
    if len(data) == 32:
        return Ed25519PublicKey.from_public_bytes(data)
    key = serialization.load_pem_public_key(data)
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("not an Ed25519 public key")
    return key


# ---------------------------------------------------------------- sums + signature
def package_files(package_dir: Path) -> list[str]:
    """Relative POSIX paths of every file covered by SHA256SUMS (sorted)."""
    return sorted(
        p.relative_to(package_dir).as_posix()
        for p in package_dir.rglob("*")
        if p.is_file() and p.relative_to(package_dir).as_posix() not in _UNSIGNED
    )


def build_sums(package_dir: Path) -> str:
    lines = [f"{sha256_file(package_dir / rel)}  {rel}" for rel in package_files(package_dir)]
    return "".join(line + "\n" for line in lines)


def sign_package(package_dir: Path, key: Ed25519PrivateKey) -> str:
    """Copy the public key in, write SHA256SUMS + SHA256SUMS.sig. Returns the key id."""
    (package_dir / PUBLIC_KEY_NAME).write_bytes(_public_pem(key.public_key()))
    sums = build_sums(package_dir).encode("utf-8")
    (package_dir / SUMS_NAME).write_bytes(sums)
    signature = key.sign(sums)
    (package_dir / SIG_NAME).write_text(base64.b64encode(signature).decode("ascii") + "\n",
                                        encoding="ascii")
    return key_id(key.public_key())


def parse_sums(text: str) -> tuple[dict[str, str], list[str]]:
    """Parse SHA256SUMS -> ({path: hash}, [malformed/unsafe lines])."""
    entries: dict[str, str] = {}
    bad: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        digest, sep, rel = line.partition("  ")
        rel = rel.removeprefix("*")
        unsafe = rel.startswith("/") or ".." in Path(rel).parts or not rel
        if not sep or len(digest) != 64 or unsafe:
            bad.append(line)
            continue
        try:
            int(digest, 16)
        except ValueError:
            bad.append(line)
            continue
        entries[rel] = digest.lower()
    return entries, bad


def _read_signature(path: Path) -> bytes:
    data = path.read_bytes()
    if len(data) == 64:
        return data
    try:
        return base64.b64decode(data.strip(), validate=True)
    except binascii.Error:
        return data


# ---------------------------------------------------------------- verification
@dataclass
class VerifyResult:
    """Outcome of :func:`verify_package`. ``ok`` only if signature and every hash match."""

    package_dir: str
    signature_valid: bool = False
    key_id: str | None = None
    key_source: str = "none"  # "explicit" | "appliance" | "bundled" | "none"
    files_checked: int = 0
    mismatched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unlisted: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (self.signature_valid and not self.mismatched and not self.missing
                and not self.unlisted and not self.errors)

    @property
    def trusted(self) -> bool:
        """True when verified against a key not taken from the package itself."""
        return self.ok and self.key_source in ("explicit", "appliance")


def verify_package(
    package_dir: Path,
    pubkey: Path | bytes | Ed25519PublicKey | None = None,
    *,
    key_dir: Path | None = None,
) -> VerifyResult:
    """Check SHA256SUMS.sig and every listed hash; report mismatching/missing/extra files.

    Key choice: ``pubkey`` if given; else the local appliance key (``key_dir`` or
    :func:`default_key_dir`) if it exists; else the public key bundled in the package
    (integrity only — ``result.trusted`` stays False).
    """
    package_dir = Path(package_dir)
    result = VerifyResult(package_dir=str(package_dir))
    sums_path, sig_path = package_dir / SUMS_NAME, package_dir / SIG_NAME
    for path in (sums_path, sig_path):
        if not path.exists():
            result.errors.append(f"missing {path.name}")
    if result.errors:
        return result

    try:
        if pubkey is not None:
            key, result.key_source = load_public_key(pubkey), "explicit"
        elif (appliance := (key_dir or default_key_dir()) / PUBLIC_KEY_NAME).exists():
            key, result.key_source = load_public_key(appliance), "appliance"
        elif (bundled := package_dir / PUBLIC_KEY_NAME).exists():
            key, result.key_source = load_public_key(bundled), "bundled"
        else:
            result.errors.append("no public key available")
            return result
    except ValueError as exc:
        result.errors.append(f"bad public key: {exc}")
        return result
    result.key_id = key_id(key)

    sums = sums_path.read_bytes()
    try:
        key.verify(_read_signature(sig_path), sums)
        result.signature_valid = True
    except InvalidSignature:
        result.errors.append("signature does not match SHA256SUMS")

    entries, bad = parse_sums(sums.decode("utf-8", errors="replace"))
    result.errors.extend(f"malformed line in SHA256SUMS: {line[:80]!r}" for line in bad)
    for rel, digest in sorted(entries.items()):
        path = package_dir / rel
        if not path.is_file():
            result.missing.append(rel)
            continue
        result.files_checked += 1
        if sha256_file(path) != digest:
            result.mismatched.append(rel)
    result.unlisted = [rel for rel in package_files(package_dir) if rel not in entries]
    return result


def format_verify_result(result: VerifyResult) -> str:
    """Human-readable multi-line summary for the CLI."""
    status = "OK" if result.ok else "FAILED"
    lines = [
        f"Package {result.package_dir}: {status}",
        f"  signature: {'valid' if result.signature_valid else 'INVALID'}"
        f" (key {result.key_id or '-'}, source {result.key_source})",
        f"  files checked: {result.files_checked}",
    ]
    lines += [f"  MODIFIED: {p}" for p in result.mismatched]
    lines += [f"  MISSING:  {p}" for p in result.missing]
    lines += [f"  UNLISTED: {p}" for p in result.unlisted]
    lines += [f"  ERROR:    {e}" for e in result.errors]
    if result.ok and not result.trusted:
        lines.append("  note: verified with the key bundled in the package (integrity only); "
                     "pass --pubkey to check the origin.")
    return "\n".join(lines)


def verify_package_cli(package_dir: str | Path, pubkey: str | Path | None = None) -> int:
    """Entry point for ``zllf verify-package <dir> [--pubkey FILE]``: prints, returns exit code."""
    result = verify_package(Path(package_dir), Path(pubkey) if pubkey else None)
    print(format_verify_result(result))
    return 0 if result.ok else 1
