from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from zllf.stages.package.signing import (
    PRIVATE_KEY_NAME,
    PUBLIC_KEY_NAME,
    SIG_NAME,
    SUMS_NAME,
    default_key_dir,
    format_verify_result,
    load_or_create_key,
    parse_sums,
    sign_package,
    verify_package,
    verify_package_cli,
)


def _package(root: Path) -> Path:
    pkg = root / "pkg"
    (pkg / "adapter").mkdir(parents=True)
    (pkg / "adapter" / "w.safetensors").write_bytes(b"\x00\x01" * 100)
    (pkg / "report.json").write_text('{"a": 1}\n')
    (pkg / "report.pdf").write_bytes(b"%PDF-1.7 fake")
    return pkg


def test_key_created_once_with_0600(tmp_path: Path) -> None:
    kd = tmp_path / "keys"
    k1 = load_or_create_key(kd)
    mode = stat.S_IMODE(os.stat(kd / PRIVATE_KEY_NAME).st_mode)
    assert mode == 0o600
    assert (kd / PUBLIC_KEY_NAME).exists()
    k2 = load_or_create_key(kd)
    assert k1.public_key().public_bytes_raw() == k2.public_key().public_bytes_raw()


def test_default_key_dir_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("ZLLF_KEY_DIR", raising=False)
    monkeypatch.setenv("ZLLF_DATA_DIR", str(tmp_path))
    assert default_key_dir() == tmp_path / "keys"
    monkeypatch.delenv("ZLLF_DATA_DIR")
    assert default_key_dir() == Path("~/.local/share/zllf/keys").expanduser()
    monkeypatch.setenv("ZLLF_KEY_DIR", str(tmp_path / "x"))
    assert default_key_dir() == tmp_path / "x"


def test_sign_then_verify_ok(tmp_path: Path) -> None:
    kd = tmp_path / "keys"
    pkg = _package(tmp_path)
    sign_package(pkg, load_or_create_key(kd))
    lines = (pkg / SUMS_NAME).read_text().splitlines()
    assert [ln.split("  ", 1)[1] for ln in lines] == sorted(
        ["adapter/w.safetensors", "report.json", "report.pdf", PUBLIC_KEY_NAME])
    result = verify_package(pkg, key_dir=kd)
    assert result.ok and result.trusted and result.key_source == "appliance"
    assert result.files_checked == 4
    # explicit key + bundled-key fallback
    assert verify_package(pkg, kd / PUBLIC_KEY_NAME).key_source == "explicit"
    bundled = verify_package(pkg, key_dir=tmp_path / "nokeys")
    assert bundled.ok and not bundled.trusted and "integrity only" in format_verify_result(bundled)


def test_tamper_one_byte_names_the_file(tmp_path: Path) -> None:
    kd = tmp_path / "keys"
    pkg = _package(tmp_path)
    sign_package(pkg, load_or_create_key(kd))
    target = pkg / "adapter" / "w.safetensors"
    data = bytearray(target.read_bytes())
    data[10] ^= 0x01
    target.write_bytes(bytes(data))
    result = verify_package(pkg, key_dir=kd)
    assert not result.ok
    assert result.signature_valid  # SUMS untouched, content changed
    assert result.mismatched == ["adapter/w.safetensors"]
    assert "MODIFIED: adapter/w.safetensors" in format_verify_result(result)


def test_tampered_sums_breaks_signature(tmp_path: Path) -> None:
    kd = tmp_path / "keys"
    pkg = _package(tmp_path)
    sign_package(pkg, load_or_create_key(kd))
    (pkg / "report.json").write_text('{"a": 2}\n')
    sums = pkg / SUMS_NAME
    from zllf.core.hashing import sha256_file

    text = sums.read_text().splitlines()
    text = [f"{sha256_file(pkg / 'report.json')}  report.json" if ln.endswith(" report.json")
            else ln for ln in text]
    sums.write_text("\n".join(text) + "\n")
    result = verify_package(pkg, key_dir=kd)
    assert not result.signature_valid and not result.ok
    assert not result.mismatched


def test_wrong_key_missing_and_extra_files(tmp_path: Path) -> None:
    pkg = _package(tmp_path)
    sign_package(pkg, load_or_create_key(tmp_path / "k1"))
    other = load_or_create_key(tmp_path / "k2")
    assert not verify_package(pkg, other.public_key()).signature_valid
    (pkg / "report.pdf").unlink()
    (pkg / "extra.txt").write_text("x")
    result = verify_package(pkg, key_dir=tmp_path / "k1")
    assert result.missing == ["report.pdf"] and result.unlisted == ["extra.txt"]


def test_unsigned_dir_and_cli_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = _package(tmp_path)
    monkeypatch.setenv("ZLLF_KEY_DIR", str(tmp_path / "keys"))
    assert verify_package_cli(pkg) == 1
    assert f"missing {SUMS_NAME}" in capsys.readouterr().out
    sign_package(pkg, load_or_create_key())
    assert verify_package_cli(pkg) == 0
    assert "OK" in capsys.readouterr().out
    assert (pkg / SIG_NAME).read_text().strip()


def test_parse_sums_rejects_traversal() -> None:
    entries, bad = parse_sums(f"{'a' * 64}  ../etc/passwd\n{'b' * 64}  ok.txt\nnonsense\n")
    assert entries == {"ok.txt": "b" * 64}
    assert len(bad) == 2
