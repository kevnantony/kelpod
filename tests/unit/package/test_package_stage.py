from __future__ import annotations

import json
from pathlib import Path

import pytest

from zllf.core.errors import ErrorCode, ZLLFError
from zllf.core.manifest import RunPaths, read_manifest, stage_output
from zllf.core.offline import block_egress
from zllf.stages import get_stage
from zllf.stages.package.registry import register_adapter
from zllf.stages.package.signing import verify_package
from zllf.stages.package.stage import PackageStage

EXPECTED = {"adapter", "report.json", "report.pdf", "sbom.cdx.json", "SHA256SUMS",
            "SHA256SUMS.sig", "appliance_ed25519.pub"}


def _run_stage(paths: RunPaths, ctx_factory, bench: Path) -> dict:
    stage = PackageStage(bench_dir=bench)
    with block_egress(), stage_output(paths, "package") as out:
        result = stage.run(ctx_factory(paths), out)
    return result.metrics


def test_registry_entry_point() -> None:
    assert get_stage("package").name == "package"


def test_stage_end_to_end_offline(fake_run: RunPaths, ctx_factory, key_dir: Path,
                                  empty_bench: Path, raw_secret_name: str,
                                  canary_secret: str) -> None:
    metrics = _run_stage(fake_run, ctx_factory, empty_bench)
    pkg = fake_run.stage("package")
    assert {p.name for p in pkg.iterdir()} == EXPECTED
    assert metrics["ready"] is True and metrics["pdf_pages"] >= 2
    assert metrics["sbom_validated"] in (True, False)
    assert (key_dir / "appliance_ed25519").exists()

    result = verify_package(pkg, key_dir=key_dir)
    assert result.ok and result.trusted, result

    report = json.loads((pkg / "report.json").read_text())
    assert report["run_id"] == read_manifest(fake_run).run_id
    assert report["signing"]["key_id"] == metrics["signing_key_id"]
    for name in ("report.json", "sbom.cdx.json", "SHA256SUMS"):
        text = (pkg / name).read_text()
        assert raw_secret_name not in text and canary_secret not in text

    reg = register_adapter(fake_run)
    assert reg["adapter_id"].endswith(report["run_id"])
    assert reg["sha256"] == report["adapter"]["sha256"]
    assert reg["size_bytes"] == report["adapter"]["size_bytes"]
    assert reg["gates_passed"] is True and reg["signed"] is True
    assert reg["path"] == str(pkg / "adapter")


def test_stage_packages_failed_gates_as_not_ready(run_factory, ctx_factory, key_dir: Path,
                                                  empty_bench: Path) -> None:
    paths = run_factory(gates_pass=False)
    metrics = _run_stage(paths, ctx_factory, empty_bench)
    assert metrics["ready"] is False
    reg = register_adapter(paths)
    assert reg["gates_passed"] is False and reg["gates"]["G1_domain"] is False


def test_stage_fails_without_adapter(run_factory, ctx_factory, key_dir: Path,
                                     empty_bench: Path) -> None:
    paths = run_factory(omit=["train/adapter"])
    with pytest.raises(ZLLFError) as err:
        _run_stage(paths, ctx_factory, empty_bench)
    assert err.value.code is ErrorCode.PACKAGE_FAILED
    assert not paths.stage("package").exists()
    with pytest.raises(ZLLFError):
        register_adapter(paths)


def test_register_before_packaging_uses_train_adapter(fake_run: RunPaths) -> None:
    reg = register_adapter(fake_run)
    assert reg["path"].endswith("04_train/adapter")
    assert reg["signed"] is False and reg["package_dir"] is None


def test_deterministic_report_except_timestamps(fake_run: RunPaths, ctx_factory, key_dir: Path,
                                                empty_bench: Path) -> None:
    _run_stage(fake_run, ctx_factory, empty_bench)
    first = json.loads((fake_run.stage("package") / "report.json").read_text())
    _run_stage(fake_run, ctx_factory, empty_bench)
    second = json.loads((fake_run.stage("package") / "report.json").read_text())
    for r in (first, second):
        r.pop("created_at")
    assert first == second
