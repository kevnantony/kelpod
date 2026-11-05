"""Stage ⑥ PACKAGE: adapter export, audit report (JSON + PDF), AI-SBOM, signed checksums."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import ClassVar

from zllf.core.errors import ErrorCode, ZLLFError
from zllf.stages.base import RunContext, Stage, StageResult
from zllf.stages.package import artefacts as A
from zllf.stages.package import report as R
from zllf.stages.package.pdf import render_scorecard
from zllf.stages.package.sbom import build_sbom, write_sbom
from zllf.stages.package.signing import key_id, load_or_create_key, sign_package


class PackageStage(Stage):
    """Writes ``adapter/``, ``report.json``, ``report.pdf``, ``sbom.cdx.json``,
    ``appliance_ed25519.pub``, ``SHA256SUMS`` and ``SHA256SUMS.sig`` into ``out_dir``.

    The package is produced whether or not the gates passed; ``report.json["ready"]``
    and the PDF banner carry the verdict and deployment is gated elsewhere.
    """

    name: ClassVar[str] = "package"

    def __init__(self, key_dir: Path | None = None, models_manifest: Path | None = None,
                 bench_dir: Path | None = None) -> None:
        self.key_dir = key_dir
        self.models_manifest = models_manifest
        self.bench_dir = bench_dir

    def run(self, ctx: RunContext, out_dir: Path) -> StageResult:
        paths = ctx.paths
        A.require_inputs(paths)
        ctx.progress(self.name, 0, 5, "exporting adapter")
        shutil.copytree(A.adapter_dir(paths), out_dir / "adapter")

        try:
            key = load_or_create_key(self.key_dir)
        except (OSError, ValueError) as exc:
            raise ZLLFError(ErrorCode.PACKAGE_FAILED, f"signing key unavailable: {exc}") from exc
        kid = key_id(key.public_key())

        ctx.check_cancelled()
        ctx.progress(self.name, 1, 5, "writing audit report")
        report = R.build_report(paths, ctx.config, run_id=ctx.run_id, signing_key_id=kid,
                                models_manifest=self.models_manifest, bench_dir=self.bench_dir)
        R.write_report(report, out_dir / "report.json")

        ctx.progress(self.name, 2, 5, "rendering scorecard")
        n_pages = render_scorecard(report, A.load_examples(paths), out_dir / "report.pdf",
                                   secrets=A.load_canary_secrets(paths))

        ctx.check_cancelled()
        ctx.progress(self.name, 3, 5, "writing AI-SBOM")
        bom = build_sbom(paths, ctx.config, report,
                         models_manifest=R.load_models_manifest(self.models_manifest),
                         bench=R.bench_datasets(self.bench_dir))
        sbom_stats = write_sbom(bom, out_dir / "sbom.cdx.json")

        ctx.progress(self.name, 4, 5, "signing package")
        sign_package(out_dir, key)
        ctx.progress(self.name, 5, 5, "done")
        return StageResult(metrics={
            "ready": report["ready"],
            "adapter_sha256": report["adapter"]["sha256"],
            "adapter_size_bytes": report["adapter"]["size_bytes"],
            "pdf_pages": n_pages,
            "sbom_components": sum(sbom_stats["components_by_type"].values()),
            "sbom_validated": sbom_stats["validated"],
            "signing_key_id": kid,
        })
