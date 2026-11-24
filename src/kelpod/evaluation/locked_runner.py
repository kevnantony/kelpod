"""Run one worktree's pipeline on one input, for ``scripts/locked_eval.py``.

Executed **by file path** with the worktree's own interpreter, in an empty temporary directory
that holds only the input file::

    <worktree python> -I locked_runner.py plan.pdf result.json \
        [--segment-backend NAME] [--ocr-backend NAME]

The backend options come from the worktree's declared run config (``kelpod-run.toml``, see
``kelpod.evaluation.run_config``) and are passed to ``convert`` only when declared, so a worktree
without a config runs exactly its defaults. A checkpoint reaches the pipeline only through the
``KELPOD_SEG_CHECKPOINT`` environment variable set by ``scripts/locked_eval.py``.

``-I`` keeps the runner's directory and the user site off ``sys.path``, so ``import kelpod``
resolves to the worktree under test. The runner imports nothing from ``kelpod.evaluation``;
it only calls the public ``kelpod.pipeline.convert`` and writes the graph, the timings and the
pipeline's own IFC check as JSON. The IFC file stays in the temp dir for an independent re-open.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

_OPTIONS = {"--segment-backend": "segment_backend", "--ocr-backend": "ocr_backend"}


def _parse(argv: list[str]) -> tuple[Path, Path, dict[str, str]]:
    src, out, rest = Path(argv[0]), Path(argv[1]), argv[2:]
    if len(rest) % 2:
        raise SystemExit(f"bad options: {rest}")
    kwargs = {}
    for flag, value in zip(rest[::2], rest[1::2], strict=True):
        if flag not in _OPTIONS:
            raise SystemExit(f"unknown option {flag}")
        kwargs[_OPTIONS[flag]] = value
    return src, out, kwargs


def main(argv: list[str]) -> int:
    src, out, kwargs = _parse(argv)
    row: dict = {"ok": False}
    t0 = time.perf_counter()
    try:
        from kelpod.pipeline import convert

        t1 = time.perf_counter()
        res = convert(src, Path("model.ifc"), **kwargs)
        row["runtime_s"] = time.perf_counter() - t1
        row["import_s"] = t1 - t0
        row["graph"] = json.loads(res.graph.model_dump_json())
        row["scale_source"] = res.graph.scale_source
        row["pipeline_reopened_ok"] = bool(res.export and res.export.reopened_ok)
        row["timings"] = {t.step: t.seconds for t in res.timings}
        row["ok"] = True
    except BaseException as exc:  # noqa: BLE001 - every failure is a crash for the panel
        row["error"] = f"{type(exc).__name__}: {exc}"[:500]
        row["trace"] = traceback.format_exc()[-2000:]
        row["runtime_s"] = time.perf_counter() - t0
    out.write_text(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
