"""Run one worktree's pipeline on one input, for ``scripts/locked_eval.py``.

Executed **by file path** with the worktree's own interpreter, in an empty temporary directory
that holds only the input file::

    <worktree python> -I locked_runner.py plan.pdf result.json

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


def main(argv: list[str]) -> int:
    src, out = Path(argv[0]), Path(argv[1])
    row: dict = {"ok": False}
    t0 = time.perf_counter()
    try:
        from kelpod.pipeline import convert

        t1 = time.perf_counter()
        res = convert(src, Path("model.ifc"))
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
