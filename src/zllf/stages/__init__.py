"""Pipeline stages. Use :func:`get_stage` to obtain an implementation by name."""

from __future__ import annotations

import importlib

from zllf.stages.base import Stage

_REGISTRY: dict[str, str] = {
    "ingest": "zllf.stages.ingest.stage:IngestStage",
    "sanitize": "zllf.stages.sanitize.stage:SanitizeStage",
    "synth": "zllf.stages.synth.stage:SynthStage",
    "train": "zllf.stages.train.stage:TrainStage",
    "evaluate": "zllf.stages.evaluate.stage:EvaluateStage",
    "package": "zllf.stages.package.stage:PackageStage",
}


def get_stage(name: str) -> Stage:
    module_name, cls_name = _REGISTRY[name].split(":")
    cls = getattr(importlib.import_module(module_name), cls_name)
    stage: Stage = cls()
    return stage
