"""Synthetic German-style floor plans with ground truth, closure metric, timing.

* :mod:`kelpod.evaluation.synth`: ``generate(seed, degrade) -> SyntheticPlan``, ``save``, ``load``;
* :mod:`kelpod.evaluation.metrics`: ``closure_metrics(graph)``,
  ``compare_to_ground_truth(pred, gt)``;
* ``scripts/benchmark.py``: the whole pipeline on N synthetic plans.

Submodules are imported lazily so ``python -m kelpod.evaluation.synth`` runs cleanly.
"""

from __future__ import annotations

from importlib import import_module

__all__ = ["metrics", "synth"]


def __getattr__(name: str):
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(name)
