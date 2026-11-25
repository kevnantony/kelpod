"""Per-class IoU of an exported model (TorchScript ``.pt``) on dev-val page sets, on the CPU.

    uv run python -m kelpod_train.val_iou ~/kelpod-models/kelpod-seg/r34/model.pt \
        --val synth,synth_v1,swiss,swiss_v1,cubicasa --limit 60

Same metric and pages as the validation inside ``kelpod_train.train`` (``validate``), so models
from different runs can be compared on the same val sets (val only, never test).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from kelpod_train.train import validate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--val", default="synth,synth_v1,swiss,swiss_v1,cubicasa")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    model = torch.jit.load(a.model, map_location="cpu")
    model.eval()
    res = validate(model, torch.device("cpu"), a.val.split(","), limit=a.limit)
    for src, r in res.items():
        print(f"{src:10s} " + " ".join(f"{k} {v:.3f}" for k, v in r.items()))
    score = float(np.mean([np.mean([r["wall"], r["window"], r["door"]]) for r in res.values()]))
    print(json.dumps({"model": a.model, "val": res, "score": score}))


if __name__ == "__main__":
    main()
