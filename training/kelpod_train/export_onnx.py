"""Export a trained checkpoint to ONNX for the main environment (onnxruntime, no torch).

    uv run python -m kelpod_train.export_onnx ~/kelpod-models/kelpod-seg/v1-s1

Reads ``<run>/best.pth`` and ``<run>/best_meta.json``, writes ``<run>/model.onnx`` (dynamic height
and width, opset 17, metadata key ``kelpod`` = the training meta) and checks it against the
PyTorch model on a test page (max logit difference and argmax agreement). Prints the SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from kelpod_train.train import Wrapped, build_model


def _meta(run: Path) -> dict:
    if (run / "best_meta.json").exists():
        return json.loads((run / "best_meta.json").read_text())
    extra = {"meta.json": ""}  # older runs (r34): the meta lives in the TorchScript file
    torch.jit.load(str(run / "model.pt"), map_location="cpu", _extra_files=extra)
    return json.loads(extra["meta.json"])


def export(run: Path, opset: int = 17, out: Path | None = None) -> Path:
    meta = _meta(run)
    model = Wrapped(build_model(meta["encoder"], weights=None))
    model.load_state_dict(torch.load(run / "best.pth", map_location="cpu", weights_only=True))
    model.eval()
    out = out or run / "model.onnx"
    x = torch.ones(1, 1, 512, 512)
    torch.onnx.export(model, x, str(out), opset_version=opset, input_names=["image"],
                      output_names=["logits"], dynamo=False,
                      dynamic_axes={"image": {2: "h", 3: "w"}, "logits": {2: "h", 3: "w"}})
    import onnx

    m = onnx.load(str(out))
    m.metadata_props.add(key="kelpod", value=json.dumps(meta))
    onnx.save(m, str(out))

    # check against torch on a random page-like input
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    page = np.ones((1, 1, 1024, 768), np.float32)
    page[0, 0, 200:220, 100:700] = 0
    page[0, 0, 200:900, 100:120] = 0
    page += rng.normal(0, 0.05, page.shape).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(page)).numpy()
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    t0 = time.perf_counter()
    got = sess.run(None, {"image": page})[0]
    dt = time.perf_counter() - t0
    agree = float((ref.argmax(1) == got.argmax(1)).mean())
    print(f"max |diff| {np.abs(ref - got).max():.2e}, argmax agreement {agree:.6f}, "
          f"onnx 1024x768 {dt:.2f} s")
    if agree < 0.999:
        raise SystemExit("ONNX export disagrees with the PyTorch model")
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"{out} sha256 {h}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    export(a.run.expanduser(), out=a.out.expanduser() if a.out else None)
