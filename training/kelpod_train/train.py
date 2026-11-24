"""Fine-tune a U-Net (segmentation_models_pytorch, ImageNet encoder) on Kelpod's 5 classes.

    uv run python -m kelpod_train.train --encoder resnet34 --iters 30000 --out ~/kelpod-models/kelpod-seg/r34

Training data: PNG pairs under ``~/kelpod-data/train/<source>/`` (see ``kelpod_train.build``),
sampled per source with ``--mix``. Validation: full pages from ``~/kelpod-data/val/<source>/``,
per-class IoU, logged as JSON lines in ``<out>/log.jsonl``. The best checkpoint (mean IoU of
wall, window, door) is exported as TorchScript ``model.pt`` for ``kelpod.segment.learned``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DATA = Path.home() / "kelpod-data"
N_CLASSES = 5
IGNORE = 255
CLASS_NAMES = ("background", "wall", "window", "door", "room")


def _pairs(d: Path) -> list[tuple[Path, Path]]:
    return [(p, p.with_name(p.name.replace("_img.png", "_lab.png")))
            for p in sorted(d.glob("*_img.png"))
            if p.with_name(p.name.replace("_img.png", "_lab.png")).exists()]


class CropDataset(Dataset):
    """Random 512 px crops (mostly centred on drawn content) with geometric and photometric
    augmentation. One epoch = ``length`` samples drawn by source weights."""

    def __init__(self, sources: dict[str, float], size: int, length: int, seed: int = 0):
        self.sources = sources
        self.size, self.length, self.seed = size, length, seed
        self._scan()

    def _scan(self) -> None:
        """(Re)list the files; sources still being built are picked up as they grow."""
        self.items = {s: _pairs(DATA / "train" / s) for s in self.sources}
        self.items = {s: v for s, v in self.items.items() if v}
        w = np.array([self.sources[s] for s in self.items], float)
        self.names, self.p = list(self.items), w / w.sum()
        self._scanned = time.time()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int):
        if time.time() - self._scanned > 900:
            self._scan()
        rng = np.random.default_rng([self.seed, i, int(time.time() * 1e6) % (1 << 31)])
        src = self.names[rng.choice(len(self.names), p=self.p)]
        ip, lp = self.items[src][rng.integers(len(self.items[src]))]
        img = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        lab = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE)
        s = self.size
        # mild scale jitter (wall thickness carries meaning, so keep it small)
        f = float(rng.uniform(0.8, 1.25))
        if abs(f - 1) > 0.02:
            img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1
                             else cv2.INTER_LINEAR)
            lab = cv2.resize(lab, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        h, w = img.shape
        if h < s or w < s:
            ph, pw = max(0, s - h), max(0, s - w)
            img = np.pad(img, ((0, ph), (0, pw)), constant_values=255)
            lab = np.pad(lab, ((0, ph), (0, pw)), constant_values=0)
            h, w = img.shape
        if rng.random() < 0.85:
            ys, xs = np.nonzero(lab[::8, ::8] > 0)
            if len(ys):
                k = rng.integers(len(ys))
                cy, cx = ys[k] * 8, xs[k] * 8
            else:
                cy, cx = h // 2, w // 2
            y0 = int(np.clip(cy - s // 2 + rng.integers(-s // 4, s // 4), 0, h - s))
            x0 = int(np.clip(cx - s // 2 + rng.integers(-s // 4, s // 4), 0, w - s))
        else:
            y0, x0 = int(rng.integers(0, h - s + 1)), int(rng.integers(0, w - s + 1))
        img, lab = img[y0:y0 + s, x0:x0 + s], lab[y0:y0 + s, x0:x0 + s]
        k = int(rng.integers(4))
        img, lab = np.rot90(img, k), np.rot90(lab, k)
        if rng.random() < 0.5:
            img, lab = img[:, ::-1], lab[:, ::-1]
        x = img.astype(np.float32) / 255.0
        # photometric: contrast / gamma, blur, noise (pages are already cleaned, keep it mild)
        if rng.random() < 0.5:
            x = np.clip(x ** float(rng.uniform(0.7, 1.4)), 0, 1)
        if rng.random() < 0.3:
            x = cv2.GaussianBlur(x, (0, 0), float(rng.uniform(0.3, 1.0)))
        if rng.random() < 0.3:
            x = np.clip(x + rng.normal(0, rng.uniform(0.01, 0.06), x.shape).astype(np.float32),
                        0, 1)
        return torch.from_numpy(np.ascontiguousarray(x))[None], torch.from_numpy(
            np.ascontiguousarray(lab).astype(np.int64))


def build_model(encoder: str, weights: str | None = "imagenet"):
    import segmentation_models_pytorch as smp

    return smp.Unet(encoder_name=encoder, encoder_weights=weights, in_channels=1,
                    classes=N_CLASSES)


class Wrapped(torch.nn.Module):
    """What gets exported: (1, 1, H, W) in [0, 1] -> logits; normalisation inside."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net((x - 0.5) / 0.25)


@torch.no_grad()
def predict_full(model, img: np.ndarray, device, tile: int = 1024) -> np.ndarray:
    h, w = img.shape
    x = torch.from_numpy(img.astype(np.float32) / 255.0)[None, None].to(device)
    ph, pw = -h % 32, -w % 32
    x = F.pad(x, (0, pw, 0, ph), value=1.0)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = model(x)
    return out[0, :, :h, :w].float().argmax(0).cpu().numpy()


def validate(model, device, sources: list[str], limit: int = 60) -> dict:
    model.eval()
    res = {}
    for src in sources:
        inter = np.zeros(N_CLASSES)
        union = np.zeros(N_CLASSES)
        for ip, lp in _pairs(DATA / "val" / src)[:limit]:
            img = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
            lab = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE)
            pr = predict_full(model, img, device)
            v = lab != IGNORE
            for c in range(N_CLASSES):
                a, b = (pr == c) & v, (lab == c) & v
                inter[c] += (a & b).sum()
                union[c] += (a | b).sum()
        iou = inter / np.maximum(union, 1)
        res[src] = {n: float(i) for n, i in zip(CLASS_NAMES, iou, strict=True)}
    model.train()
    return res


def export(model, path: Path, meta: dict) -> None:
    m = Wrapped(model.net if isinstance(model, Wrapped) else model).eval().cpu().float()
    ex = torch.ones(1, 1, 512, 512)
    with torch.no_grad():
        ts = torch.jit.trace(m, ex)
    ts.save(str(path), _extra_files={"meta.json": json.dumps(meta)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mix", default="synth:0.35,swiss:0.25,resplan:0.2,cubicasa:0.2")
    ap.add_argument("--val", default="synth,cubicasa,swiss")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--val-limit", type=int, default=60)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--init", default=None, help="state dict to start from (continue training)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mix = {k: float(v) for k, v in (s.split(":") for s in a.mix.split(","))}
    ds = CropDataset(mix, a.size, a.iters * a.batch)
    print({s: len(v) for s, v in ds.items.items()}, flush=True)
    dl = DataLoader(ds, batch_size=a.batch, num_workers=a.workers, pin_memory=True,
                    drop_last=True, persistent_workers=True, prefetch_factor=4)
    model = Wrapped(build_model(a.encoder)).to(device)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=True))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    warm = 500

    def lr_at(it):
        if it < warm:
            return (it + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (it - warm) / max(1, a.iters - warm))) * 0.98 + 0.02

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    cw = torch.tensor([1.0, 2.0, 4.0, 4.0, 1.0], device=device)
    import segmentation_models_pytorch as smp

    dice = smp.losses.DiceLoss("multiclass", ignore_index=IGNORE)
    log = open(out / "log.jsonl", "a")
    best = -1.0
    t0 = time.time()
    run = 0.0
    val_srcs = [s for s in a.val.split(",") if (DATA / "val" / s).exists()]
    for it, (x, y) in enumerate(dl):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x)
        logits = logits.float()
        loss = F.cross_entropy(logits, y, weight=cw, ignore_index=IGNORE) + dice(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * float(loss) if it else float(loss)
        if (it + 1) % 100 == 0:
            print(f"it {it + 1} loss {run:.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{(time.time() - t0) / 3600:.2f} h", flush=True)
        if (it + 1) % a.eval_every == 0 or it + 1 == a.iters:
            v = validate(model, device, val_srcs, limit=a.val_limit)
            score = float(np.mean([np.mean([r["wall"], r["window"], r["door"]])
                                   for r in v.values()]))
            rec = {"iter": it + 1, "loss": run, "val": v, "score": score,
                   "hours": (time.time() - t0) / 3600}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            torch.save(model.state_dict(), out / "last.pth")
            if score > best:
                best = score
                torch.save(model.state_dict(), out / "best.pth")
                meta = {"encoder": a.encoder, "classes": CLASS_NAMES, "iter": it + 1,
                        "m_per_px": 0.0254 / 200 * 100, "input": "gray/255, white=1",
                        "val": v, "mix": mix}
                export(model, out / "model.pt", meta)
                model.to(device)
        if it + 1 >= a.iters:
            break
    print(f"done, best score {best:.4f}, {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
