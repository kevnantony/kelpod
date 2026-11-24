"""Zero-shot baseline: the released CubiCasa5k model (CC BY-NC 4.0) mapped to Kelpod's 5 classes.

Room channels: Wall -> wall; Background, Outdoor, Railing -> background; every room type ->
room interior. Icon channels: Window -> window, Door -> door (drawn over walls, since CubiCasa
walls run through openings). ``scale`` resizes the page before inference (CubiCasa's own walls
are ~19 px thick, Kelpod's 24 cm walls at 200 dpi also ~19 px, so 1.0 is the natural choice;
the evaluation picks it on development seeds, not on the test split).
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from kelpod.segment.learned import BG, DOOR, ROOM, WALL, WINDOW

DEFAULT_WEIGHTS = Path.home() / "kelpod-models/cubicasa5k/model_best_val_loss_var.pkl"
ROOM_MAP = np.array([BG, BG, WALL, ROOM, ROOM, ROOM, ROOM, ROOM, BG, ROOM, ROOM, ROOM], np.uint8)


class CubiCasaPredictor:
    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, *, scale: float = 1.0,
                 device: str | None = None) -> None:
        import torch

        from kelpod_train.third_party.cubicasa_hg import hg_furukawa_original

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu":
            torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
        m = hg_furukawa_original(51)
        m.conv4_ = torch.nn.Conv2d(256, 44, bias=True, kernel_size=1)
        m.upsample = torch.nn.ConvTranspose2d(44, 44, kernel_size=4, stride=4)
        ck = torch.load(str(weights), map_location="cpu", weights_only=True)
        m.load_state_dict(ck["model_state"])
        self.model = m.eval().to(self.device)
        self.scale = scale

    def __call__(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        img = gray
        if abs(self.scale - 1) > 1e-3:
            img = cv2.resize(gray, (round(w * self.scale), round(h * self.scale)),
                             interpolation=cv2.INTER_AREA if self.scale < 1 else cv2.INTER_LINEAR)
        hh, ww = img.shape
        ph, pw = -hh % 32, -ww % 32
        img = np.pad(img, ((0, ph), (0, pw)), constant_values=255)
        x = np.repeat((2 * (img.astype(np.float32) / 255.0) - 1)[None, None], 3, axis=1)
        with self.torch.inference_mode():
            out = self.model(self.torch.from_numpy(x).to(self.device))[0, :, :hh, :ww]
            rooms = out[21:33].argmax(0).cpu().numpy()
            icons = out[33:44].argmax(0).cpu().numpy()
        lab = ROOM_MAP[rooms]
        lab[icons == 1] = WINDOW
        lab[icons == 2] = DOOR
        if (hh, ww) != (h, w):
            lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST)
        return lab
