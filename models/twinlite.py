"""TwinLiteNet learned baseline.

Runs the official pretrained checkpoint (pretrained/best.pth, 0.44M params) from
https://github.com/chequanghuy/TwinLiteNet through the vendored architecture in
models/twinlite_arch.py. Input is resized to 640x360 (matching the official
test_image.py), converted BGR->RGB, normalised to [0,1]; the lane branch output is
the second of the two logit maps, thresholded by argmax over its channel axis.
"""
import os
from pathlib import Path

import cv2
import numpy as np
import torch

import config
from models import twinlite_arch

INPUT_W, INPUT_H = 640, 360


def _default_weight_path():
    return (os.environ.get("TWINLITE_MODEL")
            or str(config.DATA_ROOT.parent / "weights" / "best.pth"))


def _strip_dataparallel(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


class TwinLiteModel:
    def __init__(self, model_path=None, device=None):
        model_path = model_path or _default_weight_path()
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"TwinLiteNet weights not found at {model_path}. Download them from "
                "https://github.com/chequanghuy/TwinLiteNet (pretrained/best.pth) and "
                f"place them at {Path(model_path)}, or set TWINLITE_MODEL."
            )
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = twinlite_arch.TwinLiteNet()
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        state = _strip_dataparallel(state)
        self.model.load_state_dict(state)
        self.model.eval().to(self.device)

    @torch.no_grad()
    def detect_lane_mask(self, img, threshold=None):
        ih, iw = img.shape[:2]
        resized = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)[None]).float().to(self.device) / 255.0
        _, lane_logits = self.model(tensor)
        lane = lane_logits[0].argmax(0).byte().cpu().numpy()
        mask = (lane > 0).astype(np.uint8) * 255
        return cv2.resize(mask, (iw, ih), interpolation=cv2.INTER_NEAREST)