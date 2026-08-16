"""TwinLiteNet wrapper (learned baseline).

Runs the pretrained TwinLiteNet ONNX model via onnxruntime. Torch is NOT required;
keep the pipeline light. The checkpoint is downloaded separately (see
download instructions below) and cached at data/weights/twinlitelite.onnx.

Downloads (official repo: https://github.com/chequanghung/TwinLiteNet):
  - Put the pretrained ONNX model at data/weights/twinlitelite.onnx,
    or set the TWINLITE_MODEL env var to its path.
"""
import os
from pathlib import Path

import numpy as np

import config

INPUT_H, INPUT_W = 320, 640


def _load_session(model_path):
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    if ort.get_available_providers() and "CoreMLExecutionProvider" in ort.get_available_providers():
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(str(model_path), providers=providers)


class TwinLiteModel:
    def __init__(self, model_path=None):
        model_path = model_path or os.environ.get("TWINLITE_MODEL") or str(config.ROOT / "data" / "weights" / "twinlitelite.onnx")
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"TwinLiteNet weights not found at {model_path}. Download the pretrained "
                f"ONNX model from the official TwinLiteNet repo "
                f"(https://github.com/chequanghung/TwinLiteNet) and place it there, "
                f"or set TWINLITE_MODEL."
            )
        self.session = _load_session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def detect_lane_mask(self, img, threshold=0.5):
        ih, iw = img.shape[:2]
        resized = np.asarray(img.resize((INPUT_W, INPUT_H))) if hasattr(img, "resize") else _resize(img)
        inp = resized[None].transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        out = self.session.run([self.output_name], {self.input_name: inp})[0]
        lane_probs = out[0, 1] if out.shape[1] == 2 else out[0, 0]
        lane_probs = np.clip(lane_probs, 0.0, 1.0)
        small = (lane_probs >= threshold).astype(np.uint8) * 255
        return _resize_back(small, iw, ih)


def _resize(img):
    import cv2

    return cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)


def _resize_back(mask, w, h):
    import cv2

    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)