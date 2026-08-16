import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import cv2

from degrade import apply_darkening, apply_fog
from eval import metrics
from models import TwinLiteModel, detect_lanes_classical


def make_synthetic_road():
    h, w = 720, 1280
    img = np.zeros((h, w, 3), np.uint8)
    horizon = int(h * 0.62)
    img[:horizon] = (135, 175, 225)
    img[horizon:] = (95, 95, 90)
    lanes = [
        [(int(w * 0.30), h), (int(w * 0.40), horizon + 60), (int(w * 0.42), horizon)],
        [(int(w * 0.62), h), (int(w * 0.55), horizon + 60), (int(w * 0.54), horizon)],
    ]
    gt = np.zeros((h, w), np.uint8)
    for lane in lanes:
        for a, b in zip(lane, lane[1:]):
            cv2.line(img, a, b, (250, 250, 250), 14, cv2.LINE_AA)
            cv2.line(gt, a, b, 255, 12)
    return img, gt


def main():
    img, gt = make_synthetic_road()
    print("synthetic frame:", img.shape)

    rows = []
    for beta in [0.0, 0.03, 0.06, 0.10]:
        fogged = apply_fog(img, beta)
        pred = detect_lanes_classical(fogged)
        rows.append(("fog", f"beta={beta:.2f}", metrics.iou(pred, gt), metrics.f1_score(pred, gt)))
    for factor in [1.0, 0.6, 0.35, 0.2]:
        dark = apply_darkening(img, factor)
        pred = detect_lanes_classical(dark)
        rows.append(("illum", f"factor={factor:.2f}", metrics.iou(pred, gt), metrics.f1_score(pred, gt)))

    print(f"{'condition':>5} {'severity':>10} {'IoU':>7} {'F1':>7}")
    for cond, sev, iou, f1 in rows:
        print(f"{cond:>5} {sev:>10} {iou:7.3f} {f1:7.3f}")

    try:
        TwinLiteModel()
        print("twinlite: model loaded (unexpected without weights)")
    except FileNotFoundError as e:
        print("twinlite: guard OK ->", str(e)[:60], "...")


if __name__ == "__main__":
    main()