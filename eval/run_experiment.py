"""Run the robustness experiment and write a degradation-curve CSV.

Sweeps fog beta and illumination factor, runs the selected lane detectors on the
degraded frames, and reports mean IoU / F1 / inference time per (condition,
method, severity) level.
"""
import argparse
import csv
import random
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import config
from degrade import apply_darkening, apply_fog
from eval import metrics
from models import TwinLiteModel, detect_lanes_classical
from tusimple.download_tusimple import parse_labels


def load_annotations(label_name, limit):
    label_path = config.DATA_ROOT / config.TRAIN_SUBDIR / label_name
    if not label_path.is_file():
        raise FileNotFoundError(
            f"no label file at {label_path}.\n"
            "The TuSimple dataset is not installed. See `python -m tusimple.download_tusimple --check` "
            "for the free academic registration + download steps."
        )
    entries = parse_labels(label_path)[:limit]
    for ann in entries:
        raw = ann["raw_file"].lstrip("/")
        img_path = config.DATA_ROOT / config.TRAIN_SUBDIR / raw
        mask_path = (config.MASKS_DIR / raw).with_suffix(".png")
        ann["_img"] = str(img_path)
        ann["_mask"] = str(mask_path)
    return entries


def _load_mask(path):
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def _run_classical(img, sev):
    return detect_lanes_classical(img)


def _run_twinlite(tl, img, sev):
    return tl.detect_lane_mask(img)


def build_conditions(do_fog, do_illum):
    conds = []
    if do_fog:
        conds += [("fog", b) for b in config.FOG_BETAS]
    if do_illum:
        conds += [("illumination", f) for f in config.ILLUM_FACTORS]
    return conds


def apply_condition(img, cond, sev):
    if cond == "fog":
        return apply_fog(img, sev)
    if cond == "illumination":
        return apply_darkening(img, sev, seed=config.SEED)
    raise ValueError(cond)


def main():
    ap = argparse.ArgumentParser(description="lane-robustness degradation experiment")
    ap.add_argument("--limit", type=int, default=config.SWEEP_LIMIT)
    ap.add_argument("--label", default="label_data_0601.json")
    ap.add_argument("--conditions", nargs="*", default=["fog", "illumination"])
    ap.add_argument("--methods", nargs="*", default=["classical"])
    ap.add_argument("--outdir", default=str(config.OUTPUTS_DIR))
    ap.add_argument("--write-images", action="store_true", help="export degraded RGB+GT dataset")
    args = ap.parse_args()

    assert args.methods, "pass at least one method, e.g. --methods classical twinlite"

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    if "twinlite" in args.methods:
        import torch

        torch.manual_seed(config.SEED)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    entries = load_annotations(args.label, args.limit)
    print(f"loaded {len(entries)} validation frames from {args.label}")

    twin = TwinLiteModel() if "twinlite" in args.methods else None

    rows = []
    raw_rows = []
    conditions = build_conditions("fog" in args.conditions, "illumination" in args.conditions)

    for cond, sev in conditions:
        tag = f"{cond}={sev:.3g}"
        for m in args.methods:
            ious, f1s, times = [], [], []
            for ann in tqdm(entries, desc=tag):
                img = cv2.imread(ann["_img"], cv2.IMREAD_COLOR)
                mask_gt = _load_mask(ann["_mask"])
                if img is None or mask_gt is None:
                    continue
                degraded = apply_condition(img, cond, sev)
                t0 = time.perf_counter()
                if m == "classical":
                    pred = _run_classical(degraded, sev)
                elif m == "twinlite":
                    pred = _run_twinlite(twin, degraded, sev)
                else:
                    raise ValueError(m)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                iou = metrics.iou(pred, mask_gt)
                f1 = metrics.f1_score(pred, mask_gt)
                ious.append(iou)
                f1s.append(f1)
                times.append(dt_ms)
                raw_rows.append([cond, sev, m, ann["raw_file"], iou, f1, dt_ms])

                if args.write_images:
                    base = outdir / "dataset" / cond / f"{sev:g}"
                    name = ann["raw_file"].replace("/", "__")
                    cv2.imwrite(str(base / f"{name}.png"), degraded)
                    cv2.imwrite(str(base / f"{name}_mask.png"), mask_gt)

            miou, siou = metrics.mean_std(ious)
            mf1, sf1 = metrics.mean_std(f1s)
            mtime = float(np.mean(times)) if times else float("nan")
            rows.append([cond, sev, m, round(miou, 4), round(siou, 4),
                         round(mf1, 4), round(sf1, 4), round(mtime, 2)])
            print(f"{tag} [{m}] IoU={miou:.4f} F1={mf1:.4f} t={mtime:.1f}ms")

    out_csv = outdir / "degradation_curve.csv"
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["condition", "severity", "method", "IoU_mean", "IoU_std",
                         "F1_mean", "F1_std", "inference_ms"])
        writer.writerows(rows)
    with open(outdir / "raw_results.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["condition", "severity", "method", "image", "IoU", "F1", "inference_ms"])
        writer.writerows(raw_rows)
    print(f"saved {out_csv}")


if __name__ == "__main__":
    main()