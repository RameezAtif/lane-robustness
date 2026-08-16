"""Convert TuSimple JSON lane annotations into binary ground-truth masks.

Each annotation gives, per lane, a list of x-coordinates sampled at fixed
h_samples rows. Lanes are drawn as polylines thickened to MASK_LINE_THICKNESS px.
Masks mirror the raw_file path under MASKS_DIR so they can be added/updated
independently of the raw dataset.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import config
from tusimple.download_tusimple import parse_labels


def lane_to_points(lane_x, h_samples):
    pts = [(x, y) for x, y in zip(lane_x, h_samples) if x >= 0]
    if not pts:
        return None
    return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)


def build_mask(annotation, height, width, thickness=None):
    thickness = thickness or config.MASK_LINE_THICKNESS
    mask = np.zeros((height, width), np.uint8)
    h_samples = annotation["h_samples"]
    for lane_x in annotation["lanes"]:
        pts = lane_to_points(lane_x, h_samples)
        if pts is None:
            continue
        cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=thickness)
    return mask


def make_masks(label_files=None, out_dir=None, verbose=True):
    out_dir = Path(out_dir or config.MASKS_DIR)
    train = config.DATA_ROOT / config.TRAIN_SUBDIR
    label_files = label_files or config.LABEL_FILES
    n = 0
    for name in label_files:
        label_path = train / name
        if not label_path.is_file():
            print(f"skip missing {label_path}")
            continue
        entries = parse_labels(label_path)
        for ann in tqdm(entries, desc=name, disable=not verbose):
            raw = ann["raw_file"].lstrip("/")
            src = train / raw
            if not src.is_file():
                print(f"skip missing image {src}")
                continue
            img = cv2.imread(str(src))
            if img is None:
                print(f"could not read {src}")
                continue
            h, w = img.shape[:2]
            mask = build_mask(ann, h, w)
            dst = out_dir / raw
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst = dst.with_suffix(".png")
            cv2.imwrite(str(dst), mask)
            n += 1
    print(f"wrote {n} masks -> {out_dir}")
    return n


def main():
    ap = argparse.ArgumentParser(description="make TuSimple lane masks")
    ap.add_argument("--out", default=str(config.MASKS_DIR))
    ap.add_argument("--labels", nargs="*", default=config.LABEL_FILES)
    args = ap.parse_args()
    make_masks(args.labels, args.out)


if __name__ == "__main__":
    main()