"""Helper for downloading the TuSimple lane-detection dataset.

TuSimple is free for academic use but requires a one-time registration:
  1. Go to https://github.com/TuSimple/tusimple-benchmark and follow the
     "Download" / dataset agreement links (you'll be asked to fill a short form).
  2. Download `Train Set` (includes label_data_0313/0531/0601.json) and extract it.
  3. Place the extracted folder so that this layout holds:

       <data root>/tusimple/train_set/clips/..../1.jpg
       <data root>/tusimple/train_set/label_data_0313.json  (etc.)

Run `python -m tusimple.download_tusimple --check` to verify the layout.
If you placed the archive somewhere else, pass `--tar path/to/train_set.tar.gz` and
this script will extract it into place for you.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import config


def _run_kaggle():
    if not (Path.home() / ".kaggle" / "kaggle.json").is_file():
        print(
            "No API token found. Create one at kaggle.com -> Settings -> API -> "
            "Create New Token, then save the downloaded kaggle.json to "
            f"{Path.home() / '.kaggle' / 'kaggle.json'}",
            file=sys.stderr,
        )
        sys.exit(2)
    (config.DATA_ROOT.parent / "tusimple").mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download",
         "-d", "manideep1108/tusimple", "-p", str(config.DATA_ROOT.parent)],
        check=True,
    )


def extract_archive(archive_path):
    dest = config.DATA_ROOT
    dest.mkdir(parents=True, exist_ok=True)
    archive_path = Path(archive_path)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest)
    else:
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(dest, filter="data")
    _normalize_layout()
    print(f"Extracted {archive_path} -> {dest}")


def _normalize_layout():
    """Move a nested train_set/test_set into place, whatever the archive's root."""
    dest = config.DATA_ROOT
    if not (dest / config.TRAIN_SUBDIR).is_dir():
        for cand in dest.rglob("train_set"):
            if cand.is_dir():
                shutil.move(str(cand), str(dest / config.TRAIN_SUBDIR))
                break
    if not (dest / "test_set").is_dir():
        for cand in dest.rglob("test_set"):
            if cand.is_dir():
                shutil.move(str(cand), str(dest / "test_set"))
                break
    for leftover in (p for p in dest.iterdir() if p.is_dir() and p.name not in
                     (config.TRAIN_SUBDIR, "test_set", "__MACOSX")):
        print(f"note: leftover directory {leftover} (safe to delete)")


def extract_newest_zip():
    zips = sorted(config.DATA_ROOT.parent.glob("tusimple*.zip"),
                  key=lambda p: p.stat().st_mtime)
    if not zips:
        print("No downloaded zip found under", config.DATA_ROOT.parent, file=sys.stderr)
        sys.exit(1)
    extract_archive(zips[-1])


def check_layout():
    issues = []
    root = config.DATA_ROOT
    train = root / config.TRAIN_SUBDIR
    if not train.is_dir():
        issues.append(f"missing {train}")
    else:
        for name in config.LABEL_FILES:
            p = train / name
            if not p.is_file():
                issues.append(f"missing {p}")
        clips = train / "clips"
        if not clips.is_dir():
            issues.append(f"missing {clips}")
        else:
            subs = list(clips.glob("*"))
            if not subs:
                issues.append(f"no clips found under {clips}")
    if not issues:
        n = sum(1 for _ in clips.glob("*/*/*.jpg"))
        print(f"OK - layout valid, {n} jpgs found.")
    else:
        print("Missing pieces:")
        for i in issues:
            print("  -", i)
        print("See module docstring for the registration/download steps.")
    return issues


def parse_labels(label_path):
    """Yield annotation dicts for a TuSimple label json (one json object per line)."""
    entries = []
    with open(label_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("sample_data"):
                continue
            entries.append(obj)
    return entries


def main():
    ap = argparse.ArgumentParser(description="TuSimple helper")
    ap.add_argument("--tar", help="path to train_set.tar.gz to extract into place")
    ap.add_argument("--kaggle", action="store_true",
                    help="download via the Kaggle CLI, then extract the newest tusimple*.zip")
    ap.add_argument("--check", action="store_true", help="verify dataset layout")
    args = ap.parse_args()
    if args.tar:
        extract_archive(args.tar)
    if args.kaggle:
        _run_kaggle()
        extract_newest_zip()
    if check_layout():
        sys.exit(1)


if __name__ == "__main__":
    main()