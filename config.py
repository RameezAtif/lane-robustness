import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_ROOT = ROOT / "data" / "tusimple"
MASKS_DIR = ROOT / "data" / "masks"

OUTPUTS_DIR = ROOT / "outputs"

TRAIN_SUBDIR = "train_set"
LABEL_FILES = ["label_data_0313.json", "label_data_0531.json", "label_data_0601.json"]

FRAME_H, FRAME_W = 720, 1280

HORIZON_FRAC = 0.62
FOG_D_FAR_M = 150.0
FOG_ATMOSPHERIC_LIGHT = 200.0

MASK_LINE_THICKNESS = 12

SWEEP_LIMIT = 200
FOG_BETAS = [0.0, 0.015, 0.03, 0.045, 0.06, 0.08, 0.10]
ILLUM_FACTORS = [1.0, 0.75, 0.6, 0.5, 0.4, 0.3, 0.2]

CV_CANNY_LOW, CV_CANNY_HIGH = 40, 120
CV_NEAR_TOP_FRAC = 0.78
CV_HOUGH_THRESHOLD = 25
CV_HOUGH_MIN_LEN = 20
CV_HOUGH_MAX_GAP = 15
CV_MAX_LANES = 4

SEED = 0