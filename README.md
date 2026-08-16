# Lane Robustness

Evaluating the robustness of lane-marking perception under adverse visual conditions.

Lane perception is a critical component of camera-based autonomous driving. This project
measures how the accuracy of lane-marking detection degrades as visibility worsens — heavier
fog and reduced illumination — comparing a **classical image-processing pipeline**
(Canny edge + Hough transform) against a **learned model** (TwinLiteNet).

The approach avoids running a driving simulator entirely: real highway images from the
[TuSimple](https://www.kaggle.com/datasets/manideep1108/tusimple) dataset are degraded in
software with physically-motivated transforms, keeping the ground-truth lane masks fixed so
degradation is controlled and repeatable. The result is a degradation curve of IoU/F1 versus
fog density and illumination level for each method.

## Preliminary results

On 200 validation frames (fog β sweep, exposure-factor sweep), the two methods degrade very
differently — a central finding:

| Condition          | Classical IoU | TwinLiteNet IoU |
|--------------------|--------------:|----------------:|
| Clear              | 0.108         | **0.381**       |
| Fog  β = 0.03      | 0.103         | 0.040           |
| Fog  β = 0.10      | 0.057         | **0.004**       |
| Exposure × 0.2     | 0.013         | **0.353**       |

The learned model (trained on BDD100K) is robust to reduced illumination but collapses
catastrophically under fog — falling *below* the classical baseline — while the classical
pipeline degrades gracefully under fog but fails in the dark. Full numbers are in
`outputs/degradation_curve.csv`.

## Degradation model

- **Fog**: Koschmieder attenuation `I_out = I·t + A·(1 − t)` with transmittance `t = e^(−β·d)`,
  where scene depth `d` grows from the bottom of the frame to the horizon (`degrade/fog.py`).
- **Illumination**: multiplicative exposure reduction with sensor-like Gaussian noise whose
  strength grows as light drops (`degrade/illumination.py`).

Both transforms are seeded and deterministic.

## Project layout

```
config.py                 # all experiment knobs (severity sweeps, CV params, paths)
degrade/                  # fog + illumination degradation transforms
models/                   # classical.py (Canny+Hough), twinlite.py (learned, ONNX)
tusimple/                 # dataset download helper + JSON annotations -> lane masks
eval/                     # metrics.py, run_experiment.py, plot.py
scripts/smoke_test.py     # end-to-end check on a synthetic road (no dataset needed)
```

## Getting started

Requires Python 3.10+.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Sanity-check the pipeline without downloading any data:

```sh
.venv/bin/python scripts/smoke_test.py
```

## Get the TuSimple dataset

TuSimple is free for academic use. The official download is a Kaggle mirror
(`manideep1108/tusimple`, ~10 GB).

1. Create a free account at [kaggle.com](https://www.kaggle.com), then
   **Settings → API → Create New Token**.
2. Save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json`.
3. Download, extract and verify the dataset (single command):

```sh
.venv/bin/python -m tusimple.download_tusimple --kaggle
```

This downloads the zip, extracts it into `data/tusimple/`, and verifies the layout. If the
Kaggle CLI reports a 403, open the dataset page in a browser and accept its terms once, then
rerun.

Convert the JSON lane annotations into binary ground-truth masks:

```sh
.venv/bin/python -m tusimple.make_masks
```

Masks are written to `data/masks/` mirroring the source paths.

## Run the experiment

Run the classical baseline over fog and illumination sweeps, then plot the results:

```sh
.venv/bin/python -m eval.run_experiment --methods classical
.venv/bin/python -m eval.plot
```

Outputs land in `outputs/`:

- `degradation_curve.csv` — mean IoU / F1 / inference time per (condition, severity, method)
- `raw_results.csv` — per-image results
- `*_curve.png` — degradation-curve plots

### Adding the learned baseline (TwinLiteNet)

The learned model is the official pretrained TwinLiteNet (0.44M params), run in PyTorch via
the vendored architecture in `models/twinlite_arch.py` (input 640×360, RGB, /255, as in the
authors' `test_image.py`):

1. Download the pretrained checkpoint from the
   [official TwinLiteNet repo](https://github.com/chequanghuy/TwinLiteNet)
   (`pretrained/best.pth`, ~1.8 MB).
2. Save it to `data/weights/best.pth` (or set `TWINLITE_MODEL=/path/to/best.pth`).
3. Re-run including both methods:

```sh
.venv/bin/python -m eval.run_experiment --methods classical twinlite
.venv/bin/python -m eval.plot
```

### Exporting a dataset (for decoupled perception experiments)

Add `--write-images` to `run_experiment` to also export the degraded RGB frames together with
their ground-truth masks:

```sh
.venv/bin/python -m eval.run_experiment --methods classical --write-images
```

Images are written under `outputs/dataset/<condition>/<severity>/`.

## Metrics

- **IoU** — intersection over union between predicted and ground-truth lane masks
- **F1** — harmonic mean of precision and recall
- **inference_ms** — wall-clock time per frame