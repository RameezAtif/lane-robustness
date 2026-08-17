# Results snapshot

Committed experiment results (200 validation frames, `label_data_0601.json`).

| File | Contents |
|------|----------|
| `degradation_curve.csv` | Mean IoU / F1 / inference time per (condition, severity, method) |
| `raw_results.csv` | Per-image results for every frame and severity level |
| `IoU_mean_curve.png` | IoU degradation curves (fog + illumination) |
| `F1_mean_curve.png` | F1 degradation curves (fog + illumination) |

To regenerate these numbers from a clean checkout, follow the steps in the
top-level README, then run:

```sh
.venv/bin/python -m eval.run_experiment --methods classical twinlite
.venv/bin/python -m eval.plot
```

Results are written to `outputs/`; the committed copies here are the exact
artifacts referenced in the README's "Preliminary results" table.