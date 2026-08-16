"""Plot IoU/F1 vs severity from the experiment CSV."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config


def plot(csv_path):
    df = pd.read_csv(csv_path)
    outdir = Path(csv_path).parent

    for metric in ["IoU_mean", "F1_mean"]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, cond in zip(axes, ["fog", "illumination"]):
            sub = df[df.condition == cond]
            for m, col in zip(sub.method.unique(), ["tab:blue", "tab:orange"]):
                s = sub[sub.method == m]
                ax.plot(s.severity, s[metric], "o-", color=col, label=m)
            title = cond
            ax.set_xlabel("beta (fog)" if cond == "fog" else "exposure factor (illumination)")
            if cond == "illumination":
                ax.invert_xaxis()
            ax.set_title(title)
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_ylabel(metric)
        fig.suptitle(f"Lane-perception {metric} vs degradation")
        fig.tight_layout()
        fig.savefig(outdir / f"{metric}_curve.png", dpi=150)
        plt.close(fig)
    print(f"plots written to {outdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(config.OUTPUTS_DIR / "degradation_curve.csv"))
    args = ap.parse_args()
    plot(args.csv)


if __name__ == "__main__":
    main()