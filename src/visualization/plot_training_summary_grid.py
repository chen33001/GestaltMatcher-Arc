"""
Create a 2x2 summary figure from TensorBoard scalar CSV exports.

The figure contains:
  - training loss
  - validation loss
  - validation Top-1 accuracy
  - validation Top-5 accuracy
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODEL_PATTERNS = {
    "Frontal-only": "*s101*view_aware_frontal*.csv",
    "Profile-only": "*s102*view_aware_profile*.csv",
    "Mixed-view": "*s103*view_aware_mixed*.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Create a 2x2 training summary figure.")
    parser.add_argument("--training_loss_dir", required=True)
    parser.add_argument("--validation_loss_dir", required=True)
    parser.add_argument("--val_top1_dir", required=True)
    parser.add_argument("--val_top5_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smooth_window", type=int, default=15)
    return parser.parse_args()


def find_one(directory, pattern):
    matches = sorted(Path(directory).glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No CSV matched {pattern} in {directory}")
    if len(matches) > 1:
        joined = "\n".join(str(path) for path in matches)
        raise ValueError(f"Multiple CSVs matched {pattern} in {directory}:\n{joined}")
    return matches[0]


def load_curves(directory, smooth_window):
    curves = {}
    for label, pattern in MODEL_PATTERNS.items():
        path = find_one(directory, pattern)
        df = pd.read_csv(path)
        required = {"Step", "Value"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        df = df.rename(columns={"Step": "step", "Value": "value"})
        df = df.sort_values("step").reset_index(drop=True)
        df["smooth_value"] = df["value"].rolling(
            window=smooth_window,
            min_periods=1,
            center=True,
        ).mean()
        curves[label] = df
    return curves


def plot_panel(ax, curves, title, ylabel):
    for label, df in curves.items():
        ax.plot(df["step"], df["smooth_value"], linewidth=1.8, label=label)
    ax.set_title(title)
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def main():
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    panels = [
        (
            load_curves(args.training_loss_dir, args.smooth_window),
            "Training loss",
            "Cross-entropy loss",
        ),
        (
            load_curves(args.validation_loss_dir, args.smooth_window),
            "Validation loss",
            "Cross-entropy loss",
        ),
        (
            load_curves(args.val_top1_dir, args.smooth_window),
            "Validation Top-1 accuracy",
            "Accuracy",
        ),
        (
            load_curves(args.val_top5_dir, args.smooth_window),
            "Validation Top-5 accuracy",
            "Accuracy",
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (curves, title, ylabel) in zip(axes.ravel(), panels):
        plot_panel(ax, curves, title, ylabel)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(bottom=0.1)

    fig.savefig(output, dpi=300)
    fig.savefig(output.with_suffix(".pdf"))
    print(f"Wrote {output}")
    print(f"Wrote {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
