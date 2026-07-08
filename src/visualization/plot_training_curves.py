"""
Plot training curves exported from TensorBoard scalar CSV files.

The expected CSV format is TensorBoard's scalar export:
    Wall time,Step,Value

Examples:
    python src/visualization/plot_training_curves.py \
      --input_dir data/GestaltMatcherDB/v1.1.3/results \
      --output_dir data/GestaltMatcherDB/v1.1.3/results/training_curves

    python src/visualization/plot_training_curves.py \
      --input_dir "data/GestaltMatcherDB/v1.1.3/results/validation-accuracy curves" \
      --output_dir data/GestaltMatcherDB/v1.1.3/results/training_curves \
      --ylabel "Validation cross-entropy loss" \
      --title "Validation loss curves" \
      --output_prefix validation_loss_curves \
      --individual_suffix validation_loss \
      --summary_name validation_loss_summary.csv
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_PATTERNS = {
    "Frontal-only": "*s101*view_aware_frontal*.csv",
    "Profile-only": "*s102*view_aware_profile*.csv",
    "Mixed-view": "*s103*view_aware_mixed*.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot TensorBoard scalar CSV training curves.")
    parser.add_argument("--input_dir", required=True, help="Directory containing exported scalar CSV files.")
    parser.add_argument("--output_dir", required=True, help="Directory where plots will be written.")
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=15,
        help="Centered rolling window for smoothed curves.",
    )
    parser.add_argument(
        "--ylabel",
        default="Cross-entropy loss",
        help="Y-axis label for the scalar value.",
    )
    parser.add_argument(
        "--title",
        default="Training loss curves",
        help="Title for combined plots.",
    )
    parser.add_argument(
        "--output_prefix",
        default="training_loss_curves",
        help="Filename prefix for combined plots.",
    )
    parser.add_argument(
        "--individual_suffix",
        default="training_loss",
        help="Filename suffix for individual model plots.",
    )
    parser.add_argument(
        "--summary_name",
        default="training_loss_summary.csv",
        help="Filename for the summary CSV.",
    )
    return parser.parse_args()


def find_one(input_dir, pattern):
    matches = sorted(input_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No CSV files matched pattern: {pattern}")
    if len(matches) > 1:
        names = "\n".join(str(path) for path in matches)
        raise ValueError(f"Pattern matched multiple files: {pattern}\n{names}")
    return matches[0]


def load_curve(path, smooth_window):
    df = pd.read_csv(path)
    required = {"Step", "Value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df = df.rename(columns={"Step": "step", "Value": "value", "Wall time": "wall_time"})
    df = df.sort_values("step").reset_index(drop=True)
    df["smooth_value"] = df["value"].rolling(
        window=smooth_window,
        min_periods=1,
        center=True,
    ).mean()
    return df


def safe_name(label):
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def save_combined(curves, output_dir, ylabel, title, output_prefix, log_scale=False):
    plt.figure(figsize=(8.4, 5.2))
    for label, df in curves.items():
        values = df["smooth_value"].clip(lower=1e-4) if log_scale else df["smooth_value"]
        plt.plot(df["step"], values, linewidth=2, label=label)

    plt.xlabel("Training step")
    plt.ylabel(f"{ylabel} (log scale)" if log_scale else ylabel)
    if log_scale:
        plt.yscale("log")
    plt.title(title)
    plt.grid(True, alpha=0.25, which="both" if log_scale else "major")
    plt.legend(frameon=False)
    plt.tight_layout()

    suffix = "_log" if log_scale else ""
    plt.savefig(output_dir / f"{output_prefix}{suffix}.png", dpi=300)
    plt.savefig(output_dir / f"{output_prefix}{suffix}.pdf")
    plt.close()


def save_individual(curves, output_dir, ylabel, individual_suffix):
    for label, df in curves.items():
        plt.figure(figsize=(7.2, 4.6))
        plt.plot(df["step"], df["value"], color="0.75", linewidth=0.9, label="Raw")
        plt.plot(df["step"], df["smooth_value"], linewidth=2, label="Smoothed")
        plt.xlabel("Training step")
        plt.ylabel(ylabel)
        plt.title(f"{label} training loss")
        plt.grid(True, alpha=0.25)
        plt.legend(frameon=False)
        plt.tight_layout()

        name = safe_name(label)
        plt.savefig(output_dir / f"{name}_{individual_suffix}.png", dpi=300)
        plt.savefig(output_dir / f"{name}_{individual_suffix}.pdf")
        plt.close()


def build_summary(curves):
    rows = []
    for label, df in curves.items():
        rows.append(
            {
                "model": label,
                "points": len(df),
                "first_step": int(df["step"].iloc[0]),
                "last_step": int(df["step"].iloc[-1]),
                "initial_value": float(df["value"].iloc[0]),
                "final_value": float(df["value"].iloc[-1]),
                "min_value": float(df["value"].min()),
                "max_value": float(df["value"].max()),
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    curves = {}
    for label, pattern in DEFAULT_PATTERNS.items():
        path = find_one(input_dir, pattern)
        curves[label] = load_curve(path, args.smooth_window)
        print(f"{label}: {path}")

    save_combined(curves, output_dir, args.ylabel, args.title, args.output_prefix, log_scale=False)
    save_combined(curves, output_dir, args.ylabel, args.title, args.output_prefix, log_scale=True)
    save_individual(curves, output_dir, args.ylabel, args.individual_suffix)
    build_summary(curves).to_csv(output_dir / args.summary_name, index=False)

    print(f"Wrote plots to: {output_dir}")


if __name__ == "__main__":
    main()
