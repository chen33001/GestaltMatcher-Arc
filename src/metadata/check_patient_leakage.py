"""
Check patient-level leakage across dataset splits.

The script validates that each patient_id appears in at most one split. It is
intended for standardized view-aware metadata before training CSVs are derived.

Example:
    python src/metadata/check_patient_leakage.py --metadata data/GMDB/view_metadata.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Check patient-level split leakage.")
    parser.add_argument("--metadata", required=True, help="Input metadata CSV.")
    parser.add_argument("--patient_col", default="patient_id", help="Patient identifier column.")
    parser.add_argument("--split_col", default="split", help="Split column.")
    parser.add_argument(
        "--ignore_splits",
        nargs="*",
        default=[],
        help="Optional split values to ignore, e.g. exclude.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metadata_path = Path(args.metadata)
    df = pd.read_csv(metadata_path)

    for column in [args.patient_col, args.split_col]:
        if column not in df.columns:
            raise ValueError(f"Missing required column '{column}' in {metadata_path}")

    checked = df.copy()
    if args.ignore_splits:
        checked = checked[~checked[args.split_col].isin(args.ignore_splits)]

    split_sets = checked.groupby(args.patient_col)[args.split_col].agg(lambda values: sorted(set(values)))
    leaks = split_sets[split_sets.apply(len) > 1]

    print(f"Metadata: {metadata_path}")
    print(f"Rows checked: {len(checked)}")
    print(f"Patients checked: {checked[args.patient_col].nunique()}")
    print(checked.groupby(args.split_col).size().reset_index(name="rows"))

    if not leaks.empty:
        leak_df = leaks.reset_index(name="splits")
        print(leak_df.head(50).to_string(index=False))
        raise ValueError(f"Found {len(leaks)} patients appearing in multiple splits.")

    print("No patient-level leakage detected.")


if __name__ == "__main__":
    main()
