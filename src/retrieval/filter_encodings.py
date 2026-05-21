"""
Filter an encoding table to one model/TTA combination.

Use this before retrieval when query and gallery encodings should match on
model, flip, and gray.

Examples:
    python src/retrieval/filter_encodings.py --input data/gallery_encodings/GMDB_gallery_encodings_v1.1.2_service.pkl --model m1 --flip 0 --gray 0 --output data/gallery_encodings/GMDB_gallery_encodings_v1.1.2_m1_flip0_gray0.pkl
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Filter encodings by model and TTA fields.")
    parser.add_argument("--input", required=True, help="Input encoding CSV/PKL.")
    parser.add_argument("--model", required=True, help="Model label to keep, e.g. m1 or m2.")
    parser.add_argument("--flip", type=int, default=0, choices=[0, 1], help="Flip TTA value to keep.")
    parser.add_argument("--gray", type=int, default=0, choices=[0, 1], help="Gray TTA value to keep.")
    parser.add_argument(
        "--output",
        default="",
        help="Output path. Defaults to input stem plus model/flip/gray suffix.",
    )
    return parser.parse_args()


def load_table(path):
    input_path = Path(path)
    if input_path.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(input_path)
    return pd.read_csv(input_path, delimiter=";" if input_path.suffix.lower() == ".csv" else ",")


def save_table(df, path):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in {".pkl", ".pickle"}:
        df.to_pickle(output_path)
    else:
        df.to_csv(output_path, index=False)


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(f"{input_path.stem}_{args.model}_flip{args.flip}_gray{args.gray}{input_path.suffix}")
    )

    df = load_table(input_path)
    required_columns = {"img_name", "model", "flip", "gray", "representations"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    filtered = df[
        (df["model"] == args.model)
        & (df["flip"] == args.flip)
        & (df["gray"] == args.gray)
    ].copy()

    if filtered.empty:
        available = df.groupby(["model", "flip", "gray"]).size().reset_index(name="rows")
        raise ValueError(
            f"No rows matched model={args.model}, flip={args.flip}, gray={args.gray}.\n"
            f"Available combinations:\n{available.to_string(index=False)}"
        )

    save_table(filtered, output_path)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Rows: {len(filtered)}")
    print(f"Unique images: {filtered['img_name'].nunique()}")
    print(filtered.groupby(["model", "flip", "gray"]).size().reset_index(name="rows"))
    first_vector = filtered.iloc[0]["representations"]
    print(f"Representation length: {len(first_vector)}")


if __name__ == "__main__":
    main()
