"""
Create train_gm_arc.py-compatible CSVs from view-aware metadata.

Output columns are:
    image_id,subject,label

The image_id is the filename stem expected by GestaltMatcherDataset, which
loads images as <imgs_dir>/<image_id><img_postfix>.jpg.

Examples:
    python src/metadata/create_training_csv.py --metadata data/GMDB/view_metadata.csv --output data/GMDB/train_frontal.csv --splits train --views frontal --image_id_from_col aligned_filename --strip_suffix _aligned.jpg
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Create training CSV from view-aware metadata.")
    parser.add_argument("--metadata", required=True, help="Input view-aware metadata CSV.")
    parser.add_argument("--output", required=True, help="Output training CSV.")
    parser.add_argument("--splits", nargs="+", required=True, help="Split values to include.")
    parser.add_argument(
        "--views",
        nargs="*",
        default=[],
        help="Optional view values to include. Omit to include all views.",
    )
    parser.add_argument("--image_id_col", default="image_id", help="Metadata image_id column.")
    parser.add_argument(
        "--image_id_from_col",
        default="",
        help="Optional filename/path column used to derive image_id, e.g. aligned_filename.",
    )
    parser.add_argument(
        "--strip_suffix",
        default="",
        help="Optional suffix removed from image_id_from_col, e.g. _aligned.jpg.",
    )
    parser.add_argument("--subject_col", default="subject_id", help="Metadata subject column.")
    parser.add_argument("--label_col", default="label", help="Metadata label column.")
    parser.add_argument(
        "--label_allowlist_csv",
        default="",
        help="Optional CSV whose label values define the labels to keep, e.g. the matching train CSV.",
    )
    parser.add_argument(
        "--label_allowlist_col",
        default="label",
        help="Label column in --label_allowlist_csv.",
    )
    parser.add_argument("--split_col", default="split", help="Metadata split column.")
    parser.add_argument("--view_col", default="view", help="Metadata view column.")
    parser.add_argument(
        "--require_aligned",
        action="store_true",
        help="Keep only rows where is_aligned is true.",
    )
    parser.add_argument("--aligned_col", default="is_aligned", help="Metadata aligned-status column.")
    parser.add_argument(
        "--drop_missing_label",
        action="store_true",
        default=True,
        help="Drop rows with empty labels. Enabled by default.",
    )
    return parser.parse_args()


def derive_image_id(row, args):
    if args.image_id_from_col:
        value = Path(str(row[args.image_id_from_col])).name
        if args.strip_suffix and value.endswith(args.strip_suffix):
            return value[: -len(args.strip_suffix)]
        return Path(value).stem
    return row[args.image_id_col]


def main():
    args = parse_args()
    metadata_path = Path(args.metadata)
    df = pd.read_csv(metadata_path)

    required = {args.subject_col, args.label_col, args.split_col}
    required.add(args.image_id_from_col or args.image_id_col)
    if args.views:
        required.add(args.view_col)
    if args.require_aligned:
        required.add(args.aligned_col)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {metadata_path}: {sorted(missing)}")

    filtered = df[df[args.split_col].isin(args.splits)].copy()
    if args.views:
        filtered = filtered[filtered[args.view_col].isin(args.views)].copy()
    if args.require_aligned:
        aligned_values = filtered[args.aligned_col]
        if aligned_values.dtype == bool:
            filtered = filtered[aligned_values].copy()
        else:
            filtered = filtered[aligned_values.astype(str).str.lower().isin(["true", "1", "yes"])].copy()

    if args.drop_missing_label:
        filtered = filtered[filtered[args.label_col].notna()]
        filtered = filtered[filtered[args.label_col].astype(str).str.len() > 0]

    if args.label_allowlist_csv:
        allowlist = pd.read_csv(args.label_allowlist_csv)
        if args.label_allowlist_col not in allowlist.columns:
            raise ValueError(
                f"Label allowlist column '{args.label_allowlist_col}' not found in {args.label_allowlist_csv}"
            )
        allowed_labels = set(allowlist[args.label_allowlist_col].dropna().astype(str))
        filtered = filtered[filtered[args.label_col].astype(str).isin(allowed_labels)]

    if filtered.empty:
        raise ValueError("No metadata rows matched the requested filters.")

    output = pd.DataFrame(
        {
            "image_id": filtered.apply(lambda row: derive_image_id(row, args), axis=1),
            "subject": filtered[args.subject_col],
            "label": filtered[args.label_col],
        }
    )
    output = output.dropna(subset=["image_id", "subject", "label"])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    print(f"Metadata: {metadata_path}")
    print(f"Output: {output_path}")
    print(f"Rows: {len(output)}")
    print(f"Unique subjects: {output['subject'].nunique()}")
    print(f"Unique labels: {output['label'].nunique()}")


if __name__ == "__main__":
    main()
