"""
Validate standardized view-aware metadata.

This checks required columns, allowed view labels, and optionally verifies that
view labels agree with path segments such as `/frontal/` or `/profile/`.

Example for CFP:
    python src/metadata/validate_view_metadata.py --metadata data/CFP/metadata/cfp_view_aware_metadata.csv --check_path_segments
"""

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "dataset",
    "image_id",
    "patient_id",
    "subject_id",
    "view",
    "split",
    "is_aligned",
    "is_paired",
    "pair_id",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Validate view-aware metadata.")
    parser.add_argument("--metadata", required=True, help="Metadata CSV to validate.")
    parser.add_argument(
        "--allowed_views",
        nargs="+",
        default=["frontal", "profile", "unknown", "exclude"],
        help="Allowed values in the view column.",
    )
    parser.add_argument("--view_col", default="view", help="View column name.")
    parser.add_argument("--source_path_col", default="source_path", help="Source path column name.")
    parser.add_argument(
        "--check_path_segments",
        action="store_true",
        help="Infer view from source_path segments and compare with view_col.",
    )
    return parser.parse_args()


def expected_view_from_source_path(source_path, allowed_views):
    normalized = f"/{str(source_path).replace('\\', '/').lower().strip('/')}/"
    for view in allowed_views:
        if view in {"unknown", "exclude"}:
            continue
        if f"/{view}/" in normalized:
            return view
    return "unknown"


def main():
    args = parse_args()
    metadata_path = Path(args.metadata)
    df = pd.read_csv(metadata_path)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if args.view_col not in df.columns:
        raise ValueError(f"View column not found: {args.view_col}")

    allowed_views = set(args.allowed_views)
    unknown_views = sorted(set(df[args.view_col]) - allowed_views)
    if unknown_views:
        raise ValueError(f"Unexpected view values: {unknown_views}")

    if args.check_path_segments:
        if args.source_path_col not in df.columns:
            raise ValueError(f"Source path column not found: {args.source_path_col}")
        df["expected_view"] = df[args.source_path_col].apply(
            lambda value: expected_view_from_source_path(value, allowed_views)
        )
        mismatches = df[df[args.view_col] != df["expected_view"]]
        if not mismatches.empty:
            print(mismatches[["image_id", args.source_path_col, args.view_col, "expected_view"]].head(20))
            raise ValueError(f"Found {len(mismatches)} view mapping mismatches.")

    print(f"Checked: {metadata_path}")
    print(f"Rows: {len(df)}")
    print(df.groupby([args.view_col, "is_paired"]).size().reset_index(name="rows"))
    print("View metadata is consistent.")


if __name__ == "__main__":
    main()
