"""
Split encoding tables into query/gallery files using view-aware metadata.

This utility keeps retrieval input preparation explicit: non-fusion encodings
are matched by aligned filename, while fusion encodings are matched by pair_id.
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Split encodings into query/gallery tables.")
    parser.add_argument("--encodings", required=True, help="Input encoding CSV/PKL.")
    parser.add_argument("--metadata", required=True, help="View-aware metadata CSV.")
    parser.add_argument("--query_output", required=True, help="Output query encoding PKL.")
    parser.add_argument("--gallery_output", required=True, help="Output gallery encoding PKL.")
    parser.add_argument("--query_splits", nargs="+", default=["test"], help="Metadata splits used as query.")
    parser.add_argument(
        "--gallery_splits",
        nargs="+",
        default=["train", "val"],
        help="Metadata splits used as gallery.",
    )
    parser.add_argument(
        "--views",
        nargs="*",
        default=[],
        help="Optional metadata view values to keep. Omit for all views.",
    )
    parser.add_argument("--metadata_key", default="aligned_filename", help="Metadata key for standard encodings.")
    parser.add_argument("--encoding_key", default="img_name", help="Encoding key for standard encodings.")
    parser.add_argument(
        "--fusion",
        action="store_true",
        help="Split fusion encodings by pair_id instead of image filename.",
    )
    parser.add_argument("--pair_col", default="pair_id", help="Pair column used with --fusion.")
    parser.add_argument(
        "--require_aligned",
        action="store_true",
        help="Keep only metadata rows where is_aligned is true.",
    )
    parser.add_argument("--aligned_col", default="is_aligned", help="Metadata aligned-status column.")
    return parser.parse_args()


def load_table(path):
    table_path = Path(path)
    if table_path.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(table_path)
    return pd.read_csv(table_path)


def save_pickle(df, path):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(output_path)


def truthy(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def filtered_metadata(metadata, args):
    required = {"split"}
    if args.views:
        required.add("view")
    if args.require_aligned:
        required.add(args.aligned_col)
    required.add(args.pair_col if args.fusion else args.metadata_key)
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

    filtered = metadata.copy()
    if args.views:
        filtered = filtered[filtered["view"].isin(args.views)].copy()
    if args.require_aligned:
        filtered = filtered[truthy(filtered[args.aligned_col])].copy()
    return filtered


def standard_split(encodings, metadata, splits, args):
    keys = set(metadata.loc[metadata["split"].isin(splits), args.metadata_key].dropna().astype(str))
    return encodings[encodings[args.encoding_key].astype(str).isin(keys)].copy()


def fusion_split(encodings, metadata, splits, args):
    keys = set(metadata.loc[metadata["split"].isin(splits), args.pair_col].dropna().astype(str))
    return encodings[encodings[args.pair_col].astype(str).isin(keys)].copy()


def main():
    args = parse_args()
    encodings = load_table(args.encodings)
    metadata = filtered_metadata(pd.read_csv(args.metadata), args)

    if args.fusion:
        if args.pair_col not in encodings.columns:
            raise ValueError(f"Fusion encodings are missing pair column: {args.pair_col}")
        query = fusion_split(encodings, metadata, args.query_splits, args)
        gallery = fusion_split(encodings, metadata, args.gallery_splits, args)
    else:
        if args.encoding_key not in encodings.columns:
            raise ValueError(f"Encodings are missing key column: {args.encoding_key}")
        query = standard_split(encodings, metadata, args.query_splits, args)
        gallery = standard_split(encodings, metadata, args.gallery_splits, args)

    if query.empty:
        raise ValueError("Query split produced no encoding rows.")
    if gallery.empty:
        raise ValueError("Gallery split produced no encoding rows.")

    save_pickle(query, args.query_output)
    save_pickle(gallery, args.gallery_output)

    print(f"Input encodings: {args.encodings}")
    print(f"Query output: {args.query_output}")
    print(f"Gallery output: {args.gallery_output}")
    print(f"Query rows: {len(query)}")
    print(f"Gallery rows: {len(gallery)}")
    if "img_name" in query.columns:
        print(f"Query unique img_name: {query['img_name'].nunique()}")
        print(f"Gallery unique img_name: {gallery['img_name'].nunique()}")
    if args.fusion:
        print(f"Query unique pairs: {query[args.pair_col].nunique()}")
        print(f"Gallery unique pairs: {gallery[args.pair_col].nunique()}")


if __name__ == "__main__":
    main()
