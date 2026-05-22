"""
Create late-fusion embeddings by concatenating paired frontal/profile vectors.

The output is compatible with src/retrieval/run_retrieval.py:
    img_name, model, flip, gray, class_conf, representations

Examples:
    python src/retrieval/fuse_paired_embeddings.py --frontal data/encodings/frontal.pkl --profile data/encodings/profile.pkl --metadata data/view_metadata.csv --metadata_key aligned_filename --pair_col pair_id --output data/encodings/fused.pkl
"""

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Fuse paired frontal/profile embeddings.")
    parser.add_argument("--frontal", required=True, help="Frontal encoding CSV/PKL.")
    parser.add_argument("--profile", required=True, help="Profile encoding CSV/PKL.")
    parser.add_argument("--metadata", required=True, help="View-aware metadata CSV.")
    parser.add_argument("--output", required=True, help="Output fused encoding PKL.")
    parser.add_argument("--encoding_key", default="img_name", help="Encoding filename key.")
    parser.add_argument("--metadata_key", default="aligned_filename", help="Metadata filename key.")
    parser.add_argument("--pair_col", default="pair_id", help="Pair identifier column.")
    parser.add_argument("--view_col", default="view", help="View column.")
    parser.add_argument("--frontal_view", default="frontal", help="Frontal view label.")
    parser.add_argument("--profile_view", default="profile", help="Profile view label.")
    parser.add_argument("--model", default="fusion_concat", help="Model label for fused rows.")
    return parser.parse_args()


def load_table(path):
    table_path = Path(path)
    if table_path.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(table_path)
    return pd.read_csv(table_path, delimiter=";" if table_path.suffix.lower() == ".csv" else ",")


def parse_representation(value):
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float32)
    if isinstance(value, str):
        try:
            return np.asarray(json.loads(value), dtype=np.float32)
        except json.JSONDecodeError:
            return np.asarray(ast.literal_eval(value), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def with_metadata(encodings, metadata, args, expected_view):
    required_encoding = {args.encoding_key, "model", "flip", "gray", "representations"}
    missing_encoding = required_encoding - set(encodings.columns)
    if missing_encoding:
        raise ValueError(f"Encoding file is missing columns: {sorted(missing_encoding)}")

    required_metadata = {args.metadata_key, args.pair_col, args.view_col}
    missing_metadata = required_metadata - set(metadata.columns)
    if missing_metadata:
        raise ValueError(f"Metadata file is missing columns: {sorted(missing_metadata)}")

    subset = metadata[metadata[args.view_col] == expected_view].copy()
    merged = encodings.merge(
        subset[[args.metadata_key, args.pair_col]],
        how="inner",
        left_on=args.encoding_key,
        right_on=args.metadata_key,
    )
    merged["representations"] = merged["representations"].apply(parse_representation)
    return merged


def main():
    args = parse_args()
    frontal = load_table(args.frontal)
    profile = load_table(args.profile)
    metadata = pd.read_csv(args.metadata)

    frontal = with_metadata(frontal, metadata, args, args.frontal_view)
    profile = with_metadata(profile, metadata, args, args.profile_view)

    paired = frontal.merge(
        profile,
        how="inner",
        on=[args.pair_col, "flip", "gray"],
        suffixes=("_frontal", "_profile"),
    )
    if paired.empty:
        raise ValueError("No paired frontal/profile embeddings matched by pair, flip, and gray.")

    rows = []
    for row in paired.itertuples(index=False):
        frontal_vec = getattr(row, "representations_frontal")
        profile_vec = getattr(row, "representations_profile")
        fused = np.concatenate([frontal_vec, profile_vec]).astype(np.float32)
        pair_id = getattr(row, args.pair_col)
        rows.append(
            {
                "img_name": f"{pair_id}_fusion",
                "model": args.model,
                "flip": int(getattr(row, "flip")),
                "gray": int(getattr(row, "gray")),
                "class_conf": "",
                "representations": fused.tolist(),
                "pair_id": pair_id,
                "frontal_img_name": getattr(row, f"{args.encoding_key}_frontal"),
                "profile_img_name": getattr(row, f"{args.encoding_key}_profile"),
                "frontal_model": getattr(row, "model_frontal"),
                "profile_model": getattr(row, "model_profile"),
            }
        )

    output = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_pickle(output_path)

    print(f"Frontal rows with metadata: {len(frontal)}")
    print(f"Profile rows with metadata: {len(profile)}")
    print(f"Fused paired rows: {len(output)}")
    print(f"Output: {output_path}")
    print(f"Fused representation length: {len(output.iloc[0]['representations'])}")


if __name__ == "__main__":
    main()
