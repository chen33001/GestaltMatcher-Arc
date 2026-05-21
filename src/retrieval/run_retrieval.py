"""
Run generic nearest-neighbor retrieval between query and gallery encodings.

The script is dataset-agnostic: it only requires encoding files with
`img_name`, `model`, `flip`, `gray`, and `representations` columns. Optional
query/gallery metadata tables can be joined by configurable keys.

Examples:
    python src/retrieval/run_retrieval.py --query data/CFP/encodings/cfp_m1_s2_glint360_r100_flip0_gray0.pkl --gallery data/gallery_encodings/GMDB_gallery_encodings_v1.1.2_m1_flip0_gray0.pkl --query_metadata data/CFP/metadata/cfp_aligned_manifest.csv --query_metadata_key aligned_filename --output data/CFP/results/cfp_m1_rankings.csv --top_k 30
"""

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"img_name", "model", "flip", "gray", "representations"}
SETTING_COLUMNS = ["model", "flip", "gray"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run generic cosine retrieval.")
    parser.add_argument("--query", required=True, help="Query encoding CSV/PKL.")
    parser.add_argument("--gallery", required=True, help="Gallery encoding CSV/PKL.")
    parser.add_argument("--output", required=True, help="Output ranking CSV.")
    parser.add_argument("--top_k", type=int, default=30, help="Neighbors per query.")
    parser.add_argument("--chunk_size", type=int, default=128, help="Query batch size.")
    parser.add_argument("--model", default="", help="Optional model label filter, e.g. m1.")
    parser.add_argument("--flip", type=int, choices=[0, 1], default=None, help="Optional flip TTA filter.")
    parser.add_argument("--gray", type=int, choices=[0, 1], default=None, help="Optional gray TTA filter.")
    parser.add_argument(
        "--allow_mismatched_settings",
        action="store_true",
        help="Allow query/gallery model, flip, gray combinations to differ.",
    )
    parser.add_argument("--query_metadata", default="", help="Optional query metadata CSV/PKL.")
    parser.add_argument("--query_key", default="img_name", help="Query encoding join key.")
    parser.add_argument(
        "--query_metadata_key",
        default="img_name",
        help="Query metadata join key.",
    )
    parser.add_argument(
        "--query_metadata_prefix",
        default="query_",
        help="Prefix added to query metadata columns in the output.",
    )
    parser.add_argument("--gallery_metadata", default="", help="Optional gallery metadata CSV/PKL.")
    parser.add_argument("--gallery_key", default="img_name", help="Gallery encoding join key.")
    parser.add_argument(
        "--gallery_metadata_key",
        default="img_name",
        help="Gallery metadata join key.",
    )
    parser.add_argument(
        "--gallery_metadata_prefix",
        default="gallery_",
        help="Prefix added to gallery metadata columns in the output.",
    )
    return parser.parse_args()


def load_table(path):
    table_path = Path(path)
    if table_path.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(table_path)
    return pd.read_csv(table_path)


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


def load_encodings(path):
    df = load_table(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df = df.copy()
    df["representations"] = df["representations"].apply(parse_representation)
    return df


def apply_filters(df, args, label):
    filtered = df
    if args.model:
        filtered = filtered[filtered["model"] == args.model]
    if args.flip is not None:
        filtered = filtered[filtered["flip"] == args.flip]
    if args.gray is not None:
        filtered = filtered[filtered["gray"] == args.gray]
    filtered = filtered.copy()
    if filtered.empty:
        available = df.groupby(SETTING_COLUMNS).size().reset_index(name="rows")
        raise ValueError(
            f"No {label} rows matched the requested filters.\n"
            f"Available combinations:\n{available.to_string(index=False)}"
        )
    return filtered


def setting_tuples(df):
    return set(tuple(row) for row in df[SETTING_COLUMNS].drop_duplicates().to_numpy())


def validate_settings(query_df, gallery_df, allow_mismatched):
    if allow_mismatched:
        return
    query_settings = setting_tuples(query_df)
    gallery_settings = setting_tuples(gallery_df)
    if len(query_settings) != 1 or len(gallery_settings) != 1:
        raise ValueError(
            "Query and gallery must each contain exactly one model/flip/gray "
            "combination. Use --model/--flip/--gray filters or pass "
            "--allow_mismatched_settings."
        )
    if query_settings != gallery_settings:
        raise ValueError(
            f"Query settings {sorted(query_settings)} do not match gallery "
            f"settings {sorted(gallery_settings)}."
        )


def l2_normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


def vector_matrix(df, label):
    lengths = df["representations"].apply(len).unique()
    if len(lengths) != 1:
        raise ValueError(f"{label} representations have inconsistent lengths: {sorted(lengths)}")
    return np.stack(df["representations"].values).astype(np.float32)


def prepare_metadata(path, metadata_key, output_key, prefix):
    if not path:
        return None
    metadata = load_table(path).copy()
    if metadata_key not in metadata.columns:
        raise ValueError(f"Metadata key '{metadata_key}' not found in {path}")

    renamed = {}
    for column in metadata.columns:
        if column == metadata_key:
            renamed[column] = output_key
        elif not column.startswith(prefix):
            renamed[column] = f"{prefix}{column}"
    metadata = metadata.rename(columns=renamed)
    return metadata


def add_metadata(rankings, metadata, key):
    if metadata is None:
        return rankings
    return rankings.merge(metadata, how="left", on=key)


def output_join_key(side, encoding_key):
    if encoding_key == "img_name":
        return f"{side}_img_name"
    return f"{side}_{encoding_key}"


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    query_df = apply_filters(load_encodings(args.query), args, "query")
    gallery_df = apply_filters(load_encodings(args.gallery), args, "gallery")
    if args.query_key not in query_df.columns:
        raise ValueError(f"Query key '{args.query_key}' not found in query encodings.")
    if args.gallery_key not in gallery_df.columns:
        raise ValueError(f"Gallery key '{args.gallery_key}' not found in gallery encodings.")
    validate_settings(query_df, gallery_df, args.allow_mismatched_settings)

    query_vectors = l2_normalize(vector_matrix(query_df, "Query"))
    gallery_vectors = l2_normalize(vector_matrix(gallery_df, "Gallery"))
    top_k = min(args.top_k, len(gallery_df))
    query_output_key = output_join_key("query", args.query_key)
    gallery_output_key = output_join_key("gallery", args.gallery_key)

    rows = []
    print(f"Query rows: {len(query_df)} from {args.query}")
    print(f"Gallery rows: {len(gallery_df)} from {args.gallery}")
    print(f"Top-k: {top_k}")

    for start in range(0, len(query_df), args.chunk_size):
        end = min(start + args.chunk_size, len(query_df))
        similarities = query_vectors[start:end] @ gallery_vectors.T
        distances = 1 - similarities
        top_indices = np.argpartition(distances, kth=top_k - 1, axis=1)[:, :top_k]

        for local_idx, candidate_indices in enumerate(top_indices):
            query_index = start + local_idx
            ordered = candidate_indices[np.argsort(distances[local_idx, candidate_indices])]
            query_row = query_df.iloc[query_index]

            for rank, gallery_index in enumerate(ordered, start=1):
                gallery_row = gallery_df.iloc[gallery_index]
                row = {
                    "query_img_name": query_row["img_name"],
                    "query_model": query_row["model"],
                    "query_flip": int(query_row["flip"]),
                    "query_gray": int(query_row["gray"]),
                    "rank": rank,
                    "distance": float(distances[local_idx, gallery_index]),
                    "gallery_img_name": gallery_row["img_name"],
                    "gallery_model": gallery_row["model"],
                    "gallery_flip": int(gallery_row["flip"]),
                    "gallery_gray": int(gallery_row["gray"]),
                }
                row.setdefault(query_output_key, query_row[args.query_key])
                row.setdefault(gallery_output_key, gallery_row[args.gallery_key])
                rows.append(row)

        print(f"Processed {end}/{len(query_df)} queries")

    rankings = pd.DataFrame(rows)
    query_metadata = prepare_metadata(
        args.query_metadata,
        args.query_metadata_key,
        query_output_key,
        args.query_metadata_prefix,
    )
    gallery_metadata = prepare_metadata(
        args.gallery_metadata,
        args.gallery_metadata_key,
        gallery_output_key,
        args.gallery_metadata_prefix,
    )
    rankings = add_metadata(rankings, query_metadata, query_output_key)
    rankings = add_metadata(rankings, gallery_metadata, gallery_output_key)
    rankings.to_csv(output_path, index=False)

    print(f"Saved rankings: {output_path}")
    print(f"Rows: {len(rankings)}")


if __name__ == "__main__":
    main()
