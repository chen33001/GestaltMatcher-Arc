"""
Evaluate GMDB encodings with the frequent/rare protocol.

Modes:
    image: evaluate image-level encodings with img_name/image_id rows.
    pair: evaluate pair-level fusion encodings mapped through view metadata.
"""

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TOP_K = [1, 5, 10, 30]
PROTOCOLS = [
    "frequent_test_vs_frequent_gallery",
    "rare_test_vs_rare_gallery",
    "frequent_test_vs_frequent_plus_rare_gallery",
    "rare_test_vs_frequent_plus_rare_gallery",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GMDB frequent/rare retrieval.")
    parser.add_argument("--mode", required=True, choices=["image", "pair"], help="Evaluation mode.")
    parser.add_argument("--metadata_dir", required=True, help="Directory containing GMDB frequent/rare metadata CSVs.")
    parser.add_argument("--version", default="v1.1.3", help="GMDB metadata version.")
    parser.add_argument("--output", required=True, help="Markdown summary output.")
    parser.add_argument("--top_k", nargs="*", type=int, default=DEFAULT_TOP_K, help="Top-k values.")
    parser.add_argument("--model", default="", help="Optional model filter.")
    parser.add_argument("--flip", type=int, choices=[0, 1], default=None, help="Optional flip filter.")
    parser.add_argument("--gray", type=int, choices=[0, 1], default=None, help="Optional gray filter.")
    parser.add_argument(
        "--allow_self_match",
        action="store_true",
        help="Do not remove identical query/gallery IDs when both sets contain the same item.",
    )

    image_group = parser.add_argument_group("image mode")
    image_group.add_argument("--encodings", default="", help="Image encoding CSV/PKL with img_name and representations.")
    image_group.add_argument(
        "--strict_member_count",
        action="store_true",
        help="Require every image to have the same number of representation rows after filtering.",
    )

    pair_group = parser.add_argument_group("pair mode")
    pair_group.add_argument("--fused_encodings", default="", help="Fused encoding CSV/PKL containing pair_id.")
    pair_group.add_argument(
        "--view_metadata",
        default="",
        help="View-aware metadata CSV containing image_id and pair_id.",
    )
    pair_group.add_argument("--pair_col", default="pair_id", help="Pair identifier column.")
    pair_group.add_argument("--label_col", default="label", help="Label column.")
    pair_group.add_argument(
        "--pair_distance",
        choices=["mean", "min"],
        default="mean",
        help="How to collapse multiple fused rows for the same query/gallery pair.",
    )

    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args):
    if args.mode == "image" and not args.encodings:
        raise ValueError("--encodings is required when --mode image.")
    if args.mode == "pair":
        missing = []
        if not args.fused_encodings:
            missing.append("--fused_encodings")
        if not args.view_metadata:
            missing.append("--view_metadata")
        if missing:
            raise ValueError(f"{', '.join(missing)} required when --mode pair.")


def canonical_id(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        return text[:-2]
    return text


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


def truthy(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def apply_encoding_filters(df, args, empty_message):
    filtered = df
    if args.model and "model" in filtered.columns:
        filtered = filtered[filtered["model"] == args.model]
    if args.flip is not None and "flip" in filtered.columns:
        filtered = filtered[filtered["flip"] == args.flip]
    if args.gray is not None and "gray" in filtered.columns:
        filtered = filtered[filtered["gray"] == args.gray]
    filtered = filtered.copy()
    if filtered.empty:
        raise ValueError(empty_message)
    return filtered


def image_id_from_img_name(value):
    name = Path(str(value)).stem
    if name.endswith("_aligned"):
        name = name[: -len("_aligned")]
    return name


def load_image_encodings(path, args):
    df = load_table(path).copy()
    missing = {"img_name", "representations"} - set(df.columns)
    if missing:
        raise ValueError(f"Encodings are missing required columns: {sorted(missing)}")

    df = apply_encoding_filters(df, args, "No encoding rows remain after filters.")
    df["image_id"] = df["img_name"].apply(image_id_from_img_name)
    df["representations"] = df["representations"].apply(parse_representation)

    grouped_rows = []
    for image_id, group in df.groupby("image_id", sort=False):
        vectors = [np.asarray(vector, dtype=np.float32) for vector in group["representations"]]
        grouped_rows.append({"image_id": str(image_id), "representations": vectors, "members": len(vectors)})

    encodings = pd.DataFrame(grouped_rows).set_index("image_id")
    member_counts = sorted(encodings["members"].unique().tolist())
    if args.strict_member_count and len(member_counts) != 1:
        raise ValueError(f"Images have inconsistent representation counts: {member_counts}")
    if len(member_counts) != 1:
        print(f"Warning: images have inconsistent representation counts: {member_counts}. Missing members are ignored.")
    return encodings


def load_pair_encodings(path, args):
    df = load_table(path).copy()
    missing = {args.pair_col, "representations"} - set(df.columns)
    if missing:
        raise ValueError(f"Fused encodings are missing required columns: {sorted(missing)}")

    df = apply_encoding_filters(df, args, "No fused encoding rows remain after filters.")
    df["pair_key"] = df[args.pair_col].apply(canonical_id)
    df["representations"] = df["representations"].apply(parse_representation)
    grouped = df.groupby("pair_key", sort=False)["representations"].apply(list).reset_index()
    return grouped.set_index("pair_key")


def metadata_path(metadata_dir, name, version):
    return Path(metadata_dir) / f"gmdb_{name}_images_{version}.csv"


def load_protocol_metadata(metadata_dir, version):
    return {
        "freq_gallery": pd.read_csv(metadata_path(metadata_dir, "frequent_gallery", version)),
        "freq_test": pd.read_csv(metadata_path(metadata_dir, "frequent_test", version)),
        "rare_gallery": pd.read_csv(metadata_path(metadata_dir, "rare_gallery", version)),
        "rare_test": pd.read_csv(metadata_path(metadata_dir, "rare_test", version)),
    }


def attach_image_vectors(metadata, encodings, name):
    data = metadata.copy()
    if "image_id" not in data.columns or "label" not in data.columns:
        raise ValueError(f"{name} metadata must contain image_id and label columns.")

    data["image_id"] = data["image_id"].astype(str)
    available = data["image_id"].isin(encodings.index)
    missing = data.loc[~available, "image_id"].drop_duplicates().tolist()
    if missing:
        print(f"Warning: {name} drops {len(missing)} image IDs without encodings. First missing: {missing[:5]}")

    data = data[available].copy()
    data["representations"] = data["image_id"].map(encodings["representations"])
    return data.reset_index(drop=True)


def image_to_pair_map(view_metadata, args):
    required = {"image_id", args.pair_col, args.label_col}
    missing = required - set(view_metadata.columns)
    if missing:
        raise ValueError(f"View metadata is missing required columns: {sorted(missing)}")

    data = view_metadata.copy()
    if "is_aligned" in data.columns:
        data = data[truthy(data["is_aligned"])].copy()
    data = data[data[args.pair_col].notna()].copy()
    data = data[data[args.pair_col].astype(str) != ""].copy()
    data["image_key"] = data["image_id"].apply(canonical_id)
    data["pair_key"] = data[args.pair_col].apply(canonical_id)
    return data[["image_key", "pair_key", args.label_col]].drop_duplicates()


def build_pair_protocol(protocol_df, pair_map, encodings, name, args):
    required = {"image_id", args.label_col}
    missing = required - set(protocol_df.columns)
    if missing:
        raise ValueError(f"{name} metadata is missing required columns: {sorted(missing)}")

    protocol = protocol_df.copy()
    protocol["image_key"] = protocol["image_id"].apply(canonical_id)
    protocol = protocol.merge(pair_map, how="left", on="image_key", suffixes=("", "_view"))

    unmapped = protocol["pair_key"].isna().sum()
    if unmapped:
        print(f"Warning: {name} drops {unmapped} image rows without pair_id mapping.")
    protocol = protocol[protocol["pair_key"].notna()].copy()

    unavailable = ~protocol["pair_key"].isin(encodings.index)
    if unavailable.any():
        print(f"Warning: {name} drops {int(unavailable.sum())} image rows without fused encodings.")
    protocol = protocol[~unavailable].copy()

    group_cols = ["pair_key"]
    if "split" in protocol.columns:
        group_cols.append("split")

    rows = []
    for group_key, group in protocol.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        label_values = sorted(set(group[args.label_col].dropna().astype(str)))
        if len(label_values) != 1:
            raise ValueError(f"{name} pair {group_key[0]} maps to multiple labels: {label_values}")
        row = {
            "pair_key": canonical_id(group_key[0]),
            args.label_col: label_values[0],
            "source_images": int(group["image_key"].nunique()),
            "representations": encodings.loc[canonical_id(group_key[0]), "representations"],
        }
        if "split" in group_cols:
            row["split"] = int(group_key[1])
        rows.append(row)

    columns = ["pair_key", args.label_col, "source_images", "representations"]
    if "split" in protocol_df.columns:
        columns.append("split")
    return pd.DataFrame(rows, columns=columns)


def l2_normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


def cosine_distance_matrix(query_vectors, gallery_vectors):
    query_vectors = l2_normalize(query_vectors.astype(np.float32))
    gallery_vectors = l2_normalize(gallery_vectors.astype(np.float32))
    return 1 - (query_vectors @ gallery_vectors.T)


def average_member_distances(query_df, gallery_df):
    distances = []
    max_query_members = max(len(vectors) for vectors in query_df["representations"])
    max_gallery_members = max(len(vectors) for vectors in gallery_df["representations"])
    member_count = min(max_query_members, max_gallery_members)

    for member_index in range(member_count):
        query_mask = query_df["representations"].apply(lambda vectors: len(vectors) > member_index).to_numpy()
        gallery_mask = gallery_df["representations"].apply(lambda vectors: len(vectors) > member_index).to_numpy()
        if not query_mask.all() or not gallery_mask.all():
            continue
        query_vectors = np.stack(query_df["representations"].apply(lambda vectors: vectors[member_index]).values)
        gallery_vectors = np.stack(gallery_df["representations"].apply(lambda vectors: vectors[member_index]).values)
        distances.append(cosine_distance_matrix(query_vectors, gallery_vectors))

    if not distances:
        raise ValueError("No shared representation members were available for this evaluation split.")
    return np.mean(np.stack(distances, axis=0), axis=0)


def pair_distance(query_vectors, gallery_vectors, mode):
    query = l2_normalize(np.stack(query_vectors).astype(np.float32))
    gallery = l2_normalize(np.stack(gallery_vectors).astype(np.float32))
    distances = 1 - (query @ gallery.T)
    if mode == "min":
        return float(distances.min())
    return float(distances.mean())


def pair_distances(query_df, gallery_df, allow_self_match, distance_mode):
    distances = np.empty((len(query_df), len(gallery_df)), dtype=np.float32)
    for query_index, query_row in enumerate(query_df.itertuples(index=False)):
        query_vectors = getattr(query_row, "representations")
        query_pair = getattr(query_row, "pair_key")
        for gallery_index, gallery_row in enumerate(gallery_df.itertuples(index=False)):
            gallery_pair = getattr(gallery_row, "pair_key")
            if not allow_self_match and query_pair == gallery_pair:
                distances[query_index, gallery_index] = np.inf
                continue
            gallery_vectors = getattr(gallery_row, "representations")
            distances[query_index, gallery_index] = pair_distance(query_vectors, gallery_vectors, distance_mode)
    return distances


def image_distances(query_df, gallery_df, allow_self_match):
    distances = average_member_distances(query_df, gallery_df)
    if not allow_self_match:
        gallery_ids = gallery_df["image_id"].astype(str).to_numpy()
        for query_index, query_id in enumerate(query_df["image_id"].astype(str).to_numpy()):
            distances[query_index, gallery_ids == query_id] = np.inf
    return distances


def unique_labels_by_rank(labels):
    seen = set()
    unique = []
    for label in labels:
        if label in seen:
            continue
        seen.add(label)
        unique.append(label)
    return unique


def empty_metrics(top_k_values, query_count, gallery_count, label_count):
    metrics = {"queries": int(query_count), "gallery": int(gallery_count), "labels": int(label_count)}
    for top_k in top_k_values:
        metrics[f"micro_top_{top_k}"] = np.nan
        metrics[f"macro_top_{top_k}"] = np.nan
    return metrics


def evaluate_split(query_df, gallery_df, top_k_values, distance_fn, label_col):
    if query_df.empty or gallery_df.empty:
        return empty_metrics(top_k_values, len(query_df), len(gallery_df), 0)

    distances = distance_fn(query_df, gallery_df)
    order = np.argsort(distances, axis=1)
    gallery_labels = gallery_df[label_col].to_numpy()
    query_labels = query_df[label_col].to_numpy()
    ranked_unique_labels = [
        unique_labels_by_rank(gallery_labels[order[query_index]].tolist()) for query_index in range(len(query_df))
    ]

    rows = []
    for query_index, query_label in enumerate(query_labels):
        row = {"label": query_label}
        for top_k in top_k_values:
            row[f"hit_top_{top_k}"] = query_label in ranked_unique_labels[query_index][:top_k]
        rows.append(row)
    hit_df = pd.DataFrame(rows)

    metrics = {
        "queries": int(len(query_df)),
        "gallery": int(len(gallery_df)),
        "labels": int(hit_df["label"].nunique()),
    }
    for top_k in top_k_values:
        hit_col = f"hit_top_{top_k}"
        metrics[f"micro_top_{top_k}"] = float(hit_df[hit_col].mean())
        metrics[f"macro_top_{top_k}"] = float(hit_df.groupby("label")[hit_col].mean().mean())
    return metrics


def split_values(*frames):
    values = set()
    for frame in frames:
        if "split" in frame.columns:
            values.update(frame["split"].dropna().astype(int).tolist())
    return sorted(values)


def evaluate_protocol(name, query_df, gallery_df, top_k_values, distance_fn, label_col, splits=None):
    if splits is None:
        metrics = evaluate_split(query_df, gallery_df, top_k_values, distance_fn, label_col)
        return [{"protocol": name, "split": "all", **metrics}]

    rows = []
    for split in splits:
        query_split = query_df if "split" not in query_df.columns else query_df[query_df["split"] == split]
        if "split" not in gallery_df.columns:
            gallery_split = gallery_df
        else:
            split_series = gallery_df["split"]
            gallery_split = gallery_df[(split_series == split) | split_series.isna()]
        metrics = evaluate_split(query_split, gallery_split, top_k_values, distance_fn, label_col)
        rows.append({"protocol": name, "split": int(split), **metrics})
    return rows


def aggregate_rows(rows, top_k_values):
    details = pd.DataFrame(rows)
    summaries = []
    for protocol, group in details.groupby("protocol", sort=False):
        summary = {
            "protocol": protocol,
            "splits": int(len(group)),
            "queries": float(group["queries"].mean()),
            "gallery": float(group["gallery"].mean()),
            "labels": float(group["labels"].mean()),
        }
        for top_k in top_k_values:
            summary[f"macro_top_{top_k}"] = float(group[f"macro_top_{top_k}"].mean())
            summary[f"micro_top_{top_k}"] = float(group[f"micro_top_{top_k}"].mean())
        summaries.append(summary)
    return pd.DataFrame(summaries), details


def markdown_table(df):
    formatted = df.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    formatted = formatted.fillna("").astype(str)
    headers = list(formatted.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in formatted.iterrows():
        lines.append("| " + " | ".join(row[column] for column in headers) + " |")
    return "\n".join(lines)


def build_image_frames(args, metadata):
    encodings = load_image_encodings(args.encodings, args)
    return {
        "freq_gallery": attach_image_vectors(metadata["freq_gallery"], encodings, "frequent gallery"),
        "freq_test": attach_image_vectors(metadata["freq_test"], encodings, "frequent test"),
        "rare_gallery": attach_image_vectors(metadata["rare_gallery"], encodings, "rare gallery"),
        "rare_test": attach_image_vectors(metadata["rare_test"], encodings, "rare test"),
    }


def build_pair_frames(args, metadata):
    encodings = load_pair_encodings(args.fused_encodings, args)
    pair_map = image_to_pair_map(pd.read_csv(args.view_metadata), args)
    return {
        "freq_gallery": build_pair_protocol(metadata["freq_gallery"], pair_map, encodings, "frequent gallery", args),
        "freq_test": build_pair_protocol(metadata["freq_test"], pair_map, encodings, "frequent test", args),
        "rare_gallery": build_pair_protocol(metadata["rare_gallery"], pair_map, encodings, "rare gallery", args),
        "rare_test": build_pair_protocol(metadata["rare_test"], pair_map, encodings, "rare test", args),
    }


def evaluate_all_protocols(frames, top_k_values, distance_fn, label_col, rare_splits, pair_mode=False):
    all_gallery = pd.concat([frames["freq_gallery"], frames["rare_gallery"]], ignore_index=True)

    rows = []
    rows.extend(
        evaluate_protocol(
            PROTOCOLS[0],
            frames["freq_test"],
            frames["freq_gallery"],
            top_k_values,
            distance_fn,
            label_col,
        )
    )
    rows.extend(
        evaluate_protocol(
            PROTOCOLS[1],
            frames["rare_test"],
            frames["rare_gallery"],
            top_k_values,
            distance_fn,
            label_col,
            rare_splits,
        )
    )
    rows.extend(
        evaluate_protocol(
            PROTOCOLS[2],
            frames["freq_test"],
            all_gallery,
            top_k_values,
            distance_fn,
            label_col,
            rare_splits if not pair_mode or not frames["rare_gallery"].empty else None,
        )
    )
    rows.extend(
        evaluate_protocol(
            PROTOCOLS[3],
            frames["rare_test"],
            all_gallery,
            top_k_values,
            distance_fn,
            label_col,
            rare_splits,
        )
    )
    return rows


def output_lines(args, summary, primary_columns, debug_columns, details_path):
    if args.mode == "image":
        title = "# GMDB Frequent/Rare Evaluation"
        input_lines = [
            f"- Encodings: `{args.encodings}`",
            f"- Metadata dir: `{args.metadata_dir}`",
            f"- Version: `{args.version}`",
            f"- Split details: `{details_path}`",
            "- Primary metric: macro label accuracy, matching the original `evaluate_ensemble.py` comparison style.",
        ]
    else:
        title = "# GMDB Pair-Level Frequent/Rare Evaluation"
        input_lines = [
            f"- Fused encodings: `{args.fused_encodings}`",
            f"- View metadata: `{args.view_metadata}`",
            f"- Metadata dir: `{args.metadata_dir}`",
            f"- Version: `{args.version}`",
            f"- Pair distance: `{args.pair_distance}`",
            f"- Split details: `{details_path}`",
            "- Primary metric: macro label accuracy over pair-level fusion embeddings.",
        ]

    return [
        title,
        "",
        *input_lines,
        "",
        "## Macro Label Accuracy",
        "",
        markdown_table(summary[primary_columns]),
        "",
        "## Micro Query Accuracy",
        "",
        markdown_table(summary[debug_columns]),
        "",
    ]


def main():
    args = parse_args()
    top_k_values = sorted(set(args.top_k))
    metadata = load_protocol_metadata(args.metadata_dir, args.version)

    if args.mode == "image":
        frames = build_image_frames(args, metadata)
        distance_fn = lambda query_df, gallery_df: image_distances(query_df, gallery_df, args.allow_self_match)
        label_col = "label"
        pair_mode = False
    else:
        frames = build_pair_frames(args, metadata)
        distance_fn = lambda query_df, gallery_df: pair_distances(
            query_df,
            gallery_df,
            args.allow_self_match,
            args.pair_distance,
        )
        label_col = args.label_col
        pair_mode = True

    rare_splits = (
        split_values(metadata["rare_gallery"], metadata["rare_test"])
        if pair_mode
        else split_values(frames["rare_gallery"], frames["rare_test"])
    )
    rows = evaluate_all_protocols(frames, top_k_values, distance_fn, label_col, rare_splits, pair_mode)
    summary, split_details = aggregate_rows(rows, top_k_values)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    details_path = output_path.with_name(f"{output_path.stem}_splits.csv")
    split_details.to_csv(details_path, index=False)

    primary_columns = ["protocol", "splits", "queries", "gallery", "labels"] + [
        f"macro_top_{top_k}" for top_k in top_k_values
    ]
    debug_columns = ["protocol"] + [f"micro_top_{top_k}" for top_k in top_k_values]
    output_path.write_text("\n".join(output_lines(args, summary, primary_columns, debug_columns, details_path)), encoding="utf-8")

    print(f"Saved summary: {output_path}")
    print(f"Saved split details: {details_path}")
    print(markdown_table(summary[primary_columns]))


if __name__ == "__main__":
    main()
