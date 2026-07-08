"""
Evaluate late-fusion retrieval rankings with pair-level labels.

Fusion rows represent frontal/profile pairs, so row-level image metadata must
first be collapsed to one row per pair_id before label accuracy is computed.
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fusion retrieval rankings by pair-level labels.")
    parser.add_argument("--rankings", required=True, help="Fusion rankings CSV from run_retrieval.py.")
    parser.add_argument("--fused_encodings", required=True, help="Fused encoding PKL containing img_name and pair_id.")
    parser.add_argument("--metadata", required=True, help="View-aware metadata CSV.")
    parser.add_argument("--output_csv", required=True, help="Enriched ranking CSV output.")
    parser.add_argument("--output_summary", required=True, help="Markdown summary output.")
    parser.add_argument("--pair_metadata_output", default="", help="Optional pair-level metadata CSV output.")
    parser.add_argument("--pair_col", default="pair_id", help="Pair identifier column.")
    parser.add_argument("--label_col", default="label", help="Label column.")
    parser.add_argument("--top_k", nargs="*", type=int, default=[1, 5, 10, 30], help="Top-k values to summarize.")
    return parser.parse_args()


def first_non_empty(values):
    for value in values:
        if pd.notna(value) and str(value) != "":
            return value
    return ""


def unique_or_first(values, column, pair_id):
    non_empty = [value for value in values if pd.notna(value) and str(value) != ""]
    unique_values = sorted(set(non_empty))
    if len(unique_values) > 1:
        raise ValueError(f"Pair {pair_id} has multiple {column} values: {unique_values}")
    return unique_values[0] if unique_values else ""


def build_pair_metadata(metadata, pair_col, label_col):
    required = {pair_col, "split", label_col, "patient_id", "view", "is_aligned"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

    paired = metadata[metadata[pair_col].notna()].copy()
    paired = paired[paired[pair_col].astype(str) != ""].copy()
    rows = []
    optional_columns = ["syndrome_id", "syndrome_name", "gene_id", "gene_name"]
    for pair_id, group in paired.groupby(pair_col, dropna=False):
        row = {
            pair_col: pair_id,
            "split": unique_or_first(group["split"], "split", pair_id),
            label_col: unique_or_first(group[label_col], label_col, pair_id),
            "patient_id": unique_or_first(group["patient_id"], "patient_id", pair_id),
            "n_frontal_aligned": int(((group["view"] == "frontal") & truthy(group["is_aligned"])).sum()),
            "n_profile_aligned": int(((group["view"] == "profile") & truthy(group["is_aligned"])).sum()),
            "n_rows": len(group),
        }
        for column in optional_columns:
            if column in group.columns:
                row[column] = first_non_empty(group[column])
        rows.append(row)
    return pd.DataFrame(rows)


def truthy(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def add_pair_ids(rankings, fused_encodings, pair_col):
    required = {"img_name", pair_col}
    missing = required - set(fused_encodings.columns)
    if missing:
        raise ValueError(f"Fused encodings are missing required columns: {sorted(missing)}")

    mapping = fused_encodings[["img_name", pair_col]].drop_duplicates()
    enriched = rankings.merge(
        mapping.rename(columns={"img_name": "query_img_name", pair_col: "query_pair_id"}),
        how="left",
        on="query_img_name",
    )
    enriched = enriched.merge(
        mapping.rename(columns={"img_name": "gallery_img_name", pair_col: "gallery_pair_id"}),
        how="left",
        on="gallery_img_name",
    )
    if enriched["query_pair_id"].isna().any() or enriched["gallery_pair_id"].isna().any():
        raise ValueError("Some ranking rows could not be mapped to query/gallery pair_id.")
    return enriched


def add_pair_metadata(rankings, pair_metadata, pair_col, label_col):
    query_metadata = pair_metadata.rename(
        columns={column: f"query_{column}" for column in pair_metadata.columns if column != pair_col}
    ).rename(columns={pair_col: "query_pair_id"})
    gallery_metadata = pair_metadata.rename(
        columns={column: f"gallery_{column}" for column in pair_metadata.columns if column != pair_col}
    ).rename(columns={pair_col: "gallery_pair_id"})
    enriched = rankings.merge(query_metadata, how="left", on="query_pair_id")
    enriched = enriched.merge(gallery_metadata, how="left", on="gallery_pair_id")
    for column in [f"query_{label_col}", f"gallery_{label_col}"]:
        if enriched[column].isna().any():
            raise ValueError(f"Missing pair metadata after merge: {column}")
    enriched["label_match"] = enriched[f"query_{label_col}"].astype(str) == enriched[f"gallery_{label_col}"].astype(str)
    return enriched


def summarize_topk(enriched, top_k_values):
    rows = []
    query_count = enriched["query_img_name"].nunique()
    for top_k in top_k_values:
        within_k = enriched[enriched["rank"] <= top_k]
        hits = within_k.groupby("query_img_name")["label_match"].any()
        rows.append(
            {
                "top_k": top_k,
                "queries": query_count,
                "hits": int(hits.sum()),
                "accuracy": float(hits.mean()),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df):
    formatted = df.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: f"{value:.4f}")
    formatted = formatted.fillna("").astype(str)
    headers = list(formatted.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in formatted.iterrows():
        lines.append("| " + " | ".join(row[column] for column in headers) + " |")
    return "\n".join(lines)


def main():
    args = parse_args()
    rankings = pd.read_csv(args.rankings)
    fused_encodings = pd.read_pickle(args.fused_encodings)
    metadata = pd.read_csv(args.metadata)

    pair_metadata = build_pair_metadata(metadata, args.pair_col, args.label_col)
    enriched = add_pair_ids(rankings, fused_encodings, args.pair_col)
    enriched = add_pair_metadata(enriched, pair_metadata, args.pair_col, args.label_col)
    summary = summarize_topk(enriched, sorted(set(args.top_k)))

    output_csv = Path(args.output_csv)
    output_summary = Path(args.output_summary)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_csv, index=False)

    if args.pair_metadata_output:
        pair_metadata_path = Path(args.pair_metadata_output)
        pair_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        pair_metadata.to_csv(pair_metadata_path, index=False)

    lines = [
        "# Fusion Label Accuracy",
        "",
        f"- Rankings: `{args.rankings}`",
        f"- Fused encodings: `{args.fused_encodings}`",
        f"- Metadata: `{args.metadata}`",
        f"- Enriched rankings: `{args.output_csv}`",
        "",
        "## Top-k Label Accuracy",
        "",
        markdown_table(summary),
        "",
    ]
    output_summary.write_text("\n".join(lines), encoding="utf-8")

    print(f"Pair metadata rows: {len(pair_metadata)}")
    print(f"Enriched ranking rows: {len(enriched)}")
    print(f"Saved enriched rankings: {output_csv}")
    print(f"Saved summary: {output_summary}")
    print(markdown_table(summary))


if __name__ == "__main__":
    main()
