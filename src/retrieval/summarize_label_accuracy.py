"""
Summarize top-k label accuracy for one or more retrieval ranking CSV files.
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize top-k label accuracy from retrieval rankings.")
    parser.add_argument("--rankings", nargs="+", required=True, help="Ranking CSV file(s).")
    parser.add_argument(
        "--labels",
        nargs="*",
        default=[],
        help="Optional labels matching --rankings. Defaults to each file stem.",
    )
    parser.add_argument("--query_label_col", default="query_label", help="Query label column.")
    parser.add_argument("--gallery_label_col", default="gallery_label", help="Gallery label column.")
    parser.add_argument("--top_k", nargs="*", type=int, default=[1, 5, 10, 30], help="Top-k values.")
    parser.add_argument("--output", required=True, help="Markdown output path.")
    return parser.parse_args()


def labels_for(args):
    if args.labels and len(args.labels) != len(args.rankings):
        raise ValueError("--labels must have the same number of values as --rankings")
    if args.labels:
        return args.labels
    return [Path(path).stem for path in args.rankings]


def summarize_file(path, label, top_k_values, query_label_col, gallery_label_col):
    rankings = pd.read_csv(path)
    required = {"query_img_name", "rank", query_label_col, gallery_label_col}
    missing = required - set(rankings.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    rankings = rankings.copy()
    rankings["label_match"] = (
        rankings[query_label_col].astype(str) == rankings[gallery_label_col].astype(str)
    )
    query_count = rankings["query_img_name"].nunique()

    rows = []
    for top_k in sorted(set(top_k_values)):
        within_k = rankings[rankings["rank"] <= top_k]
        hits = within_k.groupby("query_img_name")["label_match"].any()
        rows.append(
            {
                "ranking": label,
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
    labels = labels_for(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summaries = [
        summarize_file(path, label, args.top_k, args.query_label_col, args.gallery_label_col)
        for path, label in zip(args.rankings, labels)
    ]
    summary = pd.concat(summaries, ignore_index=True)

    lines = [
        "# Retrieval Label Accuracy Summary",
        "",
        "## Inputs",
        "",
    ]
    for path, label in zip(args.rankings, labels):
        lines.append(f"- `{label}`: `{path}`")
    lines.extend(
        [
            "",
            "## Top-k Label Accuracy",
            "",
            markdown_table(summary),
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved summary: {output_path}")
    print(markdown_table(summary))


if __name__ == "__main__":
    main()
