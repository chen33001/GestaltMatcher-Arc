"""
Summarize one or more generic retrieval ranking CSV files.

The summary focuses on top-1 distance distributions and optional grouped
counts, without assuming a specific dataset such as CFP or GMDB.

Examples:
    python src/retrieval/summarize_rankings.py --rankings data/CFP/results/cfp_m1_rankings.csv --labels m1 --group_by query_view --output data/CFP/results/cfp_m1_summary.md
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize retrieval ranking CSVs.")
    parser.add_argument("--rankings", nargs="+", required=True, help="Ranking CSV file(s).")
    parser.add_argument(
        "--labels",
        nargs="*",
        default=[],
        help="Optional labels matching --rankings. Defaults to each file stem.",
    )
    parser.add_argument(
        "--group_by",
        nargs="*",
        default=[],
        help="Optional columns for grouped top-1 distance summaries.",
    )
    parser.add_argument(
        "--count_columns",
        nargs="*",
        default=[],
        help="Optional columns for top-1 value counts, e.g. gallery_syndrome_name.",
    )
    parser.add_argument("--top_n_counts", type=int, default=10, help="Rows per count table group.")
    parser.add_argument("--output", required=True, help="Markdown output path.")
    return parser.parse_args()


def labels_for(args):
    if args.labels and len(args.labels) != len(args.rankings):
        raise ValueError("--labels must have the same number of values as --rankings")
    if args.labels:
        return args.labels
    return [Path(path).stem for path in args.rankings]


def summarize_distances(rankings, label, group_by):
    top1 = rankings[rankings["rank"] == 1].copy()
    if top1.empty:
        return pd.DataFrame()

    group_columns = list(group_by)
    if not group_columns:
        top1["_all"] = "all"
        group_columns = ["_all"]

    rows = []
    for keys, group in top1.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {"ranking": label, "queries": len(group)}
        for column, value in zip(group_columns, keys):
            if column != "_all":
                row[column] = value
        row.update(
            {
                "mean_top1_distance": group["distance"].mean(),
                "median_top1_distance": group["distance"].median(),
                "p25_top1_distance": group["distance"].quantile(0.25),
                "p75_top1_distance": group["distance"].quantile(0.75),
                "min_top1_distance": group["distance"].min(),
                "max_top1_distance": group["distance"].max(),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_counts(rankings, label, value_column, group_by, top_n):
    if value_column not in rankings.columns:
        raise ValueError(f"Count column '{value_column}' not found in rankings.")
    top1 = rankings[rankings["rank"] == 1].copy()
    if top1.empty:
        return pd.DataFrame()

    group_columns = list(group_by)
    if not group_columns:
        top1["_all"] = "all"
        group_columns = ["_all"]

    rows = []
    for keys, group in top1.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        counts = group[value_column].fillna("Unknown").replace("", "Unknown").value_counts().head(top_n)
        for value, count in counts.items():
            row = {"ranking": label}
            for column, group_value in zip(group_columns, keys):
                if column != "_all":
                    row[column] = group_value
            row.update(
                {
                    value_column: value,
                    "count": count,
                    "fraction": count / len(group),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(df, float_digits=4):
    if df.empty:
        return "_No rows._"
    formatted = df.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: f"{value:.{float_digits}f}")
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

    ranking_frames = []
    for path, label in zip(args.rankings, labels):
        frame = pd.read_csv(path)
        required = {"rank", "distance"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        ranking_frames.append((label, frame))

    distance_summary = pd.concat(
        [summarize_distances(frame, label, args.group_by) for label, frame in ranking_frames],
        ignore_index=True,
    )

    lines = [
        "# Retrieval Ranking Summary",
        "",
        "## Inputs",
        "",
    ]
    for path, label in zip(args.rankings, labels):
        lines.append(f"- `{label}`: `{path}`")
    lines.extend(
        [
            "",
            "## Top-1 Distance Summary",
            "",
            markdown_table(distance_summary),
            "",
        ]
    )

    for column in args.count_columns:
        count_summary = pd.concat(
            [
                summarize_counts(frame, label, column, args.group_by, args.top_n_counts)
                for label, frame in ranking_frames
            ],
            ignore_index=True,
        )
        lines.extend(
            [
                f"## Top-1 Counts: `{column}`",
                "",
                markdown_table(count_summary),
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved summary: {output_path}")
    print(markdown_table(distance_summary))


if __name__ == "__main__":
    main()
