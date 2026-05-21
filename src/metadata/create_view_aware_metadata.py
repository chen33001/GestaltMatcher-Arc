"""
Create a standardized view-aware metadata table from a dataset manifest.

The output schema is shared by retrieval, view filtering, split checks, and
future frontal/profile experiments. Dataset-specific details are supplied by
column-name arguments instead of hard-coded Python scripts.

Example for CFP:
    python src/metadata/create_view_aware_metadata.py --dataset CFP --manifest data/CFP/metadata/cfp_aligned_manifest.csv --output data/CFP/metadata/cfp_view_aware_metadata.csv --schema data/CFP/metadata/view_aware_metadata_schema.md --split validation
"""

import argparse
from pathlib import Path

import pandas as pd


VIEW_AWARE_COLUMNS = [
    "dataset",
    "image_id",
    "patient_id",
    "person_id",
    "subject_id",
    "view",
    "split",
    "is_aligned",
    "is_paired",
    "pair_id",
    "label",
    "syndrome_id",
    "syndrome_name",
    "gene_id",
    "gene_name",
    "source_path",
    "flat_filename",
    "aligned_filename",
    "aligned_path",
]


SCHEMA = {
    "dataset": "Dataset/source name, e.g. CFP, GMDB, or an external validation set.",
    "image_id": "Unique image identifier within the project metadata.",
    "patient_id": "Leakage-control identifier. Split train/val/test by this field.",
    "person_id": "Original dataset person identifier, if available.",
    "subject_id": "Repo-compatible subject field. Defaults to patient_id if no subject column is supplied.",
    "view": "Explicit view label, e.g. frontal, profile, or unknown.",
    "split": "Dataset split or usage label, e.g. validation, train, val, test, external_test.",
    "is_aligned": "Whether aligned_path points to a successfully aligned image.",
    "is_paired": "Whether this row's pair_id has all requested paired views.",
    "pair_id": "Identifier used to group paired views from the same patient or subject.",
    "label": "Training class label, if available.",
    "syndrome_id": "Internal syndrome/disorder label, if available.",
    "syndrome_name": "Human-readable syndrome name, if available.",
    "gene_id": "Gene identifier, if available.",
    "gene_name": "Gene name, if available.",
    "source_path": "Path inside the original source dataset.",
    "flat_filename": "Unique flattened raw filename.",
    "aligned_filename": "Aligned image filename.",
    "aligned_path": "Path to aligned image used for encoding/training.",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Create standardized view-aware metadata.")
    parser.add_argument("--dataset", required=True, help="Dataset/source name written to every row.")
    parser.add_argument("--manifest", required=True, help="Input manifest CSV.")
    parser.add_argument("--output", required=True, help="Output metadata CSV.")
    parser.add_argument("--schema", default="", help="Optional schema markdown output.")
    parser.add_argument("--split", default="validation", help="Split label written to every row.")
    parser.add_argument(
        "--paired_views",
        nargs="+",
        default=["frontal", "profile"],
        help="Views required for is_paired=True.",
    )
    parser.add_argument("--image_id_col", default="image_id")
    parser.add_argument("--patient_id_col", default="patient_id")
    parser.add_argument("--person_id_col", default="person_id")
    parser.add_argument("--subject_id_col", default="")
    parser.add_argument("--view_col", default="view")
    parser.add_argument("--pair_id_col", default="")
    parser.add_argument("--source_path_col", default="source_path")
    parser.add_argument("--flat_filename_col", default="flat_filename")
    parser.add_argument("--aligned_filename_col", default="aligned_filename")
    parser.add_argument("--aligned_path_col", default="aligned_path")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--syndrome_id_col", default="syndrome_id")
    parser.add_argument("--syndrome_name_col", default="syndrome_name")
    parser.add_argument("--gene_id_col", default="gene_id")
    parser.add_argument("--gene_name_col", default="gene_name")
    return parser.parse_args()


def optional_value(row, column):
    if not column or column not in row.index:
        return ""
    value = row[column]
    return "" if pd.isna(value) else value


def required_value(row, column, purpose):
    if column not in row.index:
        raise ValueError(f"Manifest is missing required {purpose} column: {column}")
    value = row[column]
    return "" if pd.isna(value) else value


def write_schema(path):
    lines = [
        "# View-Aware Metadata Schema",
        "",
        "This schema is the project-level metadata format for view-aware retrieval, validation, and training split derivation.",
        "",
        "## Columns",
        "",
    ]
    for column in VIEW_AWARE_COLUMNS:
        lines.append(f"- `{column}`: {SCHEMA[column]}")
    lines.extend(
        [
            "",
            "## Training CSV Derivation",
            "",
            "For `train_gm_arc.py`, derive training CSVs with the repo's expected columns:",
            "",
            "```text",
            "image_id,subject,label",
            "```",
            "",
            "Filter this metadata by `view`, `split`, and available labels before supervised training.",
            "",
            "## Leakage Rule",
            "",
            "Patient-level splits should be checked with `patient_id`: no `patient_id` should appear in more than one train/val/test/external split.",
            "",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    manifest = pd.read_csv(args.manifest)

    pair_column = args.pair_id_col or args.patient_id_col
    if pair_column not in manifest.columns:
        raise ValueError(f"Pair column not found in manifest: {pair_column}")
    if args.view_col not in manifest.columns:
        raise ValueError(f"View column not found in manifest: {args.view_col}")

    required_views = set(args.paired_views)
    view_counts = manifest.groupby(pair_column)[args.view_col].agg(lambda values: set(values))
    paired_ids = {
        pair_id
        for pair_id, views in view_counts.items()
        if required_views.issubset(views)
    }

    rows = []
    for _, row in manifest.iterrows():
        patient_id = required_value(row, args.patient_id_col, "patient_id")
        pair_id = optional_value(row, args.pair_id_col) if args.pair_id_col else patient_id
        subject_id = optional_value(row, args.subject_id_col) if args.subject_id_col else patient_id
        rows.append(
            {
                "dataset": args.dataset,
                "image_id": required_value(row, args.image_id_col, "image_id"),
                "patient_id": patient_id,
                "person_id": optional_value(row, args.person_id_col),
                "subject_id": subject_id,
                "view": required_value(row, args.view_col, "view"),
                "split": args.split,
                "is_aligned": bool(optional_value(row, args.aligned_path_col)),
                "is_paired": pair_id in paired_ids,
                "pair_id": pair_id if pair_id in paired_ids else "",
                "label": optional_value(row, args.label_col),
                "syndrome_id": optional_value(row, args.syndrome_id_col),
                "syndrome_name": optional_value(row, args.syndrome_name_col),
                "gene_id": optional_value(row, args.gene_id_col),
                "gene_name": optional_value(row, args.gene_name_col),
                "source_path": optional_value(row, args.source_path_col),
                "flat_filename": optional_value(row, args.flat_filename_col),
                "aligned_filename": optional_value(row, args.aligned_filename_col),
                "aligned_path": optional_value(row, args.aligned_path_col),
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=VIEW_AWARE_COLUMNS)
    df.to_csv(output_path, index=False)

    if args.schema:
        schema_path = Path(args.schema)
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        write_schema(schema_path)
        print(f"Saved schema: {schema_path}")

    print(f"Saved metadata: {output_path}")
    print(f"Rows: {len(df)}")
    print(df.groupby(["view", "is_paired"]).size().reset_index(name="rows"))


if __name__ == "__main__":
    main()
