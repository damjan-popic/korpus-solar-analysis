#!/usr/bin/env python3
"""Prepare a qualitative-review sheet for MWE/collocation candidates.

The quantitative extraction deliberately over-generates. This helper takes one
or more MWE candidate tables and creates a review-ready TSV with empty columns
for manual validation and language-contact notes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def load_tables(paths: List[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(path, sep="\t")
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        df["source_file"] = path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create manual MWE validation TSV.")
    parser.add_argument(
        "--input",
        type=Path,
        nargs="*",
        default=None,
        help="MWE TSV files. Default: analysis/mwe/mwe_candidates_by_region.tsv and by_contact_zone.tsv.",
    )
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis"))
    parser.add_argument("--output", type=Path, default=Path("analysis/mwe/mwe_manual_review.tsv"))
    parser.add_argument("--top-per-group", type=int, default=100)
    args = parser.parse_args()

    if args.input:
        input_paths = args.input
    else:
        input_paths = [
            args.analysis_dir / "mwe" / "mwe_candidates_by_region.tsv",
            args.analysis_dir / "mwe" / "mwe_candidates_by_contact_zone.tsv",
        ]

    df = load_tables(input_paths)
    if df.empty:
        raise SystemExit("No MWE candidates found. Run solar_analysis.py with --compute-mwe first.")

    sort_cols = [c for c in ["source_file", "region", "contact_zone", "n", "count", "pmi", "dice"] if c in df.columns]
    ascending = [True] * len(sort_cols)
    for i, col in enumerate(sort_cols):
        if col in {"count", "pmi", "dice"}:
            ascending[i] = False
    df = df.sort_values(sort_cols, ascending=ascending)

    group_cols = [c for c in ["source_file", "region", "contact_zone", "n"] if c in df.columns]
    if group_cols:
        df = df.groupby(group_cols, dropna=False).head(args.top_per_group).reset_index(drop=True)

    df.insert(0, "review_status", "")
    df.insert(1, "is_valid_mwe", "")
    df.insert(2, "is_formulaic", "")
    df.insert(3, "possible_contact_language", "")
    df.insert(4, "style_register_note", "")
    df.insert(5, "review_comment", "")

    preferred = [
        "review_status",
        "is_valid_mwe",
        "is_formulaic",
        "possible_contact_language",
        "style_register_note",
        "review_comment",
        "source_file",
        "region",
        "contact_zone",
        "grade",
        "school_level",
        "year_start",
        "n",
        "ngram",
        "upos_pattern",
        "count",
        "doc_count",
        "pmi",
        "dice",
        "t_score",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, sep="\t", index=False)
    print(f"Review sheet written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
