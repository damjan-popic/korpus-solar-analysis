#!/usr/bin/env python3
"""Build a compact Markdown report from ``scripts/solar_analysis.py`` outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


KEY_METRICS = [
    "lemma_mattr_50",
    "lemma_mtld",
    "lemma_hdd",
    "lexical_density",
    "rare_lemma_token_share_top3000",
    "deprel_entropy",
    "mean_dependency_distance",
]


def read_tsv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return None


def md_table(df: pd.DataFrame, columns: Optional[List[str]] = None, n: int = 12) -> str:
    if df is None or df.empty:
        return "_No data available._"
    if columns is not None:
        columns = [c for c in columns if c in df.columns]
        df = df[columns]
    df = df.head(n).copy()
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    return df.to_markdown(index=False)


def find_summary(analysis_dir: Path, name: str) -> Optional[pd.DataFrame]:
    return read_tsv(analysis_dir / "summaries" / name)


def top_metric_table(summary: Optional[pd.DataFrame], group_col: str) -> str:
    if summary is None or summary.empty:
        return "_No grouped summary was produced._"
    metric_cols = [c for c in [group_col, "n_docs", "mean_n_word_tokens"] + ["mean_" + m for m in KEY_METRICS] if c in summary.columns]
    return md_table(summary, metric_cols, n=20)


def write_report(analysis_dir: Path, output: Path) -> None:
    summary_path = analysis_dir / "corpus_summary.json"
    corpus_summary = {}
    if summary_path.exists():
        corpus_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    doc_df = read_tsv(analysis_dir / "document_metrics.tsv")
    region_summary = find_summary(analysis_dir, "summary_by_region.tsv")
    grade_summary = find_summary(analysis_dir, "summary_by_grade.tsv")
    level_summary = find_summary(analysis_dir, "summary_by_school_level.tsv")
    contact_summary = find_summary(analysis_dir, "summary_by_contact_zone.tsv")
    year_summary = find_summary(analysis_dir, "summary_by_year_start.tsv")

    stats_dir = analysis_dir / "stats"
    spearman = read_tsv(stats_dir / "spearman_progression.tsv")
    kruskal = read_tsv(stats_dir / "kruskal_region.tsv")
    mw = read_tsv(stats_dir / "mannwhitney_school_level.tsv")
    ols = read_tsv(analysis_dir / "models" / "ols_control_models.tsv")

    rare_region = read_tsv(analysis_dir / "rare_words" / "rarest_lemmas_by_region.tsv")
    rare_contact = read_tsv(analysis_dir / "rare_words" / "rarest_lemmas_by_contact_zone.tsv")
    mwe_region = read_tsv(analysis_dir / "mwe" / "mwe_candidates_by_region.tsv")
    mwe_contact = read_tsv(analysis_dir / "mwe" / "mwe_candidates_by_contact_zone.tsv")

    lines: List[str] = []
    lines.append("# Šolar 3.0 lexical analysis report")
    lines.append("")
    lines.append("## Corpus and metadata coverage")
    lines.append("")
    if corpus_summary:
        coverage = pd.DataFrame([
            {"item": "documents_in_corpus", "value": corpus_summary.get("documents_in_corpus", "")},
            {"item": "sentences_in_corpus", "value": corpus_summary.get("sentences_in_corpus", "")},
            {"item": "tokens_in_corpus", "value": corpus_summary.get("tokens_in_corpus", "")},
            {"item": "documents_in_metadata", "value": corpus_summary.get("documents_in_metadata", "")},
            {"item": "documents_matched_to_metadata", "value": corpus_summary.get("documents_matched_to_metadata", "")},
            {"item": "reference_frequency_source", "value": corpus_summary.get("reference_frequency_source", "")},
        ])
        lines.append(md_table(coverage, n=20))
    else:
        lines.append("_No corpus summary found._")
    lines.append("")

    lines.append("## Methodological scope")
    lines.append("")
    lines.append(
        "The pipeline operationalizes the disposition around lexical diversity/pestrost, lexical density/gostota, "
        "lexical sophistication/zahtevnost through frequency ranks, syntactic diversity from UD dependency output, "
        "and automatic MWE/collocation candidate extraction. Main grouping variables are region, contact zone, grade, "
        "school level, collection year, and text type."
    )
    lines.append("")
    lines.append("Key document-level measures include MATTR-50, MTLD, HD-D, lemma TTR variants, lexical density, rare-lemma shares outside top frequency bands, dependency relation diversity, dependency distance, tree depth, and subordination ratio.")
    lines.append("")

    lines.append("## Main grouped summaries")
    lines.append("")
    lines.append("### By school level")
    lines.append(top_metric_table(level_summary, "school_level"))
    lines.append("")
    lines.append("### By grade")
    lines.append(top_metric_table(grade_summary, "grade"))
    lines.append("")
    lines.append("### By region")
    lines.append(top_metric_table(region_summary, "region"))
    lines.append("")
    lines.append("### By contact zone")
    lines.append(top_metric_table(contact_summary, "contact_zone"))
    lines.append("")
    lines.append("### By year")
    lines.append(top_metric_table(year_summary, "year_start"))
    lines.append("")

    lines.append("## Statistical tests")
    lines.append("")
    lines.append("### Grade/year progression: Spearman")
    lines.append(md_table(spearman, n=30))
    lines.append("")
    lines.append("### Region differences: Kruskal-Wallis")
    lines.append(md_table(kruskal, n=30))
    lines.append("")
    lines.append("### Osnovna vs. srednja: Mann-Whitney U")
    lines.append(md_table(mw, n=30))
    lines.append("")
    lines.append("### Exploratory OLS control models")
    lines.append(md_table(ols, ["metric", "term", "coef", "std_err", "t", "p_value", "n", "r_squared", "formula"], n=60))
    lines.append("")

    lines.append("## Rare lemma tables")
    lines.append("")
    lines.append("### Rarest lemmas by region")
    lines.append(md_table(rare_region, ["region", "lemma_norm", "group_count", "ref_rank", "relative_freq_per_million"], n=40))
    lines.append("")
    lines.append("### Rarest lemmas by contact zone")
    lines.append(md_table(rare_contact, ["contact_zone", "lemma_norm", "group_count", "ref_rank", "relative_freq_per_million"], n=40))
    lines.append("")

    lines.append("## MWE / collocation candidates")
    lines.append("")
    lines.append("### Top regional candidates")
    lines.append(md_table(mwe_region, ["region", "n", "ngram", "upos_pattern", "count", "doc_count", "pmi", "dice", "t_score"], n=40))
    lines.append("")
    lines.append("### Top contact-zone candidates")
    lines.append(md_table(mwe_contact, ["contact_zone", "n", "ngram", "upos_pattern", "count", "doc_count", "pmi", "dice", "t_score"], n=40))
    lines.append("")

    lines.append("## Files to inspect next")
    lines.append("")
    lines.append("- `document_metrics.tsv` for document-level modeling.")
    lines.append("- `diagnostics/metadata_join_diagnostics.tsv` to verify metadata coverage.")
    lines.append("- `mwe/mwe_candidates_by_region.tsv` and `mwe/mwe_candidates_by_contact_zone.tsv` for qualitative validation.")
    lines.append("- `plots/` for quick visual checks when `--make-plots` was used.")
    lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to: {output}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Markdown report from Solar analysis outputs.")
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis"))
    parser.add_argument("--output", type=Path, default=Path("analysis/solar_analysis_report.md"))
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    write_report(args.analysis_dir.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
