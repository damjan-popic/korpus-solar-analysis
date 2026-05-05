#!/usr/bin/env python3
"""One-command runner for the Šolar 3.0 lexical analysis workflow.

The runner executes, in order:
1. ID-safe CLASSLA reannotation of the original CoNLL-U file.
2. Corpus analysis based on the reannotated CoNLL-U and metadata.
3. Optional Markdown report generation.

All paths can be edited in ``configs/solar_pipeline.example.json`` or supplied
through CLI overrides.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rel(path_value: str | None) -> Optional[Path]:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def run_command(cmd: List[str], dry_run: bool = False) -> None:
    print("\n$ " + " ".join(str(part) for part in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def build_reannotation_cmd(config: Dict[str, Any], args: argparse.Namespace) -> List[str]:
    reann = config.get("reannotation", {})
    input_path = args.input or rel(reann.get("input"))
    output_path = args.annotated_output or rel(reann.get("output"))
    metadata_path = args.metadata or rel(reann.get("metadata"))

    if input_path is None or output_path is None:
        raise SystemExit("Reannotation requires input and output paths.")

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "solar_reannotate.py"),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--diff-report",
        str(rel(reann.get("diff_report")) or REPO_ROOT / "reports" / "solar-classla-diff.tsv"),
        "--summary",
        str(rel(reann.get("summary")) or REPO_ROOT / "reports" / "solar-classla-summary.json"),
        "--errors",
        str(rel(reann.get("errors")) or REPO_ROOT / "reports" / "solar-classla-errors.tsv"),
        "--validation-report",
        str(rel(reann.get("validation_report")) or REPO_ROOT / "reports" / "solar-classla-validation.tsv"),
        "--metadata-id-columns",
        reann.get("metadata_id_columns", "OrigID,CorrID"),
        "--metadata-strip-suffix-regex",
        reann.get("metadata_strip_suffix_regex", "[st]$"),
        "--min-metadata-match-ratio",
        str(reann.get("min_metadata_match_ratio", 0.95)),
        "--require-sent-id",
        "--require-clean-output",
        "--batch-size",
        str(reann.get("batch_size", 64)),
    ]
    if metadata_path is not None:
        cmd.extend(["--metadata", str(metadata_path)])
    models_dir = args.models_dir or rel(config.get("models_dir"))
    if models_dir is not None:
        cmd.extend(["--models-dir", str(models_dir)])
    if args.download_models or config.get("download_models", False):
        cmd.append("--download-models")
    for key in ["lang", "classla_type", "processors"]:
        if key in reann:
            option = "--" + key.replace("_", "-")
            cmd.extend([option, str(reann[key])])
    return cmd


def build_analysis_cmd(config: Dict[str, Any], args: argparse.Namespace) -> List[str]:
    analysis = config.get("analysis", {})
    input_path = args.annotated_output or rel(analysis.get("input")) or rel(config.get("reannotation", {}).get("output"))
    metadata_path = args.metadata or rel(analysis.get("metadata")) or rel(config.get("reannotation", {}).get("metadata"))
    output_dir = args.output_dir or rel(analysis.get("output_dir")) or REPO_ROOT / "analysis"

    if input_path is None or metadata_path is None:
        raise SystemExit("Analysis requires annotated input and metadata paths.")

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "solar_analysis.py"),
        "--input",
        str(input_path),
        "--metadata",
        str(metadata_path),
        "--output-dir",
        str(output_dir),
        "--rare-top-n",
        str(analysis.get("rare_top_n", 50)),
        "--rare-min-count",
        str(analysis.get("rare_min_count", 2)),
        "--mwe-min-n",
        str(analysis.get("mwe_min_n", 2)),
        "--mwe-max-n",
        str(analysis.get("mwe_max_n", 4)),
        "--mwe-min-count",
        str(analysis.get("mwe_min_count", 5)),
    ]
    reference = args.reference_freq or rel(analysis.get("reference_freq"))
    if reference is not None:
        cmd.extend(["--reference-freq", str(reference)])
    if args.compute_mwe or analysis.get("compute_mwe", True):
        cmd.append("--compute-mwe")
    if args.run_stats or analysis.get("run_stats", True):
        cmd.append("--run-stats")
    if args.run_models or analysis.get("run_models", True):
        cmd.append("--run-models")
    if args.make_plots or analysis.get("make_plots", True):
        cmd.append("--make-plots")
    return cmd


def build_report_cmd(config: Dict[str, Any], args: argparse.Namespace) -> List[str]:
    analysis = config.get("analysis", {})
    output_dir = args.output_dir or rel(analysis.get("output_dir")) or REPO_ROOT / "analysis"
    report_path = args.report or rel(config.get("report")) or output_dir / "solar_analysis_report.md"
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_report.py"),
        "--analysis-dir",
        str(output_dir),
        "--output",
        str(report_path),
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full Šolar reannotation + analysis workflow.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "solar_pipeline.example.json")
    parser.add_argument("--input", type=Path, default=None, help="Override raw CoNLL-U input path.")
    parser.add_argument("--annotated-output", type=Path, default=None, help="Override reannotated CoNLL-U output path.")
    parser.add_argument("--metadata", type=Path, default=None, help="Override solar-meta.tsv path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override analysis output directory.")
    parser.add_argument("--reference-freq", type=Path, default=None, help="Optional external lemma frequency list.")
    parser.add_argument("--models-dir", type=Path, default=None, help="Shared CLASSLA models directory.")
    parser.add_argument("--download-models", action="store_true", help="Download CLASSLA models first.")
    parser.add_argument("--skip-reannotation", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--compute-mwe", action="store_true")
    parser.add_argument("--run-stats", action="store_true")
    parser.add_argument("--run-models", action="store_true")
    parser.add_argument("--make-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    config = load_config(args.config)

    if not args.skip_reannotation:
        run_command(build_reannotation_cmd(config, args), dry_run=args.dry_run)
    if not args.skip_analysis:
        run_command(build_analysis_cmd(config, args), dry_run=args.dry_run)
    if not args.skip_report:
        run_command(build_report_cmd(config, args), dry_run=args.dry_run)

    print("\nPipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
