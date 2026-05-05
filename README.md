# Šolar 3.0 lexical analysis pipeline

Standalone repo for safe CLASSLA reannotation plus the full lexical/syntactic analysis described in the disposition.

The repo is intentionally specific to Šolar. Generic tagging of unrelated corpora can live elsewhere; this project keeps the Solar metadata logic, ID preservation, and thesis-facing outputs in one place.

## What this pipeline does

1. **Reannotates the original Šolar CoNLL-U with CLASSLA** while preserving:
   - original `# sent_id` values, e.g. `solar1.1.1`,
   - original sentence boundaries,
   - original token IDs and token forms,
   - `SpaceAfter=No` and other non-annotation MISC information.

   It overwrites the annotation layers: lemma, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, and NER.

2. **Checks that metadata linkage survives.** Generated IDs such as `solar-orig-1` are treated as a problem, because they break the join to `solar-meta.tsv`.

3. **Runs the analysis requested in the disposition:**
   - lexical diversity / lexical richness: TTR variants, MATTR, MTLD, HD-D, Maas,
   - lexical density: content-word share based on UPOS,
   - lexical sophistication: rare/low-frequency lemma shares using an external reference frequency list when supplied,
   - syntactic diversity: dependency relation diversity, dependency distance, tree depth, clause/subordination proxies,
   - rarest lemmas by region, contact zone, grade, year, school level, and text type,
   - MWE/collocation candidate extraction with n-grams and association scores,
   - basic non-parametric statistics and plots,
   - a Markdown report and an MWE manual-review sheet.

## Repository layout

```text
solar3-lexical-analysis/
├── configs/                         # JSON config for one-command pipeline runs
├── data/
│   ├── raw/                         # put solar-orig.conllu here
│   ├── metadata/solar-meta.tsv      # included metadata table
│   └── annotated/                   # generated solar-classla.conllu
├── analysis/                        # generated tables/plots/report
├── reports/                         # reannotation diff, errors, validation, summary
├── scripts/
│   ├── run_solar_pipeline.py        # full workflow runner
│   ├── solar_reannotate.py          # ID-safe CLASSLA reannotation
│   ├── solar_analysis.py            # thesis-facing quantitative analysis
│   ├── build_report.py              # Markdown report builder
│   ├── prepare_mwe_review.py        # manual review sheet for MWE candidates
│   └── check_solar_conllu.py        # quick CoNLL-U/metadata sanity checker
├── src/solar3_lexical_analysis/     # shared CLASSLA helpers
├── tests/                           # small smoke tests
└── docs/                            # disposition and methodology notes
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

CLASSLA models can be shared across projects. The default config uses:

```text
~/classla_resources_shared
```

## Input files

Put the original corpus here:

```text
data/raw/solar-orig.conllu
```

The metadata table is already included here:

```text
data/metadata/solar-meta.tsv
```

The original `solar-orig.conllu` is not bundled in this zip because it was not part of the current upload.

## One-command run

```bash
python scripts/run_solar_pipeline.py \
  --models-dir ~/classla_resources_shared \
  --download-models
```

To run without downloading models, omit `--download-models` once the models exist.

## Step-by-step run

### 1. Check the original file

```bash
python scripts/check_solar_conllu.py \
  --input data/raw/solar-orig.conllu \
  --metadata data/metadata/solar-meta.tsv \
  --fail-on-generated-ids
```

### 2. Reannotate with CLASSLA

```bash
python scripts/solar_reannotate.py \
  --input data/raw/solar-orig.conllu \
  --output data/annotated/solar-classla.conllu \
  --metadata data/metadata/solar-meta.tsv \
  --metadata-id-columns OrigID,CorrID \
  --metadata-strip-suffix-regex '[st]$' \
  --min-metadata-match-ratio 0.95 \
  --require-sent-id \
  --require-clean-output \
  --diff-report reports/solar-classla-diff.tsv \
  --summary reports/solar-classla-summary.json \
  --errors reports/solar-classla-errors.tsv \
  --validation-report reports/solar-classla-validation.tsv \
  --models-dir ~/classla_resources_shared \
  --download-models
```

### 3. Run the analysis

```bash
python scripts/solar_analysis.py \
  --input data/annotated/solar-classla.conllu \
  --metadata data/metadata/solar-meta.tsv \
  --output-dir analysis \
  --compute-mwe \
  --run-stats \
  --run-models \
  --make-plots
```

This analysis step is CPU-bound. The heavy work here is pandas/statsmodels plus exploratory MWE extraction, so a quiet terminal or low GPU utilization is expected unless you are running the separate CLASSLA reannotation stage.

Optional lexical sophistication reference list:

```bash
python scripts/solar_analysis.py \
  --input data/annotated/solar-classla.conllu \
  --metadata data/metadata/solar-meta.tsv \
  --reference-freq path/to/reference_frequency.tsv \
  --output-dir analysis \
  --compute-mwe \
  --run-stats \
  --run-models \
  --make-plots
```

The reference list should contain a lemma-like column (`lemma`, `lem`, `word`, `token`, or `entry`) and either a rank-like column (`rank`, `freq_rank`, `frequency_rank`) or a frequency-like column (`freq`, `frequency`, `count`, `tokens`, `occurrences`).

### 4. Build a report and MWE review sheet

```bash
python scripts/build_report.py \
  --analysis-dir analysis \
  --output analysis/solar_analysis_report.md

python scripts/prepare_mwe_review.py \
  --analysis-dir analysis \
  --output analysis/mwe/mwe_manual_review.tsv
```

## Key outputs

```text
reports/solar-classla-summary.json
reports/solar-classla-validation.tsv
reports/solar-classla-diff.tsv
analysis/corpus_summary.json
analysis/document_metrics.tsv
analysis/summaries/*.tsv
analysis/rare_words/*.tsv
analysis/mwe/*.tsv
analysis/stats/*.tsv
analysis/plots/*.png
analysis/solar_analysis_report.md
```

## Sanity check after reannotation

```bash
grep -m 5 '^# sent_id' data/annotated/solar-classla.conllu
```

Good output looks like:

```conllu
# sent_id = solar1.1.1
# sent_id = solar1.2.1
```

Bad output looks like:

```conllu
# sent_id = solar-orig-1
```

If you see generated IDs, discard that output and rerun `scripts/solar_reannotate.py` from the original CoNLL-U.

## Notes on interpretation

The contact-zone labels in `scripts/solar_analysis.py` are coarse triage labels for qualitative MWE inspection, not dialectological conclusions. Edit `infer_contact_zone()` after checking the actual metadata distribution and the thesis framing.

The MWE extraction is intentionally exploratory. Treat the output as a ranked candidate list that still needs manual validation.
