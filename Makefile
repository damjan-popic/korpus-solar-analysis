.PHONY: setup check reannotate analyse report mwe-review pipeline test clean

MODELS_DIR ?= $(HOME)/classla_resources_shared
RAW ?= data/raw/solar-orig.conllu
ANNOTATED ?= data/annotated/solar-classla.conllu
META ?= data/metadata/solar-meta.tsv
ANALYSIS ?= analysis

setup:
	python -m pip install -r requirements.txt

check:
	python scripts/check_solar_conllu.py --input $(RAW) --metadata $(META) --fail-on-generated-ids

reannotate:
	python scripts/solar_reannotate.py \
	  --input $(RAW) \
	  --output $(ANNOTATED) \
	  --metadata $(META) \
	  --metadata-id-columns OrigID,CorrID \
	  --metadata-strip-suffix-regex '[st]$$' \
	  --min-metadata-match-ratio 0.95 \
	  --require-sent-id \
	  --require-clean-output \
	  --diff-report reports/solar-classla-diff.tsv \
	  --summary reports/solar-classla-summary.json \
	  --errors reports/solar-classla-errors.tsv \
	  --validation-report reports/solar-classla-validation.tsv \
	  --models-dir $(MODELS_DIR)

analyse:
	python scripts/solar_analysis.py \
	  --input $(ANNOTATED) \
	  --metadata $(META) \
	  --output-dir $(ANALYSIS) \
	  --compute-mwe \
	  --run-stats \
	  --make-plots

report:
	python scripts/build_report.py --analysis-dir $(ANALYSIS) --output $(ANALYSIS)/solar_analysis_report.md

mwe-review:
	python scripts/prepare_mwe_review.py --analysis-dir $(ANALYSIS) --output $(ANALYSIS)/mwe/mwe_manual_review.tsv

pipeline:
	python scripts/run_solar_pipeline.py --models-dir $(MODELS_DIR)

test:
	pytest -q

clean:
	rm -rf analysis/* reports/* data/annotated/solar-classla.conllu .pytest_cache
