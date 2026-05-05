# Configuration

`solar_pipeline.example.json` is the default config for `scripts/run_solar_pipeline.py`.

Edit these keys most often:

- `models_dir`: shared CLASSLA model directory.
- `reannotation.input`: original Šolar CoNLL-U path.
- `analysis.reference_freq`: optional external lemma frequency list, e.g. a Sloleks/Sloleks-derived TSV with `lemma` plus either `rank` or `frequency`.
- `analysis.mwe_min_count`: increase for cleaner MWE candidates, decrease for exploratory work.
