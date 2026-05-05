# Methodology encoded from the disposition

This repo operationalizes the dissertation plan around the following questions:

1. How do lexical diversity and lexical density differ by region and age/school group?
2. How does lexical sophistication differ by region and age/school group?
3. Where do formulaic or established multiword units appear in Šolar?
4. Are multiword units regionally marked, especially in border/contact regions?

## Variables

Main explanatory/grouping variables:

- `region`
- `contact_zone` derived from region for qualitative triage
- `grade` and `grade_num`
- `school_level`, inferred as `osnovna`, `srednja`, or `neznano`
- `year_start`, derived from metadata date/year
- `text_type`

Controls noted in the disposition and represented in outputs:

- document length: `n_word_tokens`, `n_sentences`, `avg_sentence_length`
- text type/genre: `text_type`
- collection year: `year_start`
- school level/grade: `school_level`, `grade`, `grade_num`

## Lexical diversity / pestrost

Implemented document-level metrics:

- `lemma_ttr`, `lemma_rttr`, `lemma_cttr`
- `lemma_mattr_50`
- `lemma_mtld`
- `lemma_hdd`
- `maas_lemma`
- `content_lemma_ttr`

These are computed on normalized lemmas for alphabetic word tokens.

## Lexical density / gostota

Implemented as:

```text
content-word tokens / wordlike tokens
```

Content UPOS are:

```text
ADJ, ADV, NOUN, PROPN, VERB
```

Function words and punctuation are excluded from content count; punctuation/symbols are excluded from wordlike tokens.

## Lexical sophistication / zahtevnost

The script supports an external reference frequency list. When supplied, it computes rare-lemma shares outside:

- top 1,000 lemmas
- top 3,000 lemmas
- top 5,000 lemmas

When no external reference list is supplied, the pipeline falls back to corpus-internal lemma ranks. That fallback is useful for debugging, but the thesis analysis should prefer an external reference list.

## Syntactic diversity

Added because the corpus is reannotated with CLASSLA UD dependencies. Document-level outputs include:

- `deprel_ttr`
- `deprel_entropy`
- `mean_dependency_distance`
- `max_dependency_distance`
- `mean_tree_depth`
- `max_tree_depth`
- `clause_density`
- `subordination_ratio`
- UPOS bigram/trigram diversity

## Multiword units / ustaljene večbesedne enote

The script extracts lemma n-grams from 2 to 4 tokens by default, filters candidates to contain alphabetic material and at least one content word, and ranks candidates with:

- raw count
- document count
- PMI
- Dice
- t-score for bigrams

The candidate tables are grouped by region, contact zone, grade, year, and school level. Use `scripts/prepare_mwe_review.py` to produce a manual-validation TSV.

## Outputs useful for thesis writing

- `analysis/document_metrics.tsv`: modeling table, one row per document.
- `analysis/summaries/summary_by_region.tsv`: regional comparison.
- `analysis/summaries/summary_by_grade.tsv`: developmental comparison.
- `analysis/summaries/summary_by_school_level.tsv`: osnovna vs. srednja.
- `analysis/rare_words/rarest_lemmas_by_region.tsv`: regional rare vocabulary.
- `analysis/mwe/mwe_candidates_by_contact_zone.tsv`: contact-zone MWE candidates.
- `analysis/stats/*.tsv`: non-parametric test outputs.
- `analysis/models/ols_control_models.tsv`: exploratory control models with document length/text type/year/region variables.
- `analysis/solar_analysis_report.md`: compact auto-report.
