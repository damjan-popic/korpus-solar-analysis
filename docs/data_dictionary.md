# Data dictionary for main generated tables

## `analysis/document_metrics.tsv`

| Column | Meaning |
|---|---|
| `doc_id` | Šolar document ID derived from original `sent_id`, e.g. `solar1`. |
| `n_word_tokens` | Wordlike tokens excluding punctuation/symbols. |
| `n_sentences` | Number of sentence blocks in the document. |
| `avg_sentence_length` | Mean wordlike sentence length. |
| `lemma_ttr` | Lemma type-token ratio. Length-sensitive; use carefully. |
| `lemma_mattr_50` | Moving-average TTR with window 50. |
| `lemma_mtld` | Measure of Textual Lexical Diversity. |
| `lemma_hdd` | Hypergeometric distribution diversity. |
| `lexical_density` | Share of content-word tokens among wordlike tokens. |
| `rare_lemma_token_share_top3000` | Token share of lemmas outside top 3000 in the reference list. |
| `mean_ref_rank_tokens` | Mean reference rank across lemma tokens. |
| `deprel_entropy` | Shannon entropy of dependency relations. |
| `mean_dependency_distance` | Mean absolute token-head distance. |
| `subordination_ratio` | Subordinate-clause dependency heads divided by clause heads. |
| `region`, `grade`, `school_level`, `text_type`, `year_start` | Metadata variables. |

## `analysis/mwe/*.tsv`

| Column | Meaning |
|---|---|
| `ngram` | Lemmatized candidate expression. |
| `upos_pattern` | UPOS sequence of candidate. |
| `count` | Frequency in the group. |
| `doc_count` | Number of documents in which candidate occurs. |
| `pmi` | Pointwise mutual information. Higher favors stronger association but may overrate rare items. |
| `dice` | Dice association score. |
| `t_score` | Bigram t-score; blank for n > 2. |

## `analysis/mwe/mwe_manual_review.tsv`

Blank human-review columns:

- `review_status`
- `is_valid_mwe`
- `is_formulaic`
- `possible_contact_language`
- `style_register_note`
- `review_comment`
