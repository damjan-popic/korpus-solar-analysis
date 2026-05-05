#!/usr/bin/env python3
"""Corpus analysis pipeline for Šolar / SOLAR-style CoNLL-U data.

The script is built around the research directions described in the student's
project outline:
- lexical diversity / lexical richness
- lexical density
- lexical sophistication (rare / low-frequency lemmas)
- syntactic diversity
- rarest words by region / year / grade / school level
- automatic MWE candidate extraction with association scores
- aggregation by metadata variables and basic statistical tests

Typical usage
-------------
    python scripts/solar_analysis.py \
        --input data/solar-classla.conllu \
        --metadata data/solar-meta.tsv \
        --output-dir analysis

Optional reference list for lexical sophistication
--------------------------------------------------
If you have a frequency list (for example, an exported Sloleks frequency file),
pass it via `--reference-freq`. The script will try to auto-detect the lemma and
frequency/rank columns. If no reference list is supplied, the script falls back
to the corpus-internal lemma frequency ranking.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TOKEN_COLUMNS: Tuple[str, ...] = (
    "ID",
    "FORM",
    "LEMMA",
    "UPOS",
    "XPOS",
    "FEATS",
    "HEAD",
    "DEPREL",
    "DEPS",
    "MISC",
)

WORD_ID_RE = re.compile(r"^\d+$")
DOC_FROM_SENT_RE = re.compile(r"^([^\.]+)")
CONTENT_UPOS = {"ADJ", "ADV", "NOUN", "PROPN", "VERB"}
WORDLIKE_UPOS_EXCLUDE = {"PUNCT", "SYM", "X"}
CLAUSE_HEAD_DEPRELS = {
    "root",
    "ccomp",
    "xcomp",
    "advcl",
    "acl",
    "acl:relcl",
    "csubj",
    "csubj:pass",
    "parataxis",
    "conj",
}
SUBORDINATE_DEPRELS = {
    "acl",
    "acl:relcl",
    "advcl",
    "ccomp",
    "csubj",
    "csubj:pass",
    "xcomp",
}
SUMMARY_METRICS = [
    "n_tokens",
    "n_word_tokens",
    "n_sentences",
    "avg_sentence_length",
    "lemma_ttr",
    "lemma_rttr",
    "lemma_cttr",
    "lemma_mattr_50",
    "lemma_mtld",
    "lemma_hdd",
    "maas_lemma",
    "lexical_density",
    "rare_lemma_token_share_top1000",
    "rare_lemma_token_share_top3000",
    "rare_lemma_token_share_top5000",
    "rare_lemma_type_share_top1000",
    "rare_lemma_type_share_top3000",
    "rare_lemma_type_share_top5000",
    "mean_ref_rank_tokens",
    "mean_ref_rank_types",
    "deprel_ttr",
    "deprel_entropy",
    "mean_dependency_distance",
    "max_dependency_distance",
    "mean_tree_depth",
    "max_tree_depth",
    "clause_density",
    "subordination_ratio",
    "upos_bigram_ttr",
    "upos_trigram_ttr",
    "entity_token_share",
]
PLOT_METRICS = [
    "lemma_mattr_50",
    "lexical_density",
    "rare_lemma_token_share_top3000",
    "deprel_entropy",
]
MWE_META_COLUMNS: Tuple[str, ...] = (
    "region",
    "grade",
    "grade_num",
    "year_start",
    "date_label",
    "school_level",
    "text_type",
    "contact_zone",
    "school",
)


@dataclass
class ConlluRow:
    values: Dict[str, str]

    @property
    def raw_id(self) -> str:
        return self.values["ID"]

    @property
    def is_word(self) -> bool:
        return bool(WORD_ID_RE.fullmatch(self.raw_id))


@dataclass
class ConlluSentence:
    comments: List[str] = field(default_factory=list)
    rows: List[ConlluRow] = field(default_factory=list)

    def get_comment_value(self, key: str) -> Optional[str]:
        prefix = f"# {key} = "
        for line in self.comments:
            if line.startswith(prefix):
                return line[len(prefix) :].strip()
        return None

    @property
    def sent_id(self) -> Optional[str]:
        return self.get_comment_value("sent_id")

    @property
    def text(self) -> Optional[str]:
        return self.get_comment_value("text")

    @property
    def doc_id(self) -> Optional[str]:
        sent_id = self.sent_id
        if not sent_id:
            return None
        match = DOC_FROM_SENT_RE.match(sent_id)
        return match.group(1) if match else sent_id

    @property
    def word_rows(self) -> List[ConlluRow]:
        return [row for row in self.rows if row.is_word]


@dataclass
class SentenceRecord:
    meta: Dict[str, object]
    tokens: List[Tuple[str, str]]



def parse_conllu_text(text: str) -> List[ConlluSentence]:
    text = text.strip("\n")
    if not text:
        return []
    sentences: List[ConlluSentence] = []
    for block in re.split(r"\n\s*\n", text):
        comments: List[str] = []
        rows: List[ConlluRow] = []
        for line in block.splitlines():
            if not line.strip():
                continue
            if line.startswith("#"):
                comments.append(line.rstrip())
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(TOKEN_COLUMNS):
                raise ValueError(
                    f"Expected {len(TOKEN_COLUMNS)} columns, got {len(parts)} in line: {line!r}"
                )
            rows.append(ConlluRow(dict(zip(TOKEN_COLUMNS, parts))))
        if comments or rows:
            sentences.append(ConlluSentence(comments=comments, rows=rows))
    return sentences


def read_conllu(path: Path) -> List[ConlluSentence]:
    return parse_conllu_text(path.read_text(encoding="utf-8"))



def parse_misc(misc_value: str) -> Dict[str, str]:
    misc_value = (misc_value or "_").strip()
    if not misc_value or misc_value == "_":
        return {}
    output: Dict[str, str] = {}
    for part in misc_value.split("|"):
        if not part:
            continue
        if "=" in part:
            key, value = part.split("=", 1)
            output[key] = value
        else:
            output[part] = "True"
    return output



def safe_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None



def has_alpha(text: str) -> bool:
    return any(char.isalpha() for char in text)


def log_progress(message: str) -> None:
    print(f"[solar_analysis] {message}", file=sys.stderr, flush=True)



def normalize_lemma(text: str) -> str:
    text = (text or "").strip()
    return text.lower()



def parse_year_start(value: object) -> Optional[int]:
    text = str(value).strip()
    match = re.search(r"(\d{4})", text)
    return int(match.group(1)) if match else None



def parse_grade_num(value: object) -> Optional[int]:
    text = str(value).strip().lower()
    match = re.search(r"(\d+)", text)
    if match:
        return int(match.group(1))
    return None



def infer_school_level(school: object, grade: object) -> str:
    school_text = str(school).strip().lower()
    grade_text = str(grade).strip().lower()
    if "osnovna" in school_text or "razred" in grade_text:
        return "osnovna"
    if (
        "gimnazija" in school_text
        or "poklicna" in school_text
        or "strokovna" in school_text
        or "letnik" in grade_text
        or "maturitetni" in grade_text
    ):
        return "srednja"
    return "neznano"


def infer_contact_zone(region: object) -> str:
    """Coarse border/contact-zone label used for qualitative MWE triage.

    The mapping is intentionally transparent and conservative: it is not a
    dialectological claim, just a reproducible grouping that can be edited after
    the metadata distribution is inspected. Unknown regions stay ``neznano``.
    """
    text = str(region or "").strip().lower()
    if not text:
        return "neznano"
    italy = {"gorica", "nova gorica", "koper", "obala", "primorska", "trst"}
    austria = {"maribor", "slovenj gradec", "ravne", "koroška", "murska sobota"}
    croatia = {"novo mesto", "krško", "brežice", "metlika", "črnomelj", "kočevje"}
    if any(item in text for item in italy):
        return "italija"
    if any(item in text for item in austria):
        return "avstrija"
    if any(item in text for item in croatia):
        return "hrvaška"
    return "notranjost"



def shannon_entropy(items: Sequence[str]) -> float:
    if not items:
        return float("nan")
    counts = Counter(items)
    total = sum(counts.values())
    probs = np.array([count / total for count in counts.values()], dtype=float)
    return float(-(probs * np.log2(probs)).sum())



def ttr(sequence: Sequence[str]) -> float:
    if not sequence:
        return float("nan")
    return len(set(sequence)) / len(sequence)



def rttr(sequence: Sequence[str]) -> float:
    if not sequence:
        return float("nan")
    return len(set(sequence)) / math.sqrt(len(sequence))



def cttr(sequence: Sequence[str]) -> float:
    if not sequence:
        return float("nan")
    return len(set(sequence)) / math.sqrt(2 * len(sequence))



def maas_ttr(sequence: Sequence[str]) -> float:
    if not sequence:
        return float("nan")
    types = len(set(sequence))
    tokens = len(sequence)
    if tokens <= 1 or types <= 0:
        return float("nan")
    return (math.log(tokens) - math.log(types)) / (math.log(tokens) ** 2)



def mattr(sequence: Sequence[str], window_size: int = 50) -> float:
    if not sequence:
        return float("nan")
    if len(sequence) <= window_size:
        return ttr(sequence)
    values: List[float] = []
    for start in range(0, len(sequence) - window_size + 1):
        window = sequence[start : start + window_size]
        values.append(len(set(window)) / window_size)
    return float(np.mean(values)) if values else float("nan")



def mtld_one_pass(sequence: Sequence[str], threshold: float = 0.72) -> float:
    if not sequence:
        return float("nan")
    factors = 0.0
    factor_tokens: List[str] = []
    seen: set[str] = set()

    for token in sequence:
        factor_tokens.append(token)
        seen.add(token)
        current_ttr = len(seen) / len(factor_tokens)
        if current_ttr <= threshold:
            factors += 1.0
            factor_tokens = []
            seen = set()

    if factor_tokens:
        current_ttr = len(seen) / len(factor_tokens)
        if current_ttr == 1.0:
            partial = 0.0
        else:
            partial = (1.0 - current_ttr) / (1.0 - threshold)
        factors += partial

    if factors == 0:
        return float(len(sequence))
    return len(sequence) / factors



def mtld(sequence: Sequence[str], threshold: float = 0.72) -> float:
    if not sequence:
        return float("nan")
    forward = mtld_one_pass(sequence, threshold=threshold)
    backward = mtld_one_pass(list(reversed(sequence)), threshold=threshold)
    return float(np.mean([forward, backward]))



def log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)



def hdd(sequence: Sequence[str], sample_size: int = 42) -> float:
    if not sequence:
        return float("nan")
    population = len(sequence)
    sample = min(sample_size, population)
    if sample <= 0:
        return float("nan")
    counts = Counter(sequence)
    denominator_log = log_comb(population, sample)
    contributions: List[float] = []
    for freq in counts.values():
        if population - freq < sample:
            prob_zero = 0.0
        else:
            prob_zero = math.exp(log_comb(population - freq, sample) - denominator_log)
        contributions.append((1.0 - prob_zero) / sample)
    return float(sum(contributions))



def pos_ngram_ttr(sequence: Sequence[str], n: int) -> float:
    if len(sequence) < n or n <= 0:
        return float("nan")
    grams = [tuple(sequence[i : i + n]) for i in range(0, len(sequence) - n + 1)]
    return len(set(grams)) / len(grams)



def compute_tree_depths(ids: Sequence[int], heads: Dict[int, int]) -> Dict[int, int]:
    memo: Dict[int, int] = {}

    def depth(node_id: int, trail: set[int]) -> int:
        if node_id in memo:
            return memo[node_id]
        if node_id in trail:
            memo[node_id] = 0
            return 0
        head = heads.get(node_id, 0)
        if head == 0 or head not in heads:
            memo[node_id] = 1
            return 1
        memo[node_id] = depth(head, trail | {node_id}) + 1
        return memo[node_id]

    for node_id in ids:
        depth(node_id, set())
    return memo



def load_metadata(path: Path) -> pd.DataFrame:
    meta = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    rename_map = {
        "SolarID": "solar_id",
        "OrigID": "orig_id",
        "CorrID": "corr_id",
        "Date": "date_label",
        "School": "school",
        "Subject": "subject",
        "Grade": "grade",
        "TextType": "text_type",
        "Region": "region",
    }
    meta = meta.rename(columns=rename_map)
    if "orig_id" not in meta.columns:
        raise ValueError(f"Metadata file {path} must contain the 'OrigID' column.")

    meta["doc_id"] = meta["orig_id"].str.replace(r"[st]$", "", regex=True)
    meta["year_start"] = meta["date_label"].map(parse_year_start)
    meta["grade_num"] = meta["grade"].map(parse_grade_num)
    meta["school_level"] = [infer_school_level(s, g) for s, g in zip(meta["school"], meta["grade"])]
    meta["contact_zone"] = meta["region"].map(infer_contact_zone) if "region" in meta.columns else "neznano"
    return meta



def build_token_dataframe(sentences: Sequence[ConlluSentence]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for sent_index, sentence in enumerate(sentences):
        sent_id = sentence.sent_id or f"sent_{sent_index + 1}"
        doc_id = sentence.doc_id or sent_id
        words = sentence.word_rows
        for token_index, row in enumerate(words, start=1):
            misc = parse_misc(row.values["MISC"])
            token_id = safe_int(row.values["ID"])
            head = safe_int(row.values["HEAD"]) or 0
            lemma = row.values["LEMMA"] if row.values["LEMMA"] != "_" else row.values["FORM"]
            upos = row.values["UPOS"]
            rows.append(
                {
                    "doc_id": doc_id,
                    "sent_id": sent_id,
                    "sent_index_global": sent_index,
                    "token_index_in_sentence": token_index,
                    "token_id": token_id,
                    "form": row.values["FORM"],
                    "lemma": lemma,
                    "lemma_norm": normalize_lemma(lemma),
                    "form_norm": normalize_lemma(row.values["FORM"]),
                    "upos": upos,
                    "xpos": row.values["XPOS"],
                    "feats": row.values["FEATS"],
                    "head": head,
                    "deprel": row.values["DEPREL"],
                    "deps": row.values["DEPS"],
                    "misc": row.values["MISC"],
                    "ner": misc.get("NER", "O"),
                    "is_wordlike": upos not in WORDLIKE_UPOS_EXCLUDE,
                    "is_content": upos in CONTENT_UPOS,
                    "has_alpha": has_alpha(lemma),
                    "is_entity": misc.get("NER", "O") != "O",
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No word tokens found in the CoNLL-U file.")
    return df



def build_sentence_dataframe(token_df: pd.DataFrame) -> pd.DataFrame:
    sentence_rows: List[Dict[str, object]] = []
    for sent_id, group in token_df.groupby("sent_id", sort=False):
        group = group.sort_values("token_index_in_sentence")
        wordlike = group[group["is_wordlike"]]
        sentence_rows.append(
            {
                "sent_id": sent_id,
                "doc_id": group["doc_id"].iloc[0],
                "sent_index_global": group["sent_index_global"].iloc[0],
                "n_tokens": len(group),
                "n_word_tokens": len(wordlike),
                "upos_sequence": list(wordlike["upos"]),
                "lemma_sequence": list(wordlike["lemma_norm"]),
            }
        )
    return pd.DataFrame(sentence_rows)



def looks_like_placeholder_path(path: Path) -> bool:
    """Return True for documentation placeholders that users often copy verbatim."""
    normalized = str(path).replace("\\", "/").strip().lower()
    placeholder_bits = (
        "path/to/",
        "path/to/reference",
        "your/path",
        "example",
        "placeholder",
    )
    return any(bit in normalized for bit in placeholder_bits)


def maybe_load_reference_frequency(path: Optional[Path], *, strict: bool = False) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    path = Path(path).expanduser()
    if not path.exists():
        message = (
            f"Reference frequency file not found: {path}. "
            "Continuing with corpus-internal lemma frequencies for lexical sophistication. "
            "Omit --reference-freq unless you have a real TSV/CSV frequency list, or pass "
            "--strict-reference-freq if this should be a hard error."
        )
        if strict and not looks_like_placeholder_path(path):
            raise FileNotFoundError(message)
        print(f"[solar_analysis] WARNING: {message}", file=sys.stderr)
        return None

    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(path, sep=sep)
    if df.empty:
        raise ValueError(f"Reference frequency file is empty: {path}")

    lower_cols = {col.lower(): col for col in df.columns}
    lemma_candidates = ["lemma", "lem", "word", "token", "entry"]
    freq_candidates = ["freq", "frequency", "count", "tokens", "occurrences"]
    rank_candidates = ["rank", "freq_rank", "frequency_rank", "order"]

    lemma_col = next((lower_cols[c] for c in lemma_candidates if c in lower_cols), None)
    rank_col = next((lower_cols[c] for c in rank_candidates if c in lower_cols), None)
    freq_col = next((lower_cols[c] for c in freq_candidates if c in lower_cols), None)

    if lemma_col is None:
        raise ValueError(
            f"Could not detect a lemma column in reference frequency file {path}. Columns: {list(df.columns)}"
        )

    out = pd.DataFrame({"lemma_norm": df[lemma_col].astype(str).map(normalize_lemma)})
    if rank_col is not None:
        out["ref_rank"] = pd.to_numeric(df[rank_col], errors="coerce")
    elif freq_col is not None:
        out["ref_freq"] = pd.to_numeric(df[freq_col], errors="coerce")
        out = out.sort_values(["ref_freq", "lemma_norm"], ascending=[False, True])
        out["ref_rank"] = np.arange(1, len(out) + 1)
    else:
        raise ValueError(
            f"Reference frequency file {path} must contain either a rank-like column or a frequency-like column."
        )

    if freq_col is not None and "ref_freq" not in out.columns:
        out["ref_freq"] = pd.to_numeric(df[freq_col], errors="coerce")
    out = out.dropna(subset=["lemma_norm", "ref_rank"])
    out = out.sort_values(["ref_rank", "lemma_norm"]).drop_duplicates(subset=["lemma_norm"], keep="first")
    out["ref_rank"] = out["ref_rank"].astype(int)
    return out[[col for col in ["lemma_norm", "ref_rank", "ref_freq"] if col in out.columns]]



def build_corpus_frequency_reference(token_df: pd.DataFrame) -> pd.DataFrame:
    ref = (
        token_df[token_df["is_wordlike"] & token_df["has_alpha"]]
        .groupby("lemma_norm", dropna=False)
        .size()
        .reset_index(name="ref_freq")
        .sort_values(["ref_freq", "lemma_norm"], ascending=[False, True])
        .reset_index(drop=True)
    )
    ref["ref_rank"] = np.arange(1, len(ref) + 1)
    return ref[["lemma_norm", "ref_rank", "ref_freq"]]



def sequence_ref_ranks(sequence: Sequence[str], ref_rank_map: Dict[str, int], default_rank: int) -> List[int]:
    return [int(ref_rank_map.get(item, default_rank)) for item in sequence if item]



def share_above_rank(ranks: Sequence[int], top_n: int) -> float:
    if not ranks:
        return float("nan")
    return float(np.mean([rank > top_n for rank in ranks]))



def compute_document_metrics(
    token_df: pd.DataFrame,
    sentence_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    reference_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if reference_df is None:
        reference_df = build_corpus_frequency_reference(token_df)
        reference_source = "corpus_internal"
    else:
        reference_source = "external"

    ref_rank_map = dict(zip(reference_df["lemma_norm"], reference_df["ref_rank"]))
    ref_freq_map = dict(zip(reference_df["lemma_norm"], reference_df.get("ref_freq", pd.Series(dtype=float))))
    default_rank = int(reference_df["ref_rank"].max()) + 1 if not reference_df.empty else 1

    document_rows: List[Dict[str, object]] = []
    sentence_lengths = sentence_df.set_index("sent_id")["n_word_tokens"].to_dict()

    for doc_id, group in token_df.groupby("doc_id", sort=False):
        group = group.sort_values(["sent_index_global", "token_index_in_sentence"])
        wordlike = group[group["is_wordlike"]]
        alpha_wordlike = wordlike[wordlike["has_alpha"]]
        content = wordlike[wordlike["is_content"]]

        lemma_seq = list(alpha_wordlike["lemma_norm"])
        form_seq = list(alpha_wordlike["form_norm"])
        content_lemma_seq = list(content[content["has_alpha"]]["lemma_norm"])
        deprels = list(wordlike[wordlike["deprel"].astype(str) != "punct"]["deprel"])

        dependency_distances: List[int] = []
        tree_depths: List[int] = []
        clause_heads = 0
        subordinate_heads = 0

        for sent_id, sent_group in group.groupby("sent_id", sort=False):
            sent_group = sent_group.sort_values("token_index_in_sentence")
            sent_wordlike = sent_group[sent_group["is_wordlike"]].copy()
            ids = [int(x) for x in sent_wordlike["token_id"].dropna().astype(int).tolist()]
            heads = {
                int(row.token_id): int(row.head)
                for row in sent_wordlike.itertuples()
                if pd.notna(row.token_id) and row.head is not None
            }
            if ids:
                depth_map = compute_tree_depths(ids, heads)
                tree_depths.extend(depth_map.values())

            for row in sent_wordlike.itertuples():
                if row.deprel == "punct" or row.head == 0 or row.token_id is None:
                    continue
                dependency_distances.append(abs(int(row.token_id) - int(row.head)))

            clause_heads += sum(
                1
                for row in sent_wordlike.itertuples()
                if str(row.deprel).lower() in CLAUSE_HEAD_DEPRELS or int(row.head) == 0
            )
            subordinate_heads += sum(
                1 for row in sent_wordlike.itertuples() if str(row.deprel).lower() in SUBORDINATE_DEPRELS
            )

        sent_ids = list(group["sent_id"].drop_duplicates())
        sent_lengths = [sentence_lengths.get(sent_id, np.nan) for sent_id in sent_ids]
        upos_seq = list(wordlike["upos"])
        ref_ranks_tokens = sequence_ref_ranks(lemma_seq, ref_rank_map, default_rank)
        ref_ranks_types = sequence_ref_ranks(sorted(set(lemma_seq)), ref_rank_map, default_rank)

        entity_token_share = float(group["is_entity"].mean()) if len(group) else float("nan")
        lexical_density = (
            len(content) / len(wordlike) if len(wordlike) else float("nan")
        )

        metadata = (
            meta_df[meta_df["doc_id"] == doc_id].iloc[0].to_dict()
            if not meta_df.empty and (meta_df["doc_id"] == doc_id).any()
            else {}
        )

        row: Dict[str, object] = {
            "doc_id": doc_id,
            "reference_source": reference_source,
            "n_tokens": int(len(group)),
            "n_word_tokens": int(len(wordlike)),
            "n_alpha_word_tokens": int(len(alpha_wordlike)),
            "n_content_tokens": int(len(content)),
            "n_sentences": int(len(sent_ids)),
            "avg_sentence_length": float(np.nanmean(sent_lengths)) if sent_lengths else float("nan"),
            "std_sentence_length": float(np.nanstd(sent_lengths, ddof=1)) if len(sent_lengths) > 1 else float("nan"),
            "form_ttr": ttr(form_seq),
            "lemma_ttr": ttr(lemma_seq),
            "lemma_rttr": rttr(lemma_seq),
            "lemma_cttr": cttr(lemma_seq),
            "lemma_mattr_50": mattr(lemma_seq, window_size=50),
            "lemma_mtld": mtld(lemma_seq),
            "lemma_hdd": hdd(lemma_seq),
            "maas_lemma": maas_ttr(lemma_seq),
            "content_lemma_ttr": ttr(content_lemma_seq),
            "lexical_density": lexical_density,
            "mean_ref_rank_tokens": float(np.mean(ref_ranks_tokens)) if ref_ranks_tokens else float("nan"),
            "median_ref_rank_tokens": float(np.median(ref_ranks_tokens)) if ref_ranks_tokens else float("nan"),
            "mean_ref_rank_types": float(np.mean(ref_ranks_types)) if ref_ranks_types else float("nan"),
            "median_ref_rank_types": float(np.median(ref_ranks_types)) if ref_ranks_types else float("nan"),
            "rare_lemma_token_share_top1000": share_above_rank(ref_ranks_tokens, 1000),
            "rare_lemma_token_share_top3000": share_above_rank(ref_ranks_tokens, 3000),
            "rare_lemma_token_share_top5000": share_above_rank(ref_ranks_tokens, 5000),
            "rare_lemma_type_share_top1000": share_above_rank(ref_ranks_types, 1000),
            "rare_lemma_type_share_top3000": share_above_rank(ref_ranks_types, 3000),
            "rare_lemma_type_share_top5000": share_above_rank(ref_ranks_types, 5000),
            "oov_reference_token_share": float(np.mean([rank == default_rank for rank in ref_ranks_tokens]))
            if ref_ranks_tokens
            else float("nan"),
            "deprel_ttr": ttr(deprels),
            "deprel_entropy": shannon_entropy(deprels),
            "mean_dependency_distance": float(np.mean(dependency_distances))
            if dependency_distances
            else float("nan"),
            "max_dependency_distance": float(np.max(dependency_distances))
            if dependency_distances
            else float("nan"),
            "mean_tree_depth": float(np.mean(tree_depths)) if tree_depths else float("nan"),
            "max_tree_depth": float(np.max(tree_depths)) if tree_depths else float("nan"),
            "clause_density": clause_heads / len(sent_ids) if sent_ids else float("nan"),
            "subordination_ratio": subordinate_heads / clause_heads if clause_heads else float("nan"),
            "upos_bigram_ttr": pos_ngram_ttr(upos_seq, 2),
            "upos_trigram_ttr": pos_ngram_ttr(upos_seq, 3),
            "entity_token_share": entity_token_share,
        }
        row.update(metadata)
        document_rows.append(row)

    doc_df = pd.DataFrame(document_rows)
    if not doc_df.empty and "doc_id" in doc_df.columns:
        doc_df = doc_df.sort_values([col for col in ["year_start", "grade_num", "region", "doc_id"] if col in doc_df.columns])
    return doc_df



def aggregate_summary(doc_df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    available_group_cols = [col for col in group_cols if col in doc_df.columns]
    if not available_group_cols:
        raise ValueError(f"No valid grouping columns found in document dataframe: {group_cols}")

    base = doc_df.copy()
    base = base.dropna(subset=available_group_cols)
    agg_spec: Dict[str, Tuple[str, str]] = {
        "n_docs": ("doc_id", "nunique"),
        "total_tokens": ("n_tokens", "sum"),
        "mean_tokens_per_doc": ("n_tokens", "mean"),
    }
    for metric in SUMMARY_METRICS:
        if metric not in base.columns:
            continue
        agg_spec[f"{metric}_mean"] = (metric, "mean")
        agg_spec[f"{metric}_median"] = (metric, "median")
        agg_spec[f"{metric}_std"] = (metric, "std")

    summary = base.groupby(available_group_cols, dropna=False).agg(**agg_spec).reset_index()
    sort_cols = list(available_group_cols)
    summary = summary.sort_values(sort_cols)
    return summary



def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sep = "\t" if path.suffix.lower() != ".csv" else ","
    df.to_csv(path, sep=sep, index=False)



def build_group_frequency_tables(token_df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    base = token_df[token_df["is_wordlike"] & token_df["has_alpha"]].copy()
    freq = (
        base.groupby("lemma_norm")
        .agg(global_count=("lemma_norm", "size"), global_doc_count=("doc_id", pd.Series.nunique))
        .reset_index()
    )
    merged = freq.merge(reference_df, on="lemma_norm", how="left")
    return merged



def compute_rarest_words_by_group(
    token_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    group_col: str,
    top_n: int,
    min_group_count: int,
) -> pd.DataFrame:
    if group_col not in token_df.columns:
        return pd.DataFrame()

    base = token_df[token_df["is_wordlike"] & token_df["has_alpha"]].copy()
    global_stats = build_group_frequency_tables(token_df, reference_df)
    max_rank = int(reference_df["ref_rank"].max()) if not reference_df.empty else 0

    group_stats = (
        base.groupby([group_col, "lemma_norm"])
        .agg(
            group_count=("lemma_norm", "size"),
            group_doc_count=("doc_id", pd.Series.nunique),
        )
        .reset_index()
    )
    group_stats = group_stats[group_stats["group_count"] >= min_group_count]
    group_stats = group_stats.merge(global_stats, on="lemma_norm", how="left")
    group_stats["ref_rank"] = group_stats["ref_rank"].fillna(max_rank + 1).astype(int)
    group_stats["is_oov_reference"] = group_stats["ref_rank"] > max_rank

    group_totals = (
        base.groupby(group_col)
        .size()
        .reset_index(name="group_total_tokens")
    )
    group_stats = group_stats.merge(group_totals, on=group_col, how="left")
    group_stats["relative_freq_per_million"] = 1_000_000 * group_stats["group_count"] / group_stats["group_total_tokens"]

    group_stats = group_stats.sort_values(
        [group_col, "ref_rank", "group_count", "lemma_norm"],
        ascending=[True, False, False, True],
    )
    top = group_stats.groupby(group_col, dropna=False).head(top_n).reset_index(drop=True)
    return top



def valid_mwe_tokens(tokens: Sequence[Tuple[str, str]]) -> bool:
    if not tokens:
        return False
    if not any(upos in CONTENT_UPOS for _, upos in tokens):
        return False
    return True



def iter_sentence_records(token_df: pd.DataFrame) -> Iterator[SentenceRecord]:
    for sent_id, group in token_df.groupby("sent_id", sort=False):
        group = group.sort_values("token_index_in_sentence")
        meta = {
            "sent_id": sent_id,
            "doc_id": group["doc_id"].iloc[0],
        }
        for col in MWE_META_COLUMNS:
            if col in group.columns:
                meta[col] = group[col].iloc[0]
        usable = group.loc[group["is_wordlike"] & group["has_alpha"], ["lemma_norm", "upos"]]
        tokens = [
            (str(lemma_norm), str(upos))
            for lemma_norm, upos in usable.itertuples(index=False, name=None)
        ]
        yield SentenceRecord(meta=meta, tokens=tokens)


def compute_mwe_candidates_from_records(
    sentence_records: Sequence[SentenceRecord],
    group_col: str,
    min_n: int,
    max_n: int,
    min_count: int,
) -> pd.DataFrame:
    if not sentence_records:
        return pd.DataFrame()
    if not any(group_col in sentence.meta for sentence in sentence_records):
        return pd.DataFrame()

    unigram_counts: Dict[str, Counter] = defaultdict(Counter)
    total_tokens: Counter = Counter()
    ngram_counts: Dict[Tuple[str, int], Counter] = defaultdict(Counter)
    ngram_doc_sets: Dict[Tuple[str, int], Dict[Tuple[str, ...], set[str]]] = defaultdict(lambda: defaultdict(set))
    total_ngrams: Counter = Counter()
    upos_patterns: Dict[Tuple[str, int, Tuple[str, ...]], Tuple[str, ...]] = {}

    for sentence in sentence_records:
        group_value = str(sentence.meta.get(group_col, ""))
        if not group_value or group_value.lower() == "nan" or not sentence.tokens:
            continue
        doc_id = str(sentence.meta.get("doc_id", ""))

        for lemma, _ in sentence.tokens:
            unigram_counts[group_value][lemma] += 1
            total_tokens[group_value] += 1

        for n in range(min_n, max_n + 1):
            if len(sentence.tokens) < n:
                continue
            for start in range(0, len(sentence.tokens) - n + 1):
                token_window = sentence.tokens[start : start + n]
                if not valid_mwe_tokens(token_window):
                    continue
                lemma_ngram = tuple(lemma for lemma, _ in token_window)
                upos_ngram = tuple(upos for _, upos in token_window)
                ngram_counts[(group_value, n)][lemma_ngram] += 1
                ngram_doc_sets[(group_value, n)][lemma_ngram].add(doc_id)
                total_ngrams[(group_value, n)] += 1
                upos_patterns[(group_value, n, lemma_ngram)] = upos_ngram

    rows_out: List[Dict[str, object]] = []
    for (group_value, n), counter in ngram_counts.items():
        total_ngram_count = total_ngrams[(group_value, n)]
        token_total = total_tokens[group_value]
        if total_ngram_count <= 0 or token_total <= 0:
            continue
        for ngram, count in counter.items():
            if count < min_count:
                continue
            unigram_product_prob = 1.0
            unigram_sum = 0
            unigram_freqs = []
            for lemma in ngram:
                freq = unigram_counts[group_value][lemma]
                unigram_freqs.append(freq)
                unigram_sum += freq
                unigram_product_prob *= freq / token_total if token_total else 0.0
            p_ngram = count / total_ngram_count
            if p_ngram > 0 and unigram_product_prob > 0:
                pmi = math.log2(p_ngram / unigram_product_prob)
            else:
                pmi = float("nan")
            dice = (n * count / unigram_sum) if unigram_sum else float("nan")

            t_score = float("nan")
            if n == 2 and token_total > 0:
                expected = (unigram_freqs[0] * unigram_freqs[1]) / token_total
                t_score = (count - expected) / math.sqrt(count) if count > 0 else float("nan")

            rows_out.append(
                {
                    group_col: group_value,
                    "n": n,
                    "ngram": " ".join(ngram),
                    "upos_pattern": " ".join(upos_patterns[(group_value, n, ngram)]),
                    "count": count,
                    "doc_count": len(ngram_doc_sets[(group_value, n)][ngram]),
                    "pmi": pmi,
                    "dice": dice,
                    "t_score": t_score,
                }
            )

    if not rows_out:
        return pd.DataFrame()
    result = pd.DataFrame(rows_out)
    result = result.sort_values([group_col, "n", "count", "pmi", "ngram"], ascending=[True, True, False, False, True])
    return result



def compute_mwe_candidates(
    token_df: pd.DataFrame,
    group_col: str,
    min_n: int,
    max_n: int,
    min_count: int,
) -> pd.DataFrame:
    if group_col not in token_df.columns:
        return pd.DataFrame()
    sentence_records = list(iter_sentence_records(token_df))
    return compute_mwe_candidates_from_records(
        sentence_records=sentence_records,
        group_col=group_col,
        min_n=min_n,
        max_n=max_n,
        min_count=min_count,
    )



def run_statistical_tests(doc_df: pd.DataFrame, output_dir: Path) -> None:
    try:
        from scipy import stats
    except Exception:
        return

    stats_dir = output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    numeric_metrics = [metric for metric in PLOT_METRICS if metric in doc_df.columns]

    spearman_rows: List[Dict[str, object]] = []
    for x_col in ["grade_num", "year_start"]:
        if x_col not in doc_df.columns:
            continue
        for metric in numeric_metrics:
            subset = doc_df[[x_col, metric]].dropna()
            if len(subset) < 10 or subset[x_col].nunique() < 2:
                continue
            rho, p_value = stats.spearmanr(subset[x_col], subset[metric], nan_policy="omit")
            spearman_rows.append(
                {
                    "predictor": x_col,
                    "metric": metric,
                    "n": len(subset),
                    "spearman_rho": rho,
                    "p_value": p_value,
                }
            )
    write_dataframe(pd.DataFrame(spearman_rows), stats_dir / "spearman_progression.tsv")

    kruskal_rows: List[Dict[str, object]] = []
    if "region" in doc_df.columns:
        for metric in numeric_metrics:
            groups = []
            labels = []
            for region, subset in doc_df.groupby("region"):
                values = subset[metric].dropna().values
                if len(values) >= 5:
                    groups.append(values)
                    labels.append(region)
            if len(groups) >= 2:
                h_stat, p_value = stats.kruskal(*groups)
                kruskal_rows.append(
                    {
                        "grouping": "region",
                        "metric": metric,
                        "n_groups": len(groups),
                        "groups": " | ".join(labels),
                        "kruskal_h": h_stat,
                        "p_value": p_value,
                    }
                )
    write_dataframe(pd.DataFrame(kruskal_rows), stats_dir / "kruskal_region.tsv")

    mw_rows: List[Dict[str, object]] = []
    if "school_level" in doc_df.columns:
        eligible = doc_df[doc_df["school_level"].isin(["osnovna", "srednja"])]
        for metric in numeric_metrics:
            basic = eligible[eligible["school_level"] == "osnovna"][metric].dropna().values
            secondary = eligible[eligible["school_level"] == "srednja"][metric].dropna().values
            if len(basic) >= 5 and len(secondary) >= 5:
                u_stat, p_value = stats.mannwhitneyu(basic, secondary, alternative="two-sided")
                mw_rows.append(
                    {
                        "metric": metric,
                        "n_osnovna": len(basic),
                        "n_srednja": len(secondary),
                        "mannwhitney_u": u_stat,
                        "p_value": p_value,
                    }
                )
    write_dataframe(pd.DataFrame(mw_rows), stats_dir / "mannwhitney_school_level.tsv")




def run_regression_models(doc_df: pd.DataFrame, output_dir: Path) -> None:
    """Fit exploratory OLS models with the control variables named in the disposition.

    Models are written only if statsmodels is installed and enough data is
    available. They are meant as a thesis-facing starting point, not as a final
    inferential analysis without diagnostics.
    """
    try:
        import statsmodels.formula.api as smf
    except Exception:
        return

    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    metrics = [metric for metric in PLOT_METRICS if metric in doc_df.columns]
    candidate_controls = []
    if "grade_num" in doc_df.columns:
        candidate_controls.append("grade_num")
    if "n_word_tokens" in doc_df.columns:
        candidate_controls.append("n_word_tokens")
    for cat in ["school_level", "region", "contact_zone", "text_type", "year_start"]:
        if cat in doc_df.columns and doc_df[cat].nunique(dropna=True) > 1:
            candidate_controls.append(f"C({cat})")

    rows = []
    for metric in metrics:
        needed_cols = [metric]
        for col in ["grade_num", "n_word_tokens", "school_level", "region", "contact_zone", "text_type", "year_start"]:
            if col in doc_df.columns:
                needed_cols.append(col)
        subset = doc_df[needed_cols].replace([np.inf, -np.inf], np.nan).dropna(subset=[metric]).copy()
        if len(subset) < 30 or not candidate_controls:
            continue
        formula = metric + " ~ " + " + ".join(candidate_controls)
        try:
            model = smf.ols(formula=formula, data=subset).fit()
        except Exception as exc:
            rows.append({"metric": metric, "formula": formula, "error": str(exc)})
            continue
        for term, coef in model.params.items():
            rows.append(
                {
                    "metric": metric,
                    "formula": formula,
                    "term": term,
                    "coef": coef,
                    "std_err": model.bse.get(term, np.nan),
                    "t": model.tvalues.get(term, np.nan),
                    "p_value": model.pvalues.get(term, np.nan),
                    "n": int(model.nobs),
                    "r_squared": model.rsquared,
                    "adj_r_squared": model.rsquared_adj,
                    "aic": model.aic,
                    "bic": model.bic,
                }
            )
    write_dataframe(pd.DataFrame(rows), model_dir / "ols_control_models.tsv")

def make_plots(doc_df: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if "grade_num" in doc_df.columns:
        grade_df = (
            doc_df.dropna(subset=["grade_num"])
            .groupby("grade_num", dropna=False)[[metric for metric in PLOT_METRICS if metric in doc_df.columns]]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values("grade_num")
        )
        if not grade_df.empty:
            for metric in [m for m in ["lemma_mattr_50", "lexical_density", "deprel_entropy"] if m in grade_df.columns]:
                fig, ax = plt.subplots(figsize=(7, 4.5))
                ax.plot(grade_df["grade_num"], grade_df[metric], marker="o")
                ax.set_xlabel("Grade number")
                ax.set_ylabel(metric)
                ax.set_title(f"{metric} by grade")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(plot_dir / f"{metric}_by_grade.png", dpi=150)
                plt.close(fig)

    if "region" in doc_df.columns:
        region_df = (
            doc_df.groupby("region", dropna=False)[[metric for metric in ["lexical_density", "lemma_mattr_50"] if metric in doc_df.columns]]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values("region")
        )
        for metric in [m for m in ["lexical_density", "lemma_mattr_50"] if m in region_df.columns]:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(region_df["region"], region_df[metric])
            ax.set_xlabel("Region")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} by region")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            fig.savefig(plot_dir / f"{metric}_by_region.png", dpi=150)
            plt.close(fig)

    if "year_start" in doc_df.columns:
        year_df = (
            doc_df.dropna(subset=["year_start"])
            .groupby("year_start", dropna=False)[[metric for metric in ["rare_lemma_token_share_top3000", "deprel_entropy"] if metric in doc_df.columns]]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values("year_start")
        )
        for metric in [m for m in ["rare_lemma_token_share_top3000", "deprel_entropy"] if m in year_df.columns]:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(year_df["year_start"], year_df[metric], marker="o")
            ax.set_xlabel("Start year")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} by year")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_dir / f"{metric}_by_year.png", dpi=150)
            plt.close(fig)



def corpus_summary_json(
    token_df: pd.DataFrame,
    sentence_df: pd.DataFrame,
    doc_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    output_path: Path,
    reference_source: str,
) -> None:
    matched_docs = set(doc_df["doc_id"]) & set(meta_df["doc_id"]) if not meta_df.empty else set()
    summary = {
        "documents_in_corpus": int(doc_df["doc_id"].nunique()) if "doc_id" in doc_df.columns else 0,
        "sentences_in_corpus": int(sentence_df["sent_id"].nunique()) if "sent_id" in sentence_df.columns else 0,
        "tokens_in_corpus": int(len(token_df)),
        "documents_in_metadata": int(meta_df["doc_id"].nunique()) if not meta_df.empty else 0,
        "documents_matched_to_metadata": int(len(matched_docs)),
        "regions": sorted([str(x) for x in meta_df["region"].dropna().unique().tolist()]) if "region" in meta_df.columns else [],
        "contact_zones": sorted([str(x) for x in meta_df["contact_zone"].dropna().unique().tolist()]) if "contact_zone" in meta_df.columns else [],
        "grades": sorted([str(x) for x in meta_df["grade"].dropna().unique().tolist()]) if "grade" in meta_df.columns else [],
        "years": sorted([str(x) for x in meta_df["date_label"].dropna().unique().tolist()]) if "date_label" in meta_df.columns else [],
        "reference_frequency_source": reference_source,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")



def enrich_tokens_with_metadata(token_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    if meta_df.empty:
        return token_df.copy()
    return token_df.merge(meta_df, on="doc_id", how="left", suffixes=("", "_meta"))



def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_path: Path = args.input.resolve()
    metadata_path: Path = args.metadata.resolve()
    output_dir: Path = args.output_dir.resolve()

    if not input_path.exists():
        raise SystemExit(f"Input CoNLL-U file not found: {input_path}")
    if not metadata_path.exists():
        raise SystemExit(f"Metadata TSV not found: {metadata_path}")

    log_progress(f"Loading corpus from {input_path}")
    sentences = read_conllu(input_path)
    log_progress(f"Loading metadata from {metadata_path}")
    meta_df = load_metadata(metadata_path)
    log_progress("Building token and sentence tables")
    token_df = build_token_dataframe(sentences)
    token_df = enrich_tokens_with_metadata(token_df, meta_df)
    sentence_df = build_sentence_dataframe(token_df)

    log_progress("Preparing frequency references and document metrics")
    external_reference = maybe_load_reference_frequency(args.reference_freq, strict=args.strict_reference_freq)
    reference_df = external_reference if external_reference is not None else build_corpus_frequency_reference(token_df)
    reference_source = "external" if external_reference is not None else "corpus_internal"

    doc_df = compute_document_metrics(token_df, sentence_df, meta_df, external_reference)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(doc_df, output_dir / "document_metrics.tsv")
    write_dataframe(reference_df, output_dir / "reference_frequency.tsv")

    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    corpus_docs = pd.DataFrame({"doc_id": sorted(token_df["doc_id"].dropna().unique())})
    if not meta_df.empty:
        join_diag = corpus_docs.merge(meta_df[["doc_id", "orig_id", "corr_id", "region", "grade", "date_label", "text_type", "contact_zone"]], on="doc_id", how="left")
    else:
        join_diag = corpus_docs
    write_dataframe(join_diag, diagnostics_dir / "metadata_join_diagnostics.tsv")

    summary_dir = output_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    for group_cols in [
        ["region"],
        ["grade"],
        ["year_start"],
        ["school_level"],
        ["text_type"],
        ["contact_zone"],
        ["region", "grade"],
        ["region", "year_start"],
    ]:
        available = [col for col in group_cols if col in doc_df.columns]
        if not available:
            continue
        summary_df = aggregate_summary(doc_df, available)
        filename = "summary_by_" + "_and_".join(available) + ".tsv"
        write_dataframe(summary_df, summary_dir / filename)

    rare_dir = output_dir / "rare_words"
    rare_dir.mkdir(parents=True, exist_ok=True)
    log_progress("Writing rare-word tables")
    for group_col in ["region", "contact_zone", "year_start", "grade", "school_level", "text_type"]:
        if group_col not in token_df.columns:
            continue
        rare_df = compute_rarest_words_by_group(
            token_df=token_df,
            reference_df=reference_df,
            group_col=group_col,
            top_n=args.rare_top_n,
            min_group_count=args.rare_min_count,
        )
        write_dataframe(rare_df, rare_dir / f"rarest_lemmas_by_{group_col}.tsv")

    if args.compute_mwe:
        mwe_dir = output_dir / "mwe"
        mwe_dir.mkdir(parents=True, exist_ok=True)
        log_progress("Preparing sentence records for MWE extraction")
        sentence_records = list(iter_sentence_records(token_df))
        for group_col in ["region", "contact_zone", "grade", "year_start", "school_level"]:
            if group_col not in token_df.columns:
                continue
            log_progress(f"Computing MWE candidates by {group_col}")
            mwe_df = compute_mwe_candidates_from_records(
                sentence_records=sentence_records,
                group_col=group_col,
                min_n=args.mwe_min_n,
                max_n=args.mwe_max_n,
                min_count=args.mwe_min_count,
            )
            output_path = mwe_dir / f"mwe_candidates_by_{group_col}.tsv"
            write_dataframe(mwe_df, output_path)
            log_progress(f"Wrote {output_path}")

    if args.run_stats:
        log_progress("Running statistical tests")
        run_statistical_tests(doc_df, output_dir)

    if args.run_models:
        log_progress("Running OLS control models")
        run_regression_models(doc_df, output_dir)

    if args.make_plots:
        log_progress("Rendering plots")
        make_plots(doc_df, output_dir)

    log_progress("Writing corpus summary")
    corpus_summary_json(
        token_df=token_df,
        sentence_df=sentence_df,
        doc_df=doc_df,
        meta_df=meta_df,
        output_path=output_dir / "corpus_summary.json",
        reference_source=reference_source,
    )

    print(f"Analysis written to: {output_dir}")
    print(f"Document metrics: {output_dir / 'document_metrics.tsv'}")
    print(f"Corpus summary: {output_dir / 'corpus_summary.json'}")
    return 0



def build_arg_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    default_input = data_dir / "annotated" / "solar-classla.conllu"
    if not default_input.exists():
        default_input = data_dir / "raw" / "solar-orig.conllu"

    parser = argparse.ArgumentParser(
        description="Compute lexical, syntactic, and frequency-based analyses for a SOLAR-style corpus."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Annotated CoNLL-U file to analyse (default: data/annotated/solar-classla.conllu if present, otherwise data/raw/solar-orig.conllu).",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=data_dir / "metadata" / "solar-meta.tsv",
        help="Metadata TSV file (default: data/metadata/solar-meta.tsv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "analysis",
        help="Directory for analysis outputs (default: analysis/).",
    )
    parser.add_argument(
        "--reference-freq",
        type=Path,
        default=None,
        help=(
            "Optional external reference frequency list (TSV/CSV) for lexical sophistication. "
            "If omitted or missing, the script falls back to corpus-internal lemma frequencies."
        ),
    )
    parser.add_argument(
        "--strict-reference-freq",
        action="store_true",
        help="Fail if --reference-freq points to a missing real file instead of falling back.",
    )
    parser.add_argument(
        "--rare-top-n",
        type=int,
        default=30,
        help="Number of rare lemmas to keep per group (default: 30).",
    )
    parser.add_argument(
        "--rare-min-count",
        type=int,
        default=2,
        help="Minimum within-group lemma count for rare-word tables (default: 2).",
    )
    parser.add_argument(
        "--compute-mwe",
        action="store_true",
        help="Compute MWE / collocation candidate tables.",
    )
    parser.add_argument(
        "--mwe-min-n",
        type=int,
        default=2,
        help="Minimum n-gram size for MWE extraction (default: 2).",
    )
    parser.add_argument(
        "--mwe-max-n",
        type=int,
        default=4,
        help="Maximum n-gram size for MWE extraction (default: 4).",
    )
    parser.add_argument(
        "--mwe-min-count",
        type=int,
        default=5,
        help="Minimum raw count for MWE candidates (default: 5).",
    )
    parser.add_argument(
        "--run-stats",
        action="store_true",
        help="Run simple non-parametric statistical tests and write them to analysis/stats/.",
    )
    parser.add_argument(
        "--run-models",
        action="store_true",
        help="Run exploratory OLS control models and write them to analysis/models/.",
    )
    parser.add_argument(
        "--make-plots",
        action="store_true",
        help="Create basic matplotlib plots in analysis/plots/.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
