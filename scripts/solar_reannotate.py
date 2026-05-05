#!/usr/bin/env python3
"""Safely re-annotate an existing CoNLL-U corpus with CLASSLA.

This script is for corpora that are already in CoNLL-U. It preserves the
corpus structure needed for downstream analysis: sentence boundaries, token
IDs, token forms, original comments such as ``# sent_id``, and spacing markers
such as ``SpaceAfter=No``. It overwrites only the annotation layers predicted by
CLASSLA: lemma, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, and NER in MISC.

For Šolar/Solar this matters because ``# sent_id = solar1.1.1`` is the bridge
back to ``solar-meta.tsv``. If you accidentally raw-annotate CoNLL-U, you get
IDs like ``solar-orig-1`` and metadata joins break. This script checks for that.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from solar3_lexical_analysis.classla_helpers import build_pipeline as build_classla_pipeline

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
EMPTY_NODE_ID_RE = re.compile(r"^\d+\.\d+$")
MULTIWORD_ID_RE = re.compile(r"^\d+-\d+$")
UPPERCASE_RE = re.compile(r"[A-ZČŠŽĆĐ]")
SOLAR_SENT_RE = re.compile(r"^(solar\d+)(?:[._-].*)?$", flags=re.IGNORECASE)
DEFAULT_METADATA_ID_COLUMNS = ("OrigID", "CorrID", "doc_id", "DocID", "id", "ID")
DEFAULT_PROCESSORS = "tokenize,pos,lemma,depparse,ner"


@dataclass
class ConlluRow:
    values: Dict[str, str]

    @property
    def raw_id(self) -> str:
        return self.values["ID"]

    @property
    def is_word(self) -> bool:
        return bool(WORD_ID_RE.fullmatch(self.raw_id))

    @property
    def is_multiword(self) -> bool:
        return bool(MULTIWORD_ID_RE.fullmatch(self.raw_id))

    @property
    def is_empty_node(self) -> bool:
        return bool(EMPTY_NODE_ID_RE.fullmatch(self.raw_id))

    def copy(self) -> "ConlluRow":
        return ConlluRow(dict(self.values))


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
    def newdoc_id(self) -> Optional[str]:
        return self.get_comment_value("newdoc id")

    @property
    def doc_id(self) -> Optional[str]:
        if self.sent_id:
            return sent_id_to_doc_id(self.sent_id)
        if self.newdoc_id:
            return normalize_doc_id(self.newdoc_id)
        return None

    @property
    def word_rows(self) -> List[ConlluRow]:
        return [row for row in self.rows if row.is_word]

    def reconstruct_text(self) -> str:
        pieces: List[str] = []
        for row in self.word_rows:
            pieces.append(row.values["FORM"])
            misc = parse_misc(row.values["MISC"])
            if misc.get("SpaceAfter") != "No":
                pieces.append(" ")
        return "".join(pieces).strip()

    def to_conllu(self) -> str:
        lines = list(self.comments)
        for row in self.rows:
            lines.append("\t".join(row.values[col] for col in TOKEN_COLUMNS))
        return "\n".join(lines)


def normalize_doc_id(value: object) -> str:
    return str(value or "").strip()


def sent_id_to_doc_id(sent_id: str) -> str:
    """Map sentence IDs to document IDs without regenerating anything.

    Solar/Šolar sentence IDs look like ``solar1.1.1`` and should map to
    ``solar1`` so they join with metadata IDs like ``solar1s`` / ``solar1t``.
    Generated raw-annotation IDs like ``solar-orig-1`` map to ``solar-orig``;
    that will fail the metadata check, which is exactly what we want.
    """
    text = normalize_doc_id(sent_id)
    solar_match = SOLAR_SENT_RE.match(text)
    if solar_match:
        return solar_match.group(1)
    if "." in text:
        return text.split(".", 1)[0]
    sent_suffix = re.match(r"^(.+?)[._-]s\d+$", text, flags=re.IGNORECASE)
    if sent_suffix:
        return sent_suffix.group(1)
    final_number = re.match(r"^(.+?)[._-]\d+$", text)
    if final_number:
        return final_number.group(1)
    return text


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


def format_misc(misc: Dict[str, str]) -> str:
    if not misc:
        return "_"
    preferred_order = ["NER", "SpaceAfter"]
    keys = [key for key in preferred_order if key in misc]
    keys.extend(sorted(key for key in misc if key not in preferred_order))
    return "|".join(f"{key}={misc[key]}" for key in keys)


def parse_conllu_text(text: str) -> List[ConlluSentence]:
    text = text.strip("\n")
    if not text:
        return []
    sentences: List[ConlluSentence] = []
    blocks = re.split(r"\n\s*\n", text)
    for block_number, block in enumerate(blocks, start=1):
        comments: List[str] = []
        rows: List[ConlluRow] = []
        for line_number, line in enumerate(block.splitlines(), start=1):
            if not line.strip():
                continue
            if line.startswith("#"):
                comments.append(line.rstrip())
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(TOKEN_COLUMNS):
                raise ValueError(
                    f"Expected {len(TOKEN_COLUMNS)} tab-separated CoNLL-U columns, got {len(parts)} "
                    f"in block {block_number}, line {line_number}: {line!r}"
                )
            rows.append(ConlluRow(dict(zip(TOKEN_COLUMNS, parts))))
        if comments or rows:
            sentences.append(ConlluSentence(comments=comments, rows=rows))
    return sentences


def read_conllu(path: Path) -> List[ConlluSentence]:
    return parse_conllu_text(path.read_text(encoding="utf-8"))


def write_conllu(path: Path, sentences: Sequence[ConlluSentence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, sentence in enumerate(sentences):
            handle.write(sentence.to_conllu())
            handle.write("\n")
            if index != len(sentences) - 1:
                handle.write("\n")
    tmp_path.replace(path)


def write_tsv(path: Path, rows: Sequence[Dict[str, str]], header: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header), delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: str(row.get(key, "")) for key in header})


def parse_metadata_id_columns(value: object) -> Optional[List[str]]:
    if value in (None, ""):
        return None
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def metadata_id_variants(value: object, strip_suffix_regex: str = r"[st]$") -> Iterable[str]:
    text = str(value or "").strip()
    if not text:
        return []
    variants = {text}
    if "." in text:
        variants.add(text.split(".", 1)[0])
    if strip_suffix_regex:
        try:
            stripped = re.sub(strip_suffix_regex, "", text)
        except re.error as exc:
            raise ValueError(f"Invalid metadata strip regex: {strip_suffix_regex!r}") from exc
        if stripped and stripped != text:
            variants.add(stripped)
            if "." in stripped:
                variants.add(stripped.split(".", 1)[0])
    return sorted(variants)


def load_metadata_doc_ids(
    path: Optional[Path],
    id_columns: Optional[Sequence[str]] = None,
    strip_suffix_regex: str = r"[st]$",
) -> set[str]:
    if path is None:
        return set()
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    doc_ids: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Metadata file {path} is missing a header row.")
        wanted = list(id_columns or [])
        if not wanted:
            wanted = [col for col in DEFAULT_METADATA_ID_COLUMNS if col in reader.fieldnames]
        if not wanted:
            wanted = [col for col in reader.fieldnames if col.lower().endswith("id")]
        if not wanted:
            raise ValueError(
                f"Could not find metadata ID columns in {path}. "
                "Pass --metadata-id-columns OrigID,CorrID or similar."
            )
        missing = [col for col in wanted if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"Metadata file {path} is missing requested columns: {', '.join(missing)}")
        for row in reader:
            for column in wanted:
                for candidate in metadata_id_variants(row.get(column, ""), strip_suffix_regex=strip_suffix_regex):
                    if candidate:
                        doc_ids.add(candidate)
    return doc_ids


def sentence_to_pretokenized_string(sentence: ConlluSentence) -> str:
    return " ".join(row.values["FORM"] for row in sentence.word_rows)


def build_pipeline(
    *,
    lang: str,
    classla_type: str,
    download_models: bool,
    models_dir: Optional[Path],
    processors: str = DEFAULT_PROCESSORS,
):
    return build_classla_pipeline(
        lang=lang,
        processors=processors,
        classla_type=classla_type,
        models_dir=models_dir,
        download=download_models,
        extra_kwargs={"tokenize_pretokenized": True},
    )


def parse_classla_doc(doc) -> List[ConlluSentence]:
    return parse_conllu_text(doc.to_conll())


def merge_misc(original: Dict[str, str], predicted: Dict[str, str]) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    if "NER" in predicted:
        merged["NER"] = predicted["NER"]
    elif "NER" in original:
        merged["NER"] = original["NER"]

    # Original spacing is structure, not annotation.
    if "SpaceAfter" in original:
        merged["SpaceAfter"] = original["SpaceAfter"]
    elif "SpaceAfter" in predicted:
        merged["SpaceAfter"] = predicted["SpaceAfter"]

    for key, value in original.items():
        if key not in {"NER", "SpaceAfter"}:
            merged[key] = value
    for key, value in predicted.items():
        if key not in {"NER", "SpaceAfter"} and key not in merged:
            merged[key] = value
    return merged


def copy_with_overwritten_annotation(
    *,
    original: ConlluSentence,
    predicted: ConlluSentence,
    diff_rows: List[Dict[str, str]],
    field_counter: Counter,
    strict_forms: bool = True,
) -> ConlluSentence:
    original_words = original.word_rows
    predicted_words = predicted.word_rows
    if len(original_words) != len(predicted_words):
        raise ValueError(
            f"Token count mismatch for sent_id={original.sent_id!r}: "
            f"{len(original_words)} original vs {len(predicted_words)} predicted"
        )

    updated_rows: List[ConlluRow] = []
    word_pointer = 0
    for row in original.rows:
        if not row.is_word:
            updated_rows.append(row.copy())
            continue

        updated = row.copy()
        predicted_row = predicted_words[word_pointer]
        word_pointer += 1

        original_form = updated.values["FORM"]
        predicted_form = predicted_row.values["FORM"]
        if strict_forms and original_form != predicted_form:
            raise ValueError(
                f"FORM mismatch for sent_id={original.sent_id!r}, token_id={row.raw_id}: "
                f"{original_form!r} != {predicted_form!r}"
            )

        for field_name in ("LEMMA", "UPOS", "XPOS", "FEATS", "HEAD", "DEPREL", "DEPS"):
            old_value = updated.values[field_name]
            new_value = predicted_row.values[field_name]
            if old_value != new_value:
                field_counter[field_name] += 1
                diff_rows.append(
                    {
                        "doc_id": original.doc_id or "",
                        "sent_id": original.sent_id or "",
                        "token_id": updated.raw_id,
                        "form": original_form,
                        "field": field_name,
                        "old": old_value,
                        "new": new_value,
                    }
                )
                updated.values[field_name] = new_value

        original_misc = parse_misc(updated.values["MISC"])
        predicted_misc = parse_misc(predicted_row.values["MISC"])
        new_misc_value = format_misc(merge_misc(original_misc, predicted_misc))
        if updated.values["MISC"] != new_misc_value:
            field_counter["MISC"] += 1
            diff_rows.append(
                {
                    "doc_id": original.doc_id or "",
                    "sent_id": original.sent_id or "",
                    "token_id": updated.raw_id,
                    "form": original_form,
                    "field": "MISC",
                    "old": updated.values["MISC"],
                    "new": new_misc_value,
                }
            )
            updated.values["MISC"] = new_misc_value

        updated_rows.append(updated)

    return ConlluSentence(comments=list(original.comments), rows=updated_rows)


def batched(items: Sequence[ConlluSentence], batch_size: int) -> Iterator[List[ConlluSentence]]:
    size = max(1, int(batch_size or 1))
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def annotate_batch(
    *,
    pipeline,
    batch: Sequence[ConlluSentence],
    diff_rows: List[Dict[str, str]],
    field_counter: Counter,
    strict_forms: bool,
) -> List[ConlluSentence]:
    """Annotate a batch and merge diffs only after the whole batch succeeds.

    CLASSLA is faster when several pretokenized sentences are sent in one call.
    If one sentence in a batch fails alignment, however, we do not want partial
    diff/counter side effects from the successful sentences in that failed batch.
    This local-commit pattern keeps reports honest and lets the caller retry
    sentence-by-sentence.
    """
    text = "\n".join(sentence_to_pretokenized_string(sentence) for sentence in batch)
    predicted_sentences = parse_classla_doc(pipeline(text))
    if len(predicted_sentences) != len(batch):
        raise ValueError(f"Expected {len(batch)} predicted sentence(s), got {len(predicted_sentences)}")

    local_diff_rows: List[Dict[str, str]] = []
    local_counter: Counter = Counter()
    updated: List[ConlluSentence] = []
    for original, predicted in zip(batch, predicted_sentences):
        updated.append(
            copy_with_overwritten_annotation(
                original=original,
                predicted=predicted,
                diff_rows=local_diff_rows,
                field_counter=local_counter,
                strict_forms=strict_forms,
            )
        )

    diff_rows.extend(local_diff_rows)
    field_counter.update(local_counter)
    return updated


def sentence_signature(sentence: ConlluSentence) -> Tuple[Optional[str], Tuple[str, ...], Tuple[str, ...]]:
    return (
        sentence.sent_id,
        tuple(row.raw_id for row in sentence.rows),
        tuple(row.values["FORM"] for row in sentence.word_rows),
    )


def root_problem(sentence: ConlluSentence) -> Optional[str]:
    word_rows = sentence.word_rows
    if not word_rows:
        return "no ordinary word tokens"
    roots = [row for row in word_rows if row.values["HEAD"] == "0"]
    if len(roots) != 1:
        return f"expected 1 HEAD=0 root, found {len(roots)}"
    if roots[0].values["DEPREL"] != "root":
        return f"HEAD=0 token has DEPREL={roots[0].values['DEPREL']!r}, expected 'root'"
    return None


def quality_report(
    original_sentences: Sequence[ConlluSentence],
    updated_sentences: Sequence[ConlluSentence],
    metadata_doc_ids: set[str],
) -> Dict[str, object]:
    original_signatures = [sentence_signature(sentence) for sentence in original_sentences]
    updated_signatures = [sentence_signature(sentence) for sentence in updated_sentences]
    sent_ids_preserved = [orig[0] for orig in original_signatures] == [upd[0] for upd in updated_signatures]
    token_ids_preserved = [orig[1] for orig in original_signatures] == [upd[1] for upd in updated_signatures]
    forms_preserved = [orig[2] for orig in original_signatures] == [upd[2] for upd in updated_signatures]

    missing_sent_ids = sum(1 for sentence in updated_sentences if not sentence.sent_id)
    corpus_doc_ids = {sentence.doc_id for sentence in updated_sentences if sentence.doc_id}
    matched_doc_ids = corpus_doc_ids & metadata_doc_ids if metadata_doc_ids else set()
    metadata_match_ratio = (len(matched_doc_ids) / len(corpus_doc_ids)) if metadata_doc_ids and corpus_doc_ids else None

    root_errors: List[Dict[str, str]] = []
    uppercase_deprels: Counter[str] = Counter()
    for sentence in updated_sentences:
        problem = root_problem(sentence)
        if problem:
            root_errors.append({"sent_id": sentence.sent_id or "", "problem": problem})
        for row in sentence.word_rows:
            deprel = row.values["DEPREL"]
            if deprel != "_" and UPPERCASE_RE.search(deprel):
                uppercase_deprels[deprel] += 1

    return {
        "sent_ids_preserved": sent_ids_preserved,
        "token_ids_preserved": token_ids_preserved,
        "forms_preserved": forms_preserved,
        "missing_sent_ids": missing_sent_ids,
        "documents_in_corpus": len(corpus_doc_ids),
        "documents_in_metadata": len(metadata_doc_ids),
        "documents_matched_to_metadata": len(matched_doc_ids) if metadata_doc_ids else None,
        "metadata_match_ratio": metadata_match_ratio,
        "documents_missing_in_metadata": sorted(corpus_doc_ids - metadata_doc_ids)[:100] if metadata_doc_ids else [],
        "root_error_count": len(root_errors),
        "root_error_examples": root_errors[:25],
        "uppercase_deprel_count": sum(uppercase_deprels.values()),
        "uppercase_deprel_examples": dict(uppercase_deprels.most_common(25)),
    }


def should_fail_quality_checks(
    *,
    report: Dict[str, object],
    error_count: int,
    min_metadata_match_ratio: float,
    require_clean_output: bool,
) -> List[str]:
    failures: List[str] = []
    if not report.get("sent_ids_preserved"):
        failures.append("sentence IDs were not preserved")
    if not report.get("token_ids_preserved"):
        failures.append("token IDs were not preserved")
    if not report.get("forms_preserved"):
        failures.append("token forms were not preserved")
    metadata_ratio = report.get("metadata_match_ratio")
    if metadata_ratio is not None and float(metadata_ratio) < min_metadata_match_ratio:
        failures.append(f"metadata match ratio is {float(metadata_ratio):.3f}, below required {min_metadata_match_ratio:.3f}")
    if require_clean_output:
        if error_count:
            failures.append(f"{error_count} sentence-level annotation errors occurred")
        if int(report.get("root_error_count", 0)):
            failures.append(f"{report.get('root_error_count')} sentence(s) do not have exactly one UD root")
        if int(report.get("uppercase_deprel_count", 0)):
            failures.append("uppercase / non-UD-looking dependency labels remain")
    return failures


def write_validation_report(
    path: Path,
    report: Dict[str, object],
    failures: Sequence[str],
    error_rows: Sequence[Dict[str, str]],
) -> None:
    rows: List[Dict[str, str]] = []
    for failure in failures:
        rows.append({"level": "error", "category": "quality_failure", "sent_id": "", "detail": failure})
    for error in error_rows:
        rows.append({"level": "error", "category": "annotation_error", "sent_id": error.get("sent_id", ""), "detail": error.get("error", "")})
    root_examples = report.get("root_error_examples", [])
    if isinstance(root_examples, list):
        for item in root_examples:
            if isinstance(item, dict):
                rows.append({"level": "warning", "category": "root_check", "sent_id": item.get("sent_id", ""), "detail": item.get("problem", "")})
    uppercase_examples = report.get("uppercase_deprel_examples", {})
    if isinstance(uppercase_examples, dict):
        for label, count in uppercase_examples.items():
            rows.append({"level": "warning", "category": "uppercase_deprel", "sent_id": "", "detail": f"{label}: {count}"})
    for key in (
        "sent_ids_preserved",
        "token_ids_preserved",
        "forms_preserved",
        "missing_sent_ids",
        "documents_in_corpus",
        "documents_in_metadata",
        "documents_matched_to_metadata",
        "metadata_match_ratio",
        "root_error_count",
        "uppercase_deprel_count",
    ):
        rows.append({"level": "info", "category": key, "sent_id": "", "detail": str(report.get(key, ""))})
    write_tsv(path, rows, header=("level", "category", "sent_id", "detail"))


def annotate_corpus(args: argparse.Namespace) -> int:
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    diff_path = args.diff_report.expanduser().resolve() if args.diff_report else output_path.with_name(output_path.stem + "-diff.tsv")
    summary_path = args.summary.expanduser().resolve() if args.summary else output_path.with_name(output_path.stem + "-summary.json")
    errors_path = args.errors.expanduser().resolve() if args.errors else output_path.with_name(output_path.stem + "-errors.tsv")
    validation_path = args.validation_report.expanduser().resolve() if getattr(args, "validation_report", None) else output_path.with_name(output_path.stem + "-validation.tsv")

    if not input_path.exists():
        raise SystemExit(f"Input CoNLL-U file not found: {input_path}")
    if input_path == output_path:
        raise SystemExit("Input and output point to the same file. Refusing to overwrite the source corpus.")

    original_sentences = read_conllu(input_path)
    if not original_sentences:
        raise SystemExit(f"Input CoNLL-U file is empty: {input_path}")

    metadata_id_columns = parse_metadata_id_columns(getattr(args, "metadata_id_columns", None))
    metadata_strip_suffix_regex = getattr(args, "metadata_strip_suffix_regex", r"[st]$")
    metadata_doc_ids = load_metadata_doc_ids(
        args.metadata.expanduser().resolve() if args.metadata else None,
        id_columns=metadata_id_columns,
        strip_suffix_regex=metadata_strip_suffix_regex,
    )

    min_metadata_ratio = getattr(args, "min_metadata_match_ratio", None)
    if min_metadata_ratio is None:
        min_metadata_ratio = 0.95 if metadata_doc_ids else 0.0
    min_metadata_ratio = float(min_metadata_ratio)

    preliminary_report = quality_report(original_sentences, original_sentences, metadata_doc_ids)
    preliminary_failures = should_fail_quality_checks(
        report=preliminary_report,
        error_count=0,
        min_metadata_match_ratio=min_metadata_ratio,
        require_clean_output=False,
    )
    if getattr(args, "require_sent_id", False) and int(preliminary_report.get("missing_sent_ids", 0)):
        preliminary_failures.append(f"{preliminary_report.get('missing_sent_ids')} sentence(s) are missing # sent_id")
    if preliminary_failures:
        joined = "; ".join(preliminary_failures)
        raise SystemExit(
            "Input corpus failed preflight checks: "
            f"{joined}. This usually means the input is not the original metadata-linked CoNLL-U."
        )

    pipeline = build_pipeline(
        lang=args.lang,
        classla_type=args.classla_type,
        download_models=args.download_models,
        models_dir=args.models_dir,
        processors=getattr(args, "processors", DEFAULT_PROCESSORS),
    )

    updated_sentences: List[ConlluSentence] = []
    diff_rows: List[Dict[str, str]] = []
    error_rows: List[Dict[str, str]] = []
    field_counter: Counter = Counter()
    strict_forms = not args.allow_form_mismatch
    fail_on_error = bool(getattr(args, "fail_on_error", True))
    batch_size = max(1, int(getattr(args, "batch_size", 64) or 1))
    progress_every = max(1, int(getattr(args, "progress_every", 250) or 250))

    processed = 0
    for batch in batched(original_sentences, batch_size):
        processed += len(batch)
        if processed == len(batch) or processed % progress_every < len(batch) or processed == len(original_sentences):
            first_id = batch[0].sent_id or "sentence"
            print(f"[{processed}/{len(original_sentences)}] Reannotating around {first_id}", file=sys.stderr)
        try:
            updated_sentences.extend(
                annotate_batch(
                    pipeline=pipeline,
                    batch=batch,
                    diff_rows=diff_rows,
                    field_counter=field_counter,
                    strict_forms=strict_forms,
                )
            )
        except Exception as batch_exc:  # pragma: no cover - depends on runtime model behaviour
            # If a batch fails, retry sentence-by-sentence so the error report is precise.
            for sentence in batch:
                try:
                    updated_sentences.extend(
                        annotate_batch(
                            pipeline=pipeline,
                            batch=[sentence],
                            diff_rows=diff_rows,
                            field_counter=field_counter,
                            strict_forms=strict_forms,
                        )
                    )
                except Exception as exc:
                    error_rows.append({"doc_id": sentence.doc_id or "", "sent_id": sentence.sent_id or "", "error": str(exc or batch_exc)})
                    if fail_on_error:
                        write_tsv(errors_path, error_rows, header=("doc_id", "sent_id", "error"))
                        raise SystemExit(
                            f"Stopped at sent_id={sentence.sent_id!r}. See error report: {errors_path}\n"
                            "No mixed old/new annotation was written. Use --keep-original-on-error only for partial output."
                        ) from exc
                    updated_sentences.append(sentence)

    post_report = quality_report(original_sentences, updated_sentences, metadata_doc_ids)
    failures = should_fail_quality_checks(
        report=post_report,
        error_count=len(error_rows),
        min_metadata_match_ratio=min_metadata_ratio,
        require_clean_output=bool(getattr(args, "require_clean_output", False)),
    )

    write_conllu(output_path, updated_sentences)
    write_tsv(diff_path, diff_rows, header=("doc_id", "sent_id", "token_id", "form", "field", "old", "new"))
    write_tsv(errors_path, error_rows, header=("doc_id", "sent_id", "error"))
    write_validation_report(validation_path, post_report, failures, error_rows)

    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "validation_report": str(validation_path),
        "sentences_total": len(original_sentences),
        "annotation_changes_by_field": dict(sorted(field_counter.items())),
        "diff_rows": len(diff_rows),
        "errors": len(error_rows),
        "classla_type": args.classla_type,
        "lang": args.lang,
        "processors": getattr(args, "processors", DEFAULT_PROCESSORS),
        "models_dir": str(args.models_dir.expanduser().resolve()) if args.models_dir else None,
        "preserved_tokenization": True,
        "preserved_sentence_boundaries": True,
        "min_metadata_match_ratio": min_metadata_ratio,
        "require_clean_output": bool(getattr(args, "require_clean_output", False)),
        "batch_size": batch_size,
        "quality": post_report,
        "quality_failures": failures,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Written reannotated corpus to: {output_path}")
    print(f"Written diff report to: {diff_path}")
    print(f"Written validation report to: {validation_path}")
    print(f"Written summary to: {summary_path}")
    if error_rows:
        print(f"Encountered {len(error_rows)} sentence-level errors. See: {errors_path}", file=sys.stderr)
    if failures:
        print("Quality check failures:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        print(f"Output was written for inspection, but the run is not clean. See: {summary_path}", file=sys.stderr)
        return 2
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-annotate Šolar/Solar CoNLL-U with CLASSLA while preserving IDs/tokenization."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input CoNLL-U file to re-annotate.")
    parser.add_argument("--output", type=Path, required=True, help="Output CoNLL-U file.")
    parser.add_argument("--metadata", type=Path, default=None, help="Optional metadata TSV for coverage checks.")
    parser.add_argument("--metadata-id-columns", default=None, help="Comma-separated metadata ID columns, e.g. OrigID,CorrID.")
    parser.add_argument("--metadata-strip-suffix-regex", default=r"[st]$", help="Regex stripped from metadata IDs. Default strips Šolar final s/t.")
    parser.add_argument("--min-metadata-match-ratio", type=float, default=None, help="Required corpus-doc metadata match ratio. Default: 0.95 when metadata is supplied.")
    parser.add_argument("--diff-report", type=Path, default=None, help="Token-level diff TSV path.")
    parser.add_argument("--summary", type=Path, default=None, help="JSON summary report path.")
    parser.add_argument("--errors", type=Path, default=None, help="Sentence-level errors TSV path.")
    parser.add_argument("--validation-report", type=Path, default=None, help="Validation/quality TSV path.")
    parser.add_argument("--models-dir", type=Path, default=None, help="Shared CLASSLA models directory.")
    parser.add_argument("--lang", default="sl", help="CLASSLA language code. Default: sl.")
    parser.add_argument("--classla-type", default="standard", choices=["standard", "nonstandard", "spoken", "web"], help="CLASSLA model type.")
    parser.add_argument("--processors", default=DEFAULT_PROCESSORS, help=f"CLASSLA processors. Default: {DEFAULT_PROCESSORS}.")
    parser.add_argument("--download-models", action="store_true", help="Download CLASSLA models before running.")
    parser.add_argument("--allow-form-mismatch", action="store_true", help="Allow predicted FORM mismatch; original FORM is still kept.")
    parser.add_argument("--require-sent-id", action="store_true", help="Fail preflight if any sentence lacks # sent_id.")
    parser.add_argument("--require-clean-output", action="store_true", help="Return nonzero if roots/deprels/errors indicate an unclean run.")
    parser.set_defaults(fail_on_error=True)
    parser.add_argument("--fail-on-error", dest="fail_on_error", action="store_true", help="Abort on the first annotation/alignment error. Default.")
    parser.add_argument("--keep-original-on-error", dest="fail_on_error", action="store_false", help="Keep original sentence on failure; creates partial/mixed output.")
    parser.add_argument("--batch-size", type=int, default=64, help="Pretokenized sentences per CLASSLA call. Default: 64.")
    parser.add_argument("--progress-every", type=int, default=500, help="Print progress every N sentences. Default: 500.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    return annotate_corpus(args)


if __name__ == "__main__":
    raise SystemExit(main())
