#!/usr/bin/env python3
"""Quick sanity checks for Šolar CoNLL-U files before/after reannotation."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

TOKEN_COLUMNS = ("ID", "FORM", "LEMMA", "UPOS", "XPOS", "FEATS", "HEAD", "DEPREL", "DEPS", "MISC")
WORD_ID_RE = re.compile(r"^\d+$")
SOLAR_SENT_RE = re.compile(r"^(solar\d+)(?:[._-].*)?$", re.IGNORECASE)


def sent_doc(sent_id: str) -> str:
    m = SOLAR_SENT_RE.match(sent_id.strip())
    if m:
        return m.group(1)
    if "." in sent_id:
        return sent_id.split(".", 1)[0]
    m = re.match(r"^(.+?)[._-]\d+$", sent_id)
    return m.group(1) if m else sent_id


def iter_blocks(path: Path):
    block = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            if block:
                yield block
                block = []
            continue
        block.append(line)
    if block:
        yield block


def parse_sent_id(block: List[str]) -> Optional[str]:
    for line in block:
        if line.startswith("# sent_id = "):
            return line.split("=", 1)[1].strip()
    return None


def metadata_doc_ids(path: Path) -> set[str]:
    if not path or not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        ids = set()
        for row in reader:
            for col in ("OrigID", "CorrID", "doc_id", "DocID", "id", "ID"):
                value = (row.get(col) or "").strip()
                if value:
                    ids.add(re.sub(r"[st]$", "", value))
        return ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Check CoNLL-U format, IDs, roots, and metadata coverage.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--fail-on-generated-ids", action="store_true")
    parser.add_argument("--min-metadata-match-ratio", type=float, default=0.95)
    args = parser.parse_args()

    n_sent = 0
    n_missing_sent_id = 0
    n_bad_col = 0
    n_bad_roots = 0
    n_generated_id_like = 0
    doc_ids = set()
    examples_generated = []

    for block in iter_blocks(args.input):
        n_sent += 1
        sent_id = parse_sent_id(block)
        if not sent_id:
            n_missing_sent_id += 1
        else:
            doc_ids.add(sent_doc(sent_id))
            if re.match(r"^solar-orig-\d+$", sent_id):
                n_generated_id_like += 1
                if len(examples_generated) < 5:
                    examples_generated.append(sent_id)
        roots = 0
        for line in block:
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 10:
                n_bad_col += 1
                continue
            if WORD_ID_RE.match(parts[0]) and parts[6] == "0":
                roots += 1
        if roots != 1:
            n_bad_roots += 1

    print(f"sentences: {n_sent}")
    print(f"documents_from_sent_id: {len(doc_ids)}")
    print(f"missing_sent_id: {n_missing_sent_id}")
    print(f"bad_column_lines: {n_bad_col}")
    print(f"sentences_with_not_exactly_one_root: {n_bad_roots}")
    print(f"generated_solar_orig_id_like_sentences: {n_generated_id_like}")
    if examples_generated:
        print("generated_id_examples: " + ", ".join(examples_generated))

    fail = False
    if n_missing_sent_id or n_bad_col or n_bad_roots:
        fail = True
    if args.fail_on_generated_ids and n_generated_id_like:
        fail = True

    if args.metadata:
        meta_ids = metadata_doc_ids(args.metadata)
        matched = doc_ids & meta_ids
        ratio = len(matched) / len(doc_ids) if doc_ids else 0.0
        print(f"metadata_documents: {len(meta_ids)}")
        print(f"metadata_matched_documents: {len(matched)}")
        print(f"metadata_match_ratio: {ratio:.4f}")
        if ratio < args.min_metadata_match_ratio:
            fail = True

    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
