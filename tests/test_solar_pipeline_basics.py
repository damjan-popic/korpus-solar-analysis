from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import solar_reannotate as reann
import solar_analysis as analysis


def test_solar_sent_id_maps_to_metadata_doc_id():
    assert reann.sent_id_to_doc_id("solar1.1.1") == "solar1"
    assert reann.sent_id_to_doc_id("solar123.4.5") == "solar123"


def test_generated_raw_annotation_id_does_not_hide_as_solar_doc():
    assert reann.sent_id_to_doc_id("solar-orig-1") == "solar-orig"


def test_lexical_metrics_on_tiny_sequence():
    seq = ["a", "b", "a", "c"]
    assert analysis.ttr(seq) == 0.75
    assert analysis.mattr(seq, window_size=2) > 0
    assert analysis.mtld(seq) > 0


def test_contact_zone_mapping_is_stable():
    assert analysis.infer_contact_zone("Gorica") == "italija"
    assert analysis.infer_contact_zone("Novo mesto") == "hrvaška"
    assert analysis.infer_contact_zone("Ljubljana") == "notranjost"


def test_prebuilt_sentence_records_match_direct_mwe_computation():
    token_df = pd.DataFrame(
        [
            {
                "sent_id": "solar1.1.1",
                "token_index_in_sentence": 1,
                "doc_id": "solar1",
                "region": "west",
                "is_wordlike": True,
                "has_alpha": True,
                "lemma_norm": "mala",
                "upos": "ADJ",
            },
            {
                "sent_id": "solar1.1.1",
                "token_index_in_sentence": 2,
                "doc_id": "solar1",
                "region": "west",
                "is_wordlike": True,
                "has_alpha": True,
                "lemma_norm": "muca",
                "upos": "NOUN",
            },
            {
                "sent_id": "solar1.1.2",
                "token_index_in_sentence": 1,
                "doc_id": "solar1",
                "region": "west",
                "is_wordlike": True,
                "has_alpha": True,
                "lemma_norm": "mala",
                "upos": "ADJ",
            },
            {
                "sent_id": "solar1.1.2",
                "token_index_in_sentence": 2,
                "doc_id": "solar1",
                "region": "west",
                "is_wordlike": True,
                "has_alpha": True,
                "lemma_norm": "muca",
                "upos": "NOUN",
            },
        ]
    )

    sentence_records = list(analysis.iter_sentence_records(token_df))
    from_records = analysis.compute_mwe_candidates_from_records(
        sentence_records=sentence_records,
        group_col="region",
        min_n=2,
        max_n=2,
        min_count=1,
    )
    direct = analysis.compute_mwe_candidates(
        token_df=token_df,
        group_col="region",
        min_n=2,
        max_n=2,
        min_count=1,
    )

    assert sentence_records[0].tokens == [("mala", "ADJ"), ("muca", "NOUN")]
    assert from_records.to_dict("records") == direct.to_dict("records")
