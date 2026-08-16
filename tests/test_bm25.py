import math
import pytest
from scout.bm25 import BM25Index, term_counts, normalize


def test_term_counts_counts_frequency_not_just_presence():
    counts = term_counts("the cat sat on the mat")
    assert counts["the"] == 2
    assert counts["cat"] == 1


def test_bm25_scores_match_hand_computed_reference():
    corpus = [
        {"content": "the cat sat on the mat"},
        {"content": "the dog sat on the log"},
    ]
    index = BM25Index(corpus)
    scores = index.scores("cat")

    expected_cat_score = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
    assert scores[0] == pytest.approx(expected_cat_score)
    assert scores[1] == 0.0


def test_bm25_rarer_term_scores_higher_than_common_term():
    corpus = [
        {"content": "widget frobnication error"},
        {"content": "widget calibration error"},
        {"content": "widget teleportation error"},
    ]
    index = BM25Index(corpus)
    # "widget" and "error" appear in every doc (uninformative). A query on
    # a term unique to one document should score that document meaningfully
    # higher than a query on a term shared by all of them.
    rare_scores = index.scores("frobnication")
    common_scores = index.scores("widget")
    assert rare_scores[0] > common_scores[0]


def test_bm25_empty_query_scores_zero():
    corpus = [{"content": "some content here"}]
    index = BM25Index(corpus)
    assert index.scores("") == [0.0]
    assert index.scores("   ") == [0.0]


def test_bm25_no_matching_terms_scores_zero():
    corpus = [{"content": "widgets and gadgets"}]
    index = BM25Index(corpus)
    assert index.scores("nonexistent") == [0.0]


def test_bm25_empty_corpus():
    index = BM25Index([])
    assert index.scores("anything") == []


def test_bm25_rebuild_replaces_previous_corpus():
    index = BM25Index([{"content": "apples and oranges"}])
    assert index.scores("apples")[0] > 0

    index.build([{"content": "bananas and grapes"}])
    assert index.scores("apples") == [0.0]
    assert index.scores("bananas")[0] > 0


def test_normalize_min_max_scales_to_unit_range():
    assert normalize([1.0, 2.0, 3.0]) == [0.0, 0.5, 1.0]


def test_normalize_handles_all_equal_scores():
    assert normalize([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]
    assert normalize([5.0, 5.0]) == [0.0, 0.0]


def test_normalize_empty_list():
    assert normalize([]) == []
