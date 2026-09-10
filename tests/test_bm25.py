import math
import pytest
from scout.bm25 import BM25Index, term_counts, normalize, normalize_array


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
    # "widget" and "error" are in every doc, so a term unique to one should
    # score meaningfully higher than one shared by all.
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


# -- incremental index -------------------------------------------------------
# The add/remove/compact path that replaced the per-ingest rebuild. The
# property that matters most: an index reached incrementally must score
# identically to one built in a single pass.


def _docs(*texts):
    return [{"content": text} for text in texts]


CORPUS = _docs(
    "widget frobnication error nine thousand",
    "gadget calibration error handling",
    "sprocket teleportation notes",
    "widget calibration guide",
)


def test_add_documents_returns_the_slots_it_assigned():
    index = BM25Index(CORPUS[:2])
    assert index.add_documents(CORPUS[2:]) == [2, 3]


def test_incremental_adds_score_identically_to_a_single_build():
    """The property the whole inverted index rests on. BM25 depends on
    corpus-wide statistics (document frequency, average document length),
    so an index that failed to keep those current as documents arrived
    would drift away from a rebuilt one without ever raising."""
    at_once = BM25Index(CORPUS)

    incremental = BM25Index(CORPUS[:1])
    incremental.add_documents(CORPUS[1:3])
    incremental.add_documents(CORPUS[3:])

    for query in ("widget", "error", "calibration", "frobnication", "widget error"):
        assert incremental.scores(query) == pytest.approx(at_once.scores(query))


def test_removing_a_slot_stops_it_scoring_and_keeps_the_list_aligned():
    index = BM25Index(CORPUS)
    assert index.scores("frobnication")[0] > 0

    index.remove_slots([0])
    scores = index.scores("frobnication")
    # The tombstone keeps its position: shifting would re-point every slot
    # the caller holds at the wrong document.
    assert len(scores) == len(CORPUS)
    assert scores[0] == 0.0


def test_removal_updates_corpus_statistics_not_just_the_posting_list():
    """A document dropped from the postings but left in the document count
    would keep inflating every other document's IDF. Compare against an
    index built from only the surviving documents, which is what the
    removal is supposed to be equivalent to."""
    index = BM25Index(CORPUS)
    index.remove_slots([3])

    rebuilt = BM25Index(CORPUS[:3])
    for query in ("widget", "calibration", "error"):
        assert index.scores(query)[:3] == pytest.approx(rebuilt.scores(query))


def test_removing_the_last_document_holding_a_term_drops_the_term():
    index = BM25Index(CORPUS)
    index.remove_slots([2])
    assert index.scores("teleportation") == [0.0] * len(CORPUS)


def test_removing_an_already_removed_slot_is_a_no_op():
    """add_chunks() can ask for the same slot twice when one batch replaces
    two sources that share a chunk; making the caller de-duplicate first
    would just move the bookkeeping."""
    index = BM25Index(CORPUS)
    index.remove_slots([1])
    live_before = index.live_docs

    index.remove_slots([1])
    assert index.live_docs == live_before
    assert index.scores("gadget")[1] == 0.0


def test_slot_accounting_tracks_live_and_tombstoned_documents():
    index = BM25Index(CORPUS)
    assert (index.size, index.live_docs, index.tombstones) == (4, 4, 0)

    index.remove_slots([0, 2])
    assert (index.size, index.live_docs, index.tombstones) == (4, 2, 2)

    index.add_documents(_docs("a replacement document"))
    assert (index.size, index.live_docs, index.tombstones) == (5, 3, 2)


def test_compacted_reclaims_tombstones_and_preserves_live_order():
    index = BM25Index(CORPUS)
    index.remove_slots([1])
    index = index.compacted()

    assert (index.size, index.live_docs, index.tombstones) == (3, 3, 0)
    # Survivors stay in insertion order, which is what lets Retriever reset
    # its mapping to a plain range afterwards.
    survivors = BM25Index([CORPUS[0], CORPUS[2], CORPUS[3]])
    for query in ("widget", "calibration", "sprocket"):
        assert index.scores(query) == pytest.approx(survivors.scores(query))


def test_compacted_on_an_index_with_no_tombstones_changes_nothing():
    index = BM25Index(CORPUS)
    before = index.scores("widget error")
    index = index.compacted()
    assert index.scores("widget error") == pytest.approx(before)
    assert index.size == len(CORPUS)


def test_adding_after_compaction_still_scores_correctly():
    """compacted() renumbers slots, so an add that followed it would collide
    with a live document if the next slot were computed from anything but
    the compacted size."""
    index = BM25Index(CORPUS)
    index.remove_slots([0, 1])
    index = index.compacted()
    index.add_documents(_docs("widget frobnication error nine thousand"))

    scores = index.scores("frobnication")
    assert scores[2] > 0
    assert scores[:2] == [0.0, 0.0]


def test_scores_array_matches_scores():
    index = BM25Index(CORPUS)
    assert index.scores_array("widget error").tolist() == pytest.approx(
        index.scores("widget error")
    )


def test_normalize_array_matches_normalize():
    import numpy as np

    raw = [0.0, 2.5, 5.0]
    assert normalize_array(np.asarray(raw)).tolist() == pytest.approx(normalize(raw))


def test_normalize_array_handles_all_equal_and_empty():
    import numpy as np

    assert normalize_array(np.zeros(3)).tolist() == [0.0, 0.0, 0.0]
    assert normalize_array(np.asarray([])).tolist() == []


def test_compacted_leaves_the_original_index_untouched():
    """It returns a new index rather than mutating, so the caller can build
    it without excluding readers and swap it in afterwards. A reader still
    holding the old one has to keep seeing a coherent index."""
    index = BM25Index(CORPUS)
    index.remove_slots([1])
    before = index.scores("widget")

    compacted = index.compacted()

    assert index.size == 4 and index.tombstones == 1
    assert index.scores("widget") == pytest.approx(before)
    assert compacted.size == 3 and compacted.tombstones == 0
