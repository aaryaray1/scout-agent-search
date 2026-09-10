import pytest

from scout.search import Retriever

def test_search():
    retriever = Retriever()
    results = retriever.search("Error 1008")
    assert len(results) > 0
    assert "insufficient balance" in results[0]["content"].lower()


def test_search_respects_top_k():
    retriever = Retriever()
    results = retriever.search("authentication token", top_k=1)
    assert len(results) == 1


def test_search_results_are_sorted_by_confidence():
    retriever = Retriever()
    results = retriever.search("rate limit exceeded")
    confidences = [r["confidence"] for r in results]
    assert confidences == sorted(confidences, reverse=True)


def test_search_keyword_score_is_zero_when_bm25_has_no_match():
    retriever = Retriever()
    # No demo doc mentions this term, so BM25 has nothing to match and
    # every keyword_score should come back 0.
    results = retriever.search("teleportation")
    assert all(r["metadata"]["keyword_score"] == 0.0 for r in results)


def test_search_keyword_score_ranks_exact_term_match_highest():
    retriever = Retriever()
    retriever.add_chunks([{
        "id": "kw-test-1",
        "content": "Zorbnificator maintenance procedure and troubleshooting steps.",
        "source": "kw-test",
        "type": "web",
        "order": 0,
    }])
    results = retriever.search("zorbnificator", top_k=1)
    assert results[0]["source"] == "kw-test"
    # The only possible match, so min-max normalization lands it at 1.0.
    assert results[0]["metadata"]["keyword_score"] == 1.0


def test_add_chunks_keeps_embeddings_in_sync():
    retriever = Retriever()
    before = len(retriever.corpus)

    new_chunk = {
        "id": "test-chunk-1",
        "content": "Widget frobnication failures are logged under error code 9001.",
        "source": "test",
        "type": "web",
        "order": 0,
    }
    retriever.add_chunks([new_chunk])

    assert len(retriever.corpus) == before + 1
    # Parallel arrays: out of lockstep, search() zips the wrong documents.
    assert retriever.corpus_embeddings.shape[0] == len(retriever.corpus)

    results = retriever.search("widget frobnication error 9001", top_k=1)
    assert results[0]["source"] == "test"


def test_ingested_content_survives_a_new_retriever_instance():
    """add_chunks() persists ingested pages to disk; a fresh Retriever
    (standing in for a process restart) should pick that content back up
    without anyone calling add_chunks() again."""
    first = Retriever()
    before = len(first.corpus)

    first.add_chunks([{
        "id": "persist-test-1",
        "content": "Persisted widget frobnication guide for error code 4242.",
        "source": "https://example.com/persisted",
        "type": "web",
        "order": 0,
    }])
    assert len(first.corpus) == before + 1

    second = Retriever()
    assert len(second.corpus) == before + 1
    results = second.search("widget frobnication error 4242", top_k=1)
    assert results[0]["source"] == "https://example.com/persisted"


def test_a_pre_segment_store_is_still_searchable_after_an_upgrade():
    """The upgrade path a real deployment hits. Covered at the store level in
    tests/test_store.py; covered here because it is the retriever that has to
    come up holding the pages, and starting empty would let the next ingest
    supersede the only copy of them."""
    import numpy as np

    from scout.embeddings import EmbeddingModel
    from scout.index import save_ingested

    content = "Legacy quernstone bearings fail under fault code Q-8812."
    chunk = {
        "id": "legacy-1",
        "content": content,
        "source": "https://example.com/legacy",
        "type": "web",
        "order": 0,
    }
    embeddings = np.asarray(EmbeddingModel().encode([content]))
    save_ingested(embeddings, [chunk])

    retriever = Retriever()

    results = retriever.search("quernstone bearing fault", top_k=1)
    assert results[0]["source"] == "https://example.com/legacy"
    # A second start does not double-count the page. Whether the import was
    # written durably is asserted at the store level, in tests/test_store.py.
    assert len(Retriever().corpus) == len(retriever.corpus)


def test_re_ingesting_same_source_replaces_instead_of_duplicating():
    retriever = Retriever()
    before = len(retriever.corpus)
    source = "https://example.com/changelog"

    retriever.add_chunks([{
        "id": "dedup-v1",
        "content": "Version 1.0 released with initial widget support.",
        "source": source,
        "type": "web",
        "order": 0,
    }])
    assert len(retriever.corpus) == before + 1

    # Re-ingesting the same source should replace its chunk, not add a
    # second one alongside it.
    retriever.add_chunks([{
        "id": "dedup-v2",
        "content": "Version 2.0 released with frobnication support.",
        "source": source,
        "type": "web",
        "order": 0,
    }])
    assert len(retriever.corpus) == before + 1
    assert retriever.corpus_embeddings.shape[0] == len(retriever.corpus)

    results = retriever.search("frobnication support version 2.0", top_k=1)
    assert results[0]["source"] == source
    assert "2.0" in results[0]["content"]

    # the stale v1.0 content should be gone, not just outranked
    all_content = " ".join(c["content"] for c in retriever.corpus)
    assert "initial widget support" not in all_content


def test_dedup_persists_correctly_across_restart():
    source = "https://example.com/changelog"
    first = Retriever()
    before = len(first.corpus)

    first.add_chunks([{
        "id": "dedup-r1",
        "content": "Version 1.0 release notes for the frobnicator.",
        "source": source,
        "type": "web",
        "order": 0,
    }])
    first.add_chunks([{
        "id": "dedup-r2",
        "content": "Version 2.0 release notes for the frobnicator.",
        "source": source,
        "type": "web",
        "order": 0,
    }])
    assert len(first.corpus) == before + 1

    second = Retriever()
    assert len(second.corpus) == before + 1
    results = second.search("frobnicator version 2.0 release notes", top_k=1)
    assert "2.0" in results[0]["content"]


def test_retriever_starts_with_no_local_markdown_corpus(tmp_path):
    """An ingest-only deployment has no docs_path corpus at all. Refusing to
    start there (the previous behaviour) made Scout unhostable without
    shipping a demo corpus alongside it."""
    retriever = Retriever(docs_path=str(tmp_path))
    assert retriever.corpus == []
    assert retriever.search("anything at all") == []


def test_ingest_only_retriever_can_still_serve_what_it_ingested(tmp_path):
    retriever = Retriever(docs_path=str(tmp_path))
    retriever.add_chunks([{
        "id": "web-only-1",
        "content": "Frobnicator calibration fails with error code 7331.",
        "source": "https://example.com/only",
        "type": "web",
        "order": 0,
    }])

    results = retriever.search("frobnicator calibration 7331", top_k=1)
    assert results[0]["source"] == "https://example.com/only"


def test_add_chunks_ignores_an_empty_batch(tmp_path):
    retriever = Retriever(docs_path=str(tmp_path))
    retriever.add_chunks([])
    assert retriever.corpus == []


@pytest.mark.parametrize("bad_top_k", [0, -1])
def test_search_rejects_a_non_positive_top_k(bad_top_k):
    """`top_k or self.top_k` used to turn 0 into the default and let a
    negative value slice from the end of the ranking."""
    retriever = Retriever()
    with pytest.raises(ValueError, match="top_k must be >= 1"):
        retriever.search("rate limit", top_k=bad_top_k)


def test_search_clamps_top_k_to_the_configured_ceiling():
    retriever = Retriever()
    results = retriever.search("rate limit", top_k=10_000)
    assert len(results) <= retriever.max_top_k


def test_search_returns_at_most_the_whole_corpus():
    retriever = Retriever()
    results = retriever.search("rate limit", top_k=retriever.max_top_k)
    assert len(results) == min(retriever.max_top_k, len(retriever.corpus))


def test_corpus_and_embeddings_stay_the_same_length_after_a_replacing_ingest():
    retriever = Retriever()
    for version in ["first version of the page", "second version of the page"]:
        retriever.add_chunks([{
            "id": f"len-{version}",
            "content": version,
            "source": "https://example.com/versioned",
            "type": "web",
            "order": 0,
        }])
        assert retriever.corpus_embeddings.shape[0] == len(retriever.corpus)


def test_search_many_answers_each_query_independently():
    retriever = Retriever()
    batched = retriever.search_many(["Error 1008", "rate limit exceeded"], top_k=2)
    assert len(batched) == 2
    assert all(len(results) == 2 for results in batched)
    assert batched[0] == retriever.search("Error 1008", top_k=2)
    assert batched[1] == retriever.search("rate limit exceeded", top_k=2)


def test_search_many_embeds_every_query_in_one_call():
    """Batching exists so a sentence-transformer sees the whole batch at
    once. A loop over search() would encode once per query, which is the
    cost this method is meant to avoid."""
    retriever = Retriever()
    calls = []
    original = retriever.embed_model.encode

    def counting_encode(texts):
        calls.append(list(texts))
        return original(texts)

    retriever.embed_model = type(
        "Counting", (), {"encode": staticmethod(counting_encode)}
    )()
    retriever.search_many(["one", "two", "three"])

    assert calls == [["one", "two", "three"]]


def test_search_many_on_an_empty_batch_returns_nothing():
    retriever = Retriever()
    assert retriever.search_many([]) == []


def test_search_many_validates_top_k_once_for_the_batch():
    retriever = Retriever()
    with pytest.raises(ValueError):
        retriever.search_many(["a", "b"], top_k=0)


# -- keyword index alignment -------------------------------------------------
# _bm25_slots is what keeps a keyword score attached to the chunk it was
# computed for. Getting it wrong does not raise, it silently scores the wrong
# document. See docs/design/retrieval.md on slots.


def _page(source, term, count=1):
    return [
        {
            "id": f"{source}-{i}",
            "content": f"Document about {term} number {i} and other filler words.",
            "source": source,
            "type": "web",
            "order": i,
        }
        for i in range(count)
    ]


def test_keyword_scores_stay_attached_to_their_chunk_after_a_replacement():
    """Re-ingesting a source tombstones its slots. Every surviving chunk
    keeps its old slot, so the mapping has to drop exactly the dead entries
    and nothing else."""
    retriever = Retriever()
    for source, term in [("s://a", "quernstone"), ("s://b", "bandersnatch"), ("s://c", "flimflammery")]:
        retriever.add_chunks(_page(source, term))

    # Tombstoning a slot in the middle: an off-by-one shows up here and not
    # when only the last source is replaced.
    retriever.add_chunks(_page("s://b", "wobblegong"))

    for term, expected in [
        ("quernstone", "s://a"),
        ("flimflammery", "s://c"),
        ("wobblegong", "s://b"),
    ]:
        top = retriever.search(term, top_k=1)[0]
        assert top["source"] == expected, f"'{term}' resolved to {top['source']}"
        assert top["metadata"]["keyword_score"] > 0


def test_replacing_a_large_source_compacts_the_keyword_index():
    """Tombstones accumulate until compaction reclaims them. Compaction
    renumbers every slot, so the mapping has to be rebuilt with it."""
    retriever = Retriever()
    retriever.add_chunks(_page("s://big", "zephyrology", count=40))
    assert retriever._bm25.tombstones == 0

    retriever.add_chunks(_page("s://big", "quinquagenary", count=40))

    assert retriever._bm25.tombstones == 0, "compaction should have reclaimed the slots"
    assert retriever._bm25.size == len(retriever.corpus)
    assert len(retriever._bm25_slots) == len(retriever.corpus)

    top = retriever.search("quinquagenary", top_k=1)[0]
    assert top["source"] == "s://big"
    assert top["metadata"]["keyword_score"] > 0
    # The replaced version must be gone from the keyword index too, not
    # merely outranked by the new one.
    assert all(
        result["metadata"]["keyword_score"] == 0.0
        for result in retriever.search("zephyrology", top_k=3)
    )


def test_slot_mapping_stays_the_same_length_as_the_corpus():
    retriever = Retriever()
    for i in range(3):
        retriever.add_chunks(_page(f"s://page-{i}", "widget", count=2))
        assert len(retriever._bm25_slots) == len(retriever.corpus)

    retriever.add_chunks(_page("s://page-1", "widget", count=5))
    assert len(retriever._bm25_slots) == len(retriever.corpus)
    assert retriever.corpus_embeddings.shape[0] == len(retriever.corpus)


# -- failure part way through an ingest --------------------------------------
# A failure mid-add_chunks used to leave its five structures disagreeing,
# which surfaces later as every search raising, or as a page that serves
# results until the next restart and then disappears.


def test_a_failed_write_leaves_the_page_unsearchable(monkeypatch):
    """Durable before searchable. If the store write fails the caller gets
    an error, and the page must not be live in memory: it would answer
    queries until the next restart and then vanish with nothing having
    reported a problem."""
    retriever = Retriever()
    before = len(retriever.corpus)

    def failing_write(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(retriever.store, "upsert", failing_write)
    with pytest.raises(OSError):
        retriever.add_chunks(_page("s://unwritable", "sesquipedalian"))

    assert len(retriever.corpus) == before
    assert all(
        result["metadata"]["keyword_score"] == 0.0
        for result in retriever.search("sesquipedalian", top_k=3)
    )


def test_a_failed_merge_leaves_the_retriever_usable(monkeypatch):
    """The parallel structures are built before any of them is assigned, so
    an allocation failure part way through is survivable. Without that, the
    chunk list outgrows the embedding matrix and every subsequent search
    raises on the mismatch."""
    import scout.search as search_module

    retriever = Retriever()
    retriever.add_chunks(_page("s://first", "widget"))
    before = len(retriever.corpus)

    real_vstack = search_module.np.vstack
    monkeypatch.setattr(
        search_module.np, "vstack", lambda *a, **k: (_ for _ in ()).throw(MemoryError())
    )
    with pytest.raises(MemoryError):
        retriever.add_chunks(_page("s://second", "widget"))
    monkeypatch.setattr(search_module.np, "vstack", real_vstack)

    assert len(retriever.corpus) == before
    assert retriever.corpus_embeddings.shape[0] == len(retriever.corpus)
    assert len(retriever._bm25_slots) == len(retriever.corpus)
    # Still serving, and a later ingest still works.
    assert retriever.search("widget", top_k=1)
    retriever.add_chunks(_page("s://third", "widget"))
    assert len(retriever.corpus) == before + 1
