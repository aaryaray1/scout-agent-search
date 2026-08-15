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


def test_add_chunks_keeps_doc_tokens_in_sync():
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
    # _doc_tokens is a parallel cache to self.corpus; add_chunks() must keep
    # both in lockstep or search() zips them against the wrong documents.
    assert len(retriever._doc_tokens) == len(retriever.corpus)

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
    assert len(retriever._doc_tokens) == len(retriever.corpus)
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
