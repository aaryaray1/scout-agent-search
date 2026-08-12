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
