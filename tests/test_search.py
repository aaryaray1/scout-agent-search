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
