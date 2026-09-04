"""Concurrency guarantees for the Retriever.

FastAPI runs Scout's sync endpoints in a threadpool, so a /api/v1/search
can land in the middle of a /api/v1/ingest. self.corpus,
self.corpus_embeddings and self._bm25 are three parallel structures that
have to be swapped as a group: a search that saw a new corpus against
stale embeddings would score chunks against the wrong vectors and return
confident nonsense, and one that caught the BM25 index mid-rebuild would
read a term table that no longer matches its document count.

These tests hammer that window rather than asserting on the locks
themselves, so they keep their meaning if the implementation changes. Both
were checked against a deliberately broken Retriever and both fail on it:
the first on an unsynchronized corpus swap, the second on an unsynchronized
ingest. An earlier third test asserted the same invariant by polling
_snapshot() directly and was dropped -- the window between two adjacent
attribute assignments is too narrow to sample, so it passed against the
broken version and would only have bought false confidence.

Worker loops are bounded rather than run-until-stopped: an unbounded
searcher re-embeds its query every pass, turning a second of contention
into minutes of test time for no extra coverage.
"""
import threading

from scout.search import Retriever


def _run_concurrently(workers):
    """Run every callable in `workers` on its own thread and re-raise the
    first failure on the main thread."""
    errors = []

    def guarded(fn):
        def run():
            try:
                fn()
            except BaseException as e:  # noqa: BLE001 - re-raised below
                errors.append(e)
        return run

    threads = [threading.Thread(target=guarded(w)) for w in workers]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)

    assert not any(t.is_alive() for t in threads), "worker thread deadlocked"
    if errors:
        raise errors[0]


def _chunk(source, order, text):
    return {
        "id": f"{source}-{order}",
        "content": text,
        "source": source,
        "type": "web",
        "order": order,
    }


def test_searches_during_ingest_return_well_formed_results():
    """End-to-end version of the above: a search racing an ingest should
    return real evidence, not raise and not come back misaligned."""
    retriever = Retriever()
    seen = []

    def ingest():
        for i in range(6):
            retriever.add_chunks([
                _chunk(f"https://example.com/doc-{i}", 0, f"Error {5000 + i} means the widget stalled.")
            ])

    def search():
        for _ in range(12):
            for result in retriever.search("widget stalled error", top_k=3):
                assert result["content"]
                assert result["source"]
                assert 0.0 <= result["metadata"]["keyword_score"] <= 1.0
                seen.append(result["source"])

    _run_concurrently([ingest, search, search])
    assert seen, "the searchers never returned a result"


def test_concurrent_ingests_of_the_same_source_leave_exactly_one_version():
    """Two ingests of the same URL racing each other must still end with a
    single version of that page, not an interleaved mix of both."""
    retriever = Retriever()
    baseline = len(retriever.corpus)
    source = "https://example.com/contested"

    def ingest(version):
        def run():
            for _ in range(4):
                retriever.add_chunks([_chunk(source, 0, f"Version {version} of the contested page.")])
        return run

    _run_concurrently([ingest("A"), ingest("B")])

    assert len(retriever.corpus) == baseline + 1
    assert retriever.corpus_embeddings.shape[0] == len(retriever.corpus)

    survivors = [c for c in retriever.corpus if c["source"] == source]
    assert len(survivors) == 1
