"""The ChunkStore seam (scout/store/base.py).

These cover the contract rather than one implementation, so the segmented
store Phase 2 adds next can be pointed at the same cases. JsonChunkStore is
today's behaviour wrapped in that contract, so anything asserted here is
also a regression test for what Retriever already relies on.
"""
import numpy as np
import pytest

from scout.index import IndexCorruptError
from scout.store import JsonChunkStore


def _chunk(source, order=0, content="some content"):
    return {
        "id": f"{source}-{order}",
        "content": content,
        "source": source,
        "type": "web",
        "order": order,
    }


def _embeddings(n, dim=4, fill=1.0):
    return np.full((n, dim), fill, dtype=float)


@pytest.fixture
def store():
    return JsonChunkStore()


def test_load_on_an_empty_store_returns_nothing_rather_than_an_empty_matrix():
    """The caller sizes its own empty matrix: only it knows how wide the
    current embedding model's vectors are."""
    chunks, embeddings = JsonChunkStore().load()
    assert chunks == []
    assert embeddings is None


def test_upsert_then_load_round_trips(store):
    store.upsert([_chunk("s://a"), _chunk("s://a", 1)], _embeddings(2))

    chunks, embeddings = JsonChunkStore().load()
    assert [c["source"] for c in chunks] == ["s://a", "s://a"]
    assert embeddings.shape == (2, 4)


def test_upsert_replaces_a_source_instead_of_appending_to_it(store):
    store.upsert([_chunk("s://a", 0, "v1"), _chunk("s://a", 1, "v1")], _embeddings(2))
    store.upsert([_chunk("s://a", 0, "v2")], _embeddings(1, fill=2.0))

    chunks, embeddings = store.load()
    assert [c["content"] for c in chunks] == ["v2"]
    # Chunk count changed between versions, which is why replacement is by
    # source and not by matching up chunk ids.
    assert embeddings.shape == (1, 4)
    assert embeddings[0][0] == 2.0


def test_upsert_leaves_other_sources_alone(store):
    store.upsert([_chunk("s://a")], _embeddings(1, fill=1.0))
    store.upsert([_chunk("s://b")], _embeddings(1, fill=2.0))
    store.upsert([_chunk("s://a", 0, "updated")], _embeddings(1, fill=3.0))

    chunks, embeddings = store.load()
    by_source = {c["source"]: e for c, e in zip(chunks, embeddings)}
    assert set(by_source) == {"s://a", "s://b"}
    assert by_source["s://b"][0] == 2.0
    assert by_source["s://a"][0] == 3.0


def test_embeddings_stay_aligned_with_chunks_across_replacements(store):
    """The failure this guards against does not raise: it scores every chunk
    against another chunk's vector."""
    for i in range(4):
        store.upsert([_chunk(f"s://{i}")], _embeddings(1, fill=float(i)))
    store.upsert([_chunk("s://1", 0, "replaced"), _chunk("s://1", 1, "replaced")],
                 _embeddings(2, fill=9.0))

    chunks, embeddings = store.load()
    assert len(chunks) == embeddings.shape[0]
    for chunk, embedding in zip(chunks, embeddings):
        expected = 9.0 if chunk["source"] == "s://1" else float(chunk["source"][-1])
        assert embedding[0] == expected


def test_delete_sources_removes_only_the_named_sources(store):
    store.upsert([_chunk("s://a"), _chunk("s://b"), _chunk("s://c")], _embeddings(3))
    store.delete_sources(["s://b"])

    chunks, _ = store.load()
    assert [c["source"] for c in chunks] == ["s://a", "s://c"]


def test_delete_sources_is_a_no_op_for_something_not_stored(store):
    store.upsert([_chunk("s://a")], _embeddings(1))
    store.delete_sources(["s://nothing"])
    assert store.stats().live_chunks == 1


def test_stats_counts_live_chunks_and_distinct_sources(store):
    store.upsert([_chunk("s://a", 0), _chunk("s://a", 1), _chunk("s://b")], _embeddings(3))
    stats = store.stats()
    assert stats.live_chunks == 3
    assert stats.sources == 2
    # Nothing is tombstoned in a store that rewrites itself in full.
    assert stats.dead_chunks == 0


def test_an_empty_upsert_writes_nothing(store):
    store.upsert([], _embeddings(0))
    assert JsonChunkStore().load() == ([], None)


def test_a_corrupt_store_raises_rather_than_reading_as_empty(store, tmp_path):
    """The ingested store is the only copy of every page ever ingested, so
    a store that cannot be read has to fail loudly. Reading as empty would
    let the next ingest overwrite still-recoverable data."""
    store.upsert([_chunk("s://a")], _embeddings(1))

    meta = tmp_path / "index" / "ingested_meta.json"
    meta.write_text("{ this is not json", encoding="utf-8")

    with pytest.raises(IndexCorruptError):
        JsonChunkStore().load()


# -- writing without having read first ---------------------------------------
#
# Every write here is a full rewrite, so a store that writes while believing
# itself empty replaces every page ever ingested with whatever was in that
# one call. Nothing in the interface says load() has to be called first, and
# these assert it does not have to be.


def test_upsert_without_calling_load_first_keeps_what_is_already_stored():
    JsonChunkStore().upsert([_chunk("s://a"), _chunk("s://b")], _embeddings(2))

    fresh = JsonChunkStore()  # never load()ed
    fresh.upsert([_chunk("s://c")], _embeddings(1, fill=2.0))

    chunks, embeddings = JsonChunkStore().load()
    assert [c["source"] for c in chunks] == ["s://a", "s://b", "s://c"]
    assert embeddings.shape == (3, 4)


def test_delete_sources_without_calling_load_first_actually_deletes():
    JsonChunkStore().upsert([_chunk("s://a"), _chunk("s://b")], _embeddings(2))

    JsonChunkStore().delete_sources(["s://a"])

    chunks, _ = JsonChunkStore().load()
    assert [c["source"] for c in chunks] == ["s://b"]


def test_stats_without_calling_load_first_reports_what_is_on_disk():
    JsonChunkStore().upsert([_chunk("s://a"), _chunk("s://b")], _embeddings(2))
    assert JsonChunkStore().stats().live_chunks == 2
