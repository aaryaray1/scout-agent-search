"""The ChunkStore contract (scout/store/).

Everything above the segmented-store divider runs against every backend,
with JsonChunkStore as the reference: a case that passes there and fails on
the segmented store is a regression, not a new expectation.
"""
import json

import numpy as np
import pytest

from scout.index import IndexCorruptError, save_ingested
from scout.store import BACKENDS, JsonChunkStore, SegmentedChunkStore, make_store


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


@pytest.fixture(params=sorted(BACKENDS), ids=sorted(BACKENDS))
def store_class(request):
    """Every contract case, once per implementation."""
    return BACKENDS[request.param]


@pytest.fixture
def store(store_class):
    return store_class()


def test_make_store_rejects_an_unknown_backend():
    with pytest.raises(ValueError, match="unknown chunk_store"):
        make_store("qdrant")


def test_load_on_an_empty_store_returns_nothing_rather_than_an_empty_matrix(store):
    """The caller sizes its own empty matrix: only it knows how wide the
    current embedding model's vectors are."""
    chunks, embeddings = store.load()
    assert chunks == []
    assert embeddings is None


def test_upsert_then_load_round_trips(store, store_class):
    store.upsert([_chunk("s://a"), _chunk("s://a", 1)], _embeddings(2))

    chunks, embeddings = store_class().load()
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


def test_a_write_whose_halves_disagree_is_refused(store):
    """Two chunks and one vector is a caller bug that does not raise on its
    own: it shifts every later chunk onto the wrong vector, and persists."""
    with pytest.raises(ValueError):
        store.upsert([_chunk("s://a"), _chunk("s://a", 1)], _embeddings(1))
    assert store.load() == ([], None)


def test_delete_sources_removes_only_the_named_sources(store):
    store.upsert([_chunk("s://a"), _chunk("s://b"), _chunk("s://c")], _embeddings(3))
    store.delete_sources(["s://b"])

    chunks, _ = store.load()
    assert [c["source"] for c in chunks] == ["s://a", "s://c"]


def test_delete_sources_survives_a_reopen(store, store_class):
    store.upsert([_chunk("s://a"), _chunk("s://b")], _embeddings(2))
    store.delete_sources(["s://a"])

    chunks, _ = store_class().load()
    assert [c["source"] for c in chunks] == ["s://b"]


def test_delete_sources_is_a_no_op_for_something_not_stored(store):
    store.upsert([_chunk("s://a")], _embeddings(1))
    store.delete_sources(["s://nothing"])
    assert store.stats().live_chunks == 1


def test_stats_counts_live_chunks_and_distinct_sources(store):
    store.upsert([_chunk("s://a", 0), _chunk("s://a", 1), _chunk("s://b")], _embeddings(3))
    stats = store.stats()
    assert stats.live_chunks == 3
    assert stats.sources == 2
    # Nothing has been replaced, so nothing is dead in either store.
    assert stats.dead_chunks == 0


def test_an_empty_upsert_writes_nothing(store, store_class):
    store.upsert([], _embeddings(0))
    assert store_class().load() == ([], None)


# -- writing without having read first ---------------------------------------
# An instance that skipped load() knows nothing, and acting on that destroys
# data in both backends. load() is not required, and these prove it.


def test_upsert_without_calling_load_first_keeps_what_is_already_stored(store_class):
    store_class().upsert([_chunk("s://a"), _chunk("s://b")], _embeddings(2))

    fresh = store_class()  # never load()ed
    fresh.upsert([_chunk("s://c")], _embeddings(1, fill=2.0))

    chunks, embeddings = store_class().load()
    assert [c["source"] for c in chunks] == ["s://a", "s://b", "s://c"]
    assert embeddings.shape == (3, 4)


def test_delete_sources_without_calling_load_first_actually_deletes(store_class):
    store_class().upsert([_chunk("s://a"), _chunk("s://b")], _embeddings(2))

    store_class().delete_sources(["s://a"])

    chunks, _ = store_class().load()
    assert [c["source"] for c in chunks] == ["s://b"]


def test_stats_without_calling_load_first_reports_what_is_on_disk(store_class):
    store_class().upsert([_chunk("s://a"), _chunk("s://b")], _embeddings(2))
    assert store_class().stats().live_chunks == 2


# -- corruption --------------------------------------------------------------
# Not parametrized, since which file holds the store differs per backend. The
# rule does not: an unreadable ingested store must fail loudly rather than
# read as empty and let the next ingest overwrite recoverable data.


def test_a_corrupt_json_store_raises_rather_than_reading_as_empty(tmp_path):
    JsonChunkStore().upsert([_chunk("s://a")], _embeddings(1))

    (tmp_path / "index" / "ingested_meta.json").write_text("{ nope", encoding="utf-8")

    with pytest.raises(IndexCorruptError):
        JsonChunkStore().load()


def test_a_corrupt_manifest_raises_rather_than_reading_as_empty(tmp_path):
    SegmentedChunkStore().upsert([_chunk("s://a")], _embeddings(1))

    (tmp_path / "index" / "ingested" / "manifest.json").write_text("{ nope", encoding="utf-8")

    with pytest.raises(IndexCorruptError):
        SegmentedChunkStore().load()


def test_a_manifest_that_is_not_a_manifest_raises(tmp_path):
    """Valid JSON of the wrong shape reads as an empty store unless
    something checks, and an empty read is the destructive one."""
    SegmentedChunkStore().upsert([_chunk("s://a")], _embeddings(1))

    (tmp_path / "index" / "ingested" / "manifest.json").write_text("[]", encoding="utf-8")

    with pytest.raises(IndexCorruptError):
        SegmentedChunkStore().load()


def test_a_manifest_from_a_newer_format_is_refused(tmp_path):
    SegmentedChunkStore().upsert([_chunk("s://a")], _embeddings(1))

    path = tmp_path / "index" / "ingested" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["format"] = 99
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(IndexCorruptError, match="format"):
        SegmentedChunkStore().load()


def test_a_missing_segment_raises_rather_than_being_skipped(tmp_path):
    """A segment the manifest names but that isn't there means pages are
    gone. Skipping it would serve a silently incomplete corpus."""
    store = SegmentedChunkStore()
    store.upsert([_chunk("s://a")], _embeddings(1))
    store.upsert([_chunk("s://b")], _embeddings(1))

    (tmp_path / "index" / "ingested" / "seg-000001.npy").unlink()

    with pytest.raises(IndexCorruptError):
        SegmentedChunkStore().load()


def test_a_segment_whose_halves_disagree_raises(tmp_path):
    """Chunks and embeddings are parallel arrays. A segment where they have
    drifted apart keeps working and attributes every score to the wrong
    chunk, which is worse than refusing to load."""
    SegmentedChunkStore().upsert([_chunk("s://a")], _embeddings(1))

    path = tmp_path / "index" / "ingested" / "seg-000001.npy"
    np.save(str(path), _embeddings(2))

    with pytest.raises(IndexCorruptError, match="inconsistent"):
        SegmentedChunkStore().load()


# -- segmented store ---------------------------------------------------------


def _segment_dir(tmp_path):
    return tmp_path / "index" / "ingested"


def _segments(tmp_path):
    return sorted(p.name for p in _segment_dir(tmp_path).glob("seg-*.json"))


def test_an_ingest_writes_one_new_segment_and_touches_no_existing_one(tmp_path):
    """The whole point of the format: writing costs the page being written,
    not the size of everything already stored."""
    store = SegmentedChunkStore()
    store.upsert([_chunk("s://a")], _embeddings(1))
    first = _segment_dir(tmp_path) / "seg-000001.json"
    before = first.read_bytes()

    store.upsert([_chunk("s://b")], _embeddings(1))

    assert _segments(tmp_path) == ["seg-000001.json", "seg-000002.json"]
    assert first.read_bytes() == before


def test_a_replaced_source_stays_on_disk_until_a_merge(tmp_path):
    """Superseding is a manifest entry, not a rewrite, so the old chunks are
    still there. stats() is how that dead weight is visible."""
    store = SegmentedChunkStore()
    store.upsert([_chunk("s://a", 0, "v1")], _embeddings(1))
    store.upsert([_chunk("s://a", 0, "v2")], _embeddings(1))

    stats = store.stats()
    assert (stats.live_chunks, stats.dead_chunks, stats.segments) == (1, 1, 2)
    assert store.load()[0][0]["content"] == "v2"


def test_enough_segments_trigger_a_merge_without_changing_what_is_stored(tmp_path):
    """A load opens two files per segment, so segments are bounded even when
    nothing is dead. The merge is the one O(store) operation, amortized
    across the ingests that caused it."""
    store = SegmentedChunkStore()
    for i in range(40):
        store.upsert([_chunk(f"s://{i}")], _embeddings(1, fill=float(i)))

    assert store.stats().segments < 40
    chunks, embeddings = SegmentedChunkStore().load()
    assert [c["source"] for c in chunks] == [f"s://{i}" for i in range(40)]
    assert [e[0] for e in embeddings] == [float(i) for i in range(40)]


def test_a_merge_reclaims_superseded_chunks(tmp_path):
    """Dead weight past the threshold is what a merge exists to reclaim, and
    the files behind it go with it."""
    store = SegmentedChunkStore()
    store.upsert([_chunk("s://a", i) for i in range(200)], _embeddings(200))
    store.upsert([_chunk("s://a", 0, "replaced")], _embeddings(1, fill=2.0))

    stats = store.stats()
    assert stats.dead_chunks == 0
    assert stats.live_chunks == 1
    assert len(_segments(tmp_path)) == stats.segments
    assert SegmentedChunkStore().load()[0][0]["content"] == "replaced"


def test_deleting_everything_leaves_a_readable_empty_store(tmp_path):
    store = SegmentedChunkStore()
    store.upsert([_chunk("s://a", i) for i in range(200)], _embeddings(200))
    store.delete_sources(["s://a"])

    assert store.stats().live_chunks == 0
    assert SegmentedChunkStore().load() == ([], None)


def test_a_segment_the_manifest_never_committed_is_ignored_and_removed(tmp_path):
    """A crash between writing a segment and writing the manifest that would
    have committed it. The manifest is the record of what exists, so an
    uncommitted segment holds nothing a reader could ever have seen."""
    SegmentedChunkStore().upsert([_chunk("s://a")], _embeddings(1))

    orphan = _segment_dir(tmp_path) / "seg-000009.json"
    orphan.write_text(json.dumps([_chunk("s://ghost")]), encoding="utf-8")
    np.save(str(_segment_dir(tmp_path) / "seg-000009.npy"), _embeddings(1))

    chunks, _ = SegmentedChunkStore().load()
    assert [c["source"] for c in chunks] == ["s://a"]
    assert not orphan.exists()


def test_a_pre_segment_store_is_imported_rather_than_ignored(tmp_path):
    """Upgrading Scout must not start the ingested store empty: the next
    ingest would supersede the only copy of every page already in it."""
    save_ingested(
        _embeddings(2, fill=7.0),
        [_chunk("s://old", 0), _chunk("s://old", 1)],
        index_dir=str(tmp_path / "index"),
    )

    chunks, embeddings = SegmentedChunkStore().load()
    assert [c["source"] for c in chunks] == ["s://old", "s://old"]
    assert embeddings.shape == (2, 4)
    assert embeddings[0][0] == 7.0
    # And the import is durable, not recomputed from the legacy files on
    # every startup.
    assert (_segment_dir(tmp_path) / "manifest.json").exists()


def test_an_imported_store_still_accepts_writes(tmp_path):
    save_ingested(
        _embeddings(1), [_chunk("s://old")], index_dir=str(tmp_path / "index")
    )

    store = SegmentedChunkStore()
    store.upsert([_chunk("s://new")], _embeddings(1, fill=2.0))

    chunks, _ = SegmentedChunkStore().load()
    assert [c["source"] for c in chunks] == ["s://old", "s://new"]


def test_the_manifest_does_not_grow_on_deletes_that_delete_nothing(tmp_path):
    """An entry naming nothing live changes nothing and is the one way this
    file could grow without bound."""
    store = SegmentedChunkStore()
    store.upsert([_chunk("s://a")], _embeddings(1))
    path = _segment_dir(tmp_path) / "manifest.json"
    before = path.read_bytes()

    for _ in range(50):
        store.delete_sources(["s://never-stored"])

    assert path.read_bytes() == before
