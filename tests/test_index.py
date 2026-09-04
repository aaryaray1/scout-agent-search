import numpy as np
import pytest

from scout.index import (
    IndexCorruptError,
    compute_corpus_hash,
    load_index,
    load_ingested,
    save_index,
    save_ingested,
)


def test_save_and_load_index_round_trip(tmp_path):
    embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])
    metadata = [{"id": "1", "content": "a"}, {"id": "2", "content": "b"}]
    corpus_hash = compute_corpus_hash(metadata)

    save_index(embeddings, metadata, corpus_hash, index_dir=str(tmp_path))
    loaded = load_index(index_dir=str(tmp_path))

    assert loaded is not None
    loaded_embeddings, loaded_metadata, loaded_hash = loaded
    np.testing.assert_array_equal(loaded_embeddings, embeddings)
    assert loaded_metadata == metadata
    assert loaded_hash == corpus_hash


def test_load_index_returns_none_when_absent(tmp_path):
    assert load_index(index_dir=str(tmp_path)) is None


def test_save_and_load_ingested_round_trip(tmp_path):
    embeddings = np.array([[5.0, 6.0]])
    metadata = [{"id": "web-1", "content": "ingested page", "source": "https://example.com"}]

    save_ingested(embeddings, metadata, index_dir=str(tmp_path))
    loaded = load_ingested(index_dir=str(tmp_path))

    assert loaded is not None
    loaded_embeddings, loaded_metadata = loaded
    np.testing.assert_array_equal(loaded_embeddings, embeddings)
    assert loaded_metadata == metadata


def test_load_ingested_returns_none_when_absent(tmp_path):
    assert load_ingested(index_dir=str(tmp_path)) is None


def test_load_ingested_returns_none_when_store_is_empty(tmp_path):
    # Nothing has ever been ingested: an empty store should read back as
    # "nothing here" rather than an empty-but-present result.
    save_ingested(np.empty((0, 4)), [], index_dir=str(tmp_path))
    assert load_ingested(index_dir=str(tmp_path)) is None


def test_saves_are_atomic_and_leave_no_partial_files(tmp_path):
    """Writes land through a temp file and os.replace, so a crash mid-write
    can't leave a truncated store that poisons the next startup."""
    save_ingested(np.array([[1.0, 2.0]]), [{"id": "1", "content": "a"}], index_dir=str(tmp_path))
    leftovers = [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


def test_save_overwrites_previous_contents_completely(tmp_path):
    save_ingested(
        np.array([[1.0, 2.0]]),
        [{"id": "1", "content": "a much longer first version of this record"}],
        index_dir=str(tmp_path),
    )
    save_ingested(np.array([[3.0, 4.0]]), [{"id": "2", "content": "b"}], index_dir=str(tmp_path))

    embeddings, metadata = load_ingested(index_dir=str(tmp_path))
    assert metadata == [{"id": "2", "content": "b"}]
    np.testing.assert_array_equal(embeddings, np.array([[3.0, 4.0]]))


def test_corpus_hash_changes_when_a_document_is_renamed(tmp_path):
    """Hashing content alone would serve a cached index that attributes
    every chunk to the wrong source after a rename."""
    before = [{"source": "a.md", "content": "shared text"}]
    after = [{"source": "b.md", "content": "shared text"}]
    assert compute_corpus_hash(before) != compute_corpus_hash(after)


def test_corpus_hash_is_stable_for_identical_input():
    docs = [{"source": "a.md", "content": "x"}, {"source": "b.md", "content": "y"}]
    assert compute_corpus_hash(docs) == compute_corpus_hash(list(docs))


def test_unreadable_docs_cache_degrades_to_a_cache_miss(tmp_path):
    """The docs cache is rebuildable, so corruption there costs a re-embed
    rather than a failed startup."""
    save_index(np.array([[1.0, 2.0]]), [{"id": "1", "content": "a"}], "somehash", index_dir=str(tmp_path))
    (tmp_path / "meta.json").write_text("{not valid json", encoding="utf-8")

    assert load_index(index_dir=str(tmp_path)) is None


def test_unreadable_ingested_store_raises_instead_of_silently_starting_empty(tmp_path):
    """The ingested store is the only copy of every page ever ingested.
    Starting up empty would let the next ingest overwrite recoverable data,
    so a damaged store has to stop the process instead."""
    save_ingested(np.array([[1.0, 2.0]]), [{"id": "1", "content": "a"}], index_dir=str(tmp_path))
    (tmp_path / "ingested_meta.json").write_text("{not valid json", encoding="utf-8")

    with pytest.raises(IndexCorruptError):
        load_ingested(index_dir=str(tmp_path))


def test_mismatched_ingested_store_is_rejected(tmp_path):
    """Embeddings and chunks are parallel arrays; a store where they've
    drifted apart would silently attribute scores to the wrong chunks."""
    save_ingested(np.array([[1.0, 2.0]]), [{"id": "1", "content": "a"}], index_dir=str(tmp_path))
    np.save(str(tmp_path / "ingested_embeddings.npy"), np.array([[1.0, 2.0], [3.0, 4.0]]))

    with pytest.raises(IndexCorruptError, match="inconsistent"):
        load_ingested(index_dir=str(tmp_path))


def test_non_ascii_content_survives_a_round_trip(tmp_path):
    """Ingested pages routinely carry accents, CJK and smart quotes; the
    store reads and writes UTF-8 explicitly rather than inheriting whatever
    the host's default encoding happens to be."""
    content = "Erreur de facturation - solde insuffisant / 支払い残高不足 / “smart quotes”"
    save_ingested(np.array([[1.0, 2.0]]), [{"id": "1", "content": content}], index_dir=str(tmp_path))

    _embeddings, metadata = load_ingested(index_dir=str(tmp_path))
    assert metadata[0]["content"] == content
