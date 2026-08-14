import numpy as np
from scout.index import (
    save_index, load_index,
    save_ingested, load_ingested,
    compute_corpus_hash,
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
