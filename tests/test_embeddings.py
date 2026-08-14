from scout.embeddings import EmbeddingModel


def test_same_model_name_reuses_the_loaded_model():
    first = EmbeddingModel()
    second = EmbeddingModel()
    # Both should share the exact same underlying SentenceTransformer
    # instance instead of loading it from disk twice.
    assert first.model is second.model


def test_encode_still_works_after_caching():
    model = EmbeddingModel()
    vectors = model.encode(["hello world"])
    assert vectors.shape[0] == 1
    assert vectors.shape[1] > 0
