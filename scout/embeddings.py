"""Sentence-embedding backend, with the loaded model shared per process."""
import logging
import threading

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_model_cache = {}
_cache_lock = threading.Lock()


def _get_cached_model(model_name):
    """Load `model_name` once per process and reuse it after that.

    SentenceTransformer() re-reads weights off disk on every call, and
    nothing about the model changes between Retriever instances.
    """
    if model_name not in _model_cache:
        with _cache_lock:
            if model_name not in _model_cache:  # re-check inside the lock
                logger.info("loading embedding model '%s'", model_name)
                _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = _get_cached_model(model_name)

    @property
    def dimension(self):
        """Vector width, needed to size an empty matrix when there is
        nothing to embed yet."""
        return self.model.get_sentence_embedding_dimension()

    def encode(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)
