import threading
from sentence_transformers import SentenceTransformer

_model_cache = {}
_cache_lock = threading.Lock()


def _get_cached_model(model_name):
    """Load `model_name` once per process and reuse it after that.

    SentenceTransformer(model_name) reads weights off disk (or downloads
    them) every time it's called. Nothing about the model changes between
    Retriever instances, so without this every new Retriever(), including
    one per test in the test suite, paid that load cost again.
    """
    if model_name not in _model_cache:
        with _cache_lock:
            if model_name not in _model_cache:  # re-check inside the lock
                _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = _get_cached_model(model_name)

    def encode(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)
