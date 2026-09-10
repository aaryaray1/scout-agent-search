"""Durable storage backends for ingested pages. See docs/design/storage.md."""
from ..config import load_config
from .base import ChunkStore, JsonChunkStore, StoreStats
from .segmented import SegmentedChunkStore

BACKENDS = {
    "segmented": SegmentedChunkStore,
    "json": JsonChunkStore,
}

__all__ = [
    "BACKENDS",
    "ChunkStore",
    "JsonChunkStore",
    "SegmentedChunkStore",
    "StoreStats",
    "make_store",
]


def make_store(name=None, index_dir=None):
    """Build the configured chunk store.

    Re-checked here as well as at config load, since this is also a library
    entry point and a caller passing a name deserves the same error.
    """
    name = name or load_config()["chunk_store"]
    if name not in BACKENDS:
        raise ValueError(
            f"unknown chunk_store {name!r}; expected one of {sorted(BACKENDS)}"
        )
    return BACKENDS[name](index_dir)
