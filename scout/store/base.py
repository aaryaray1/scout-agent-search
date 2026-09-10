"""The ChunkStore contract, plus JsonChunkStore, the full-rewrite backend
Scout used before segments and still tests every other backend against.

See docs/design/storage.md, including why there is deliberately no search().
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..index import load_ingested, save_ingested


def source_counts(chunks):
    """Map each source in `chunks` to how many chunks it contributed."""
    counts = {}
    for chunk in chunks:
        source = chunk["source"]
        counts[source] = counts.get(source, 0) + 1
    return counts


def check_alignment(chunks, embeddings):
    """Refuse a write whose two halves describe different things.

    Mismatched parallel arrays don't raise later, they attribute every score
    to the wrong chunk, and once written that is what the next restart reads.
    """
    if len(embeddings) != len(chunks):
        raise ValueError(
            f"{len(chunks)} chunks against {len(embeddings)} embeddings; "
            "they must be parallel"
        )


def _stack(existing, new):
    """vstack that tolerates an absent or empty left-hand side, since a
    store's first write has nothing to stack onto."""
    new = np.asarray(new)
    if existing is None or len(existing) == 0:
        return new
    return np.vstack([existing, new])


@dataclass(frozen=True)
class StoreStats:
    """What a store reports about itself, for /health and for deciding when
    to compact."""

    live_chunks: int
    sources: int
    # Written but no longer live. Always 0 for a store that rewrites in full.
    dead_chunks: int = 0
    segments: int = 1


class ChunkStore(ABC):
    """Durable storage for chunks brought in through /api/v1/ingest.

    Implementations own crash safety and are not thread-safe; Retriever
    serializes writes behind its ingest lock.
    """

    @abstractmethod
    def load(self):
        """Return `(chunks, embeddings)` for everything live, or `([], None)`
        when empty -- the caller sizes its own empty matrix."""

    @abstractmethod
    def upsert(self, chunks, embeddings):
        """Persist `chunks`, replacing everything stored under any source in
        them -- by source, since a re-ingested page can have a different
        chunk count and matching ids up would leave orphans."""

    @abstractmethod
    def delete_sources(self, sources):
        """Remove every chunk belonging to any of `sources`."""

    @abstractmethod
    def stats(self):
        """Return a StoreStats describing what is currently held."""


class JsonChunkStore(ChunkStore):
    """Two files rewritten in full per write, so a write costs everything
    ever ingested. Holds the live set in memory, which rewriting requires."""

    def __init__(self, index_dir=None):
        self._index_dir = index_dir
        self._chunks = []
        self._embeddings = None
        # Distinct from "the store is empty": see _ensure_loaded.
        self._loaded = False

    def load(self):
        stored = load_ingested(self._index_dir)
        self._loaded = True
        if not stored:
            self._chunks, self._embeddings = [], None
            return [], None
        embeddings, chunks = stored
        self._chunks, self._embeddings = list(chunks), embeddings
        return self._chunks, self._embeddings

    def _ensure_loaded(self):
        """Read from disk before anything that depends on what is there.

        Without it a caller that skipped load() writes as though empty, which
        for a full rewrite destroys every page ever ingested.
        """
        if not self._loaded:
            self.load()

    def _drop(self, sources):
        """Remove `sources` from the in-memory view without writing, since
        upsert() must not flush a store missing the chunks it is about to
        add -- a crash there loses the page entirely."""
        if not self._chunks:
            return
        keep = np.array(
            [c["source"] not in sources for c in self._chunks], dtype=bool
        )
        self._chunks = [c for c, live in zip(self._chunks, keep) if live]
        self._embeddings = self._embeddings[keep]

    def delete_sources(self, sources):
        self._ensure_loaded()
        before = len(self._chunks)
        self._drop(set(sources))
        if len(self._chunks) != before:
            self._flush()

    def upsert(self, chunks, embeddings):
        if not len(chunks):
            return
        check_alignment(chunks, embeddings)
        self._ensure_loaded()
        self._drop({c["source"] for c in chunks})
        self._embeddings = _stack(self._embeddings, embeddings)
        self._chunks = self._chunks + list(chunks)
        self._flush()

    def _flush(self):
        save_ingested(self._embeddings, self._chunks, self._index_dir)

    def stats(self):
        self._ensure_loaded()
        return StoreStats(
            live_chunks=len(self._chunks),
            sources=len(source_counts(self._chunks)),
        )
