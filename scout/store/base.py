"""The seam between the retriever and wherever ingested pages actually live.

`Retriever` currently talks to `scout.index.save_ingested` / `load_ingested`
directly, which means the storage strategy is welded to the retriever. That
was fine while there was exactly one strategy. Phase 2 introduces a second
(an append-only segmented store) and leaves the door open to a third (an
external vector database), so the strategy needs a name and a contract.

This module is that contract, plus one implementation that wraps today's
behaviour unchanged. It exists so the segmented store can be written and
tested against a defined interface rather than by editing `Retriever` in
place, and so switching between them is a config value rather than a diff.

**Deliberately not a vector-search interface.** There is no `search()` here.
Scout fuses dense and sparse scores over one in-memory view of the corpus,
so retrieval reads the whole embedding matrix rather than delegating a
nearest-neighbour query. An external vector database would want the
opposite, and the honest thing to record is that adopting one is a change to
retrieval, not only to storage. See
docs/architecture/adr-001-incremental-index.md for the measurements behind
deferring that, and the conditions that would reopen it. Inventing the
delegated-search half of the interface now, with no implementation to check
it against, would be guessing at a shape rather than discovering it.

**Sources are the unit of replacement.** Every chunk carries a `source` (a
URL for ingested pages), and re-ingesting a URL replaces that source's chunks
rather than adding to them. That rule belongs to the store, because it is
what a segmented implementation records as a tombstone and what an external
one would express as a filtered delete.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..index import load_ingested, save_ingested


def _stack(existing, new):
    """vstack that tolerates an absent or empty left-hand side.

    A store's first write has nothing to stack onto, and np.vstack against a
    (0, 0) array raises rather than broadcasting, so the empty case is
    handled here instead of at every call site.
    """
    new = np.asarray(new)
    if existing is None or len(existing) == 0:
        return new
    return np.vstack([existing, new])


@dataclass(frozen=True)
class StoreStats:
    """What a store can report about itself, for /health and for deciding
    when to compact."""

    live_chunks: int
    sources: int
    # Chunks written but no longer live. Always 0 for a store that rewrites
    # itself in full; meaningful for one that appends and tombstones.
    dead_chunks: int = 0
    segments: int = 1


class ChunkStore(ABC):
    """Durable storage for chunks brought in through /api/v1/ingest.

    Implementations own crash safety. The existing rules are the contract
    and are not negotiable per backend: a partially written store must never
    be readable as a complete one, and a store that cannot be read back must
    raise rather than present itself as empty, because it holds the only
    copy of every page ever ingested. `scout.index` earns that with a temp
    file plus an atomic rename; anything else has to earn it its own way.

    Not thread-safe. `Retriever` serializes writes behind its ingest lock,
    which is also where the embedding and the in-memory update happen, so
    pushing locking down here would only duplicate it.
    """

    @abstractmethod
    def load(self):
        """Return `(chunks, embeddings)` for everything live in the store.

        Returns `([], None)` when nothing has been ingested yet, leaving the
        caller to size an empty matrix against its own embedding model
        rather than having this guess a width.
        """

    @abstractmethod
    def upsert(self, chunks, embeddings):
        """Persist `chunks` and their `embeddings`, replacing every chunk
        already stored under any source appearing in `chunks`.

        Replacement is by source rather than by chunk id because a page that
        gained or lost a section produces a different number of chunks than
        the version it replaces, so matching them up individually would
        leave the difference behind as orphans.
        """

    @abstractmethod
    def delete_sources(self, sources):
        """Remove every chunk belonging to any of `sources`."""

    @abstractmethod
    def stats(self):
        """Return a StoreStats describing what is currently held."""


class JsonChunkStore(ChunkStore):
    """The store Scout has today: one embeddings file and one metadata file,
    both rewritten in full on every write.

    Kept as the reference implementation, and as the thing a segmented store
    has to stay behaviourally identical to. Its weakness is the whole reason
    Phase 2 exists: `upsert` costs the size of everything ever ingested, not
    the size of what changed, which is 238ms per ingest at 20k chunks and
    grows linearly (`scripts/bench_index.py`).

    It holds the live set in memory between calls because that is what
    rewriting in full requires: there is no way to replace one source's
    chunks in this format without rewriting every other source's too.
    """

    def __init__(self, index_dir=None):
        self._index_dir = index_dir
        self._chunks = []
        self._embeddings = None
        # Distinct from "the store is empty": an instance that has not read
        # from disk yet knows nothing, and must not be allowed to write on
        # the strength of that. See _ensure_loaded.
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
        """Read the store from disk before any operation that depends on
        what it already holds.

        Without this, a store whose caller never happened to call load()
        writes as though the store were empty, and since every write is a
        full rewrite, that silently replaces every page ever ingested with
        whatever was in the current call. The interface never says load()
        must be called first, and it should not have to: an operation that
        destroys data when a caller skips an optional step is a bug in the
        store, not in the caller.
        """
        if not self._loaded:
            self.load()

    def _drop(self, sources):
        """Remove `sources` from the in-memory view without writing.

        Split from delete_sources() because upsert() needs the same removal
        but must not flush a store that is missing the replacement chunks
        it is about to add: a crash in that window would lose the page
        rather than leave the previous version in place.
        """
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
            sources=len({c["source"] for c in self._chunks}),
        )
