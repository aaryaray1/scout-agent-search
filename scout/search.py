"""Hybrid retrieval over the combined corpus."""
import logging
import threading

import numpy as np

from .bm25 import BM25Index, normalize_array
from .config import load_config
from .embeddings import EmbeddingModel
from .index import compute_corpus_hash, load_index, load_ingested, save_index, save_ingested
from .ingest import chunk_docs, load_markdown_docs

logger = logging.getLogger(__name__)

# Tombstoned BM25 slots cost memory and make every scores() call allocate a
# longer list than it needs, but compaction invalidates every slot number
# the retriever holds, so it isn't free either. Compact only once the dead
# weight is worth the renumbering: a meaningful share of the index, and more
# than a handful of documents, so a tiny corpus doesn't compact on every
# re-ingest.
_COMPACT_TOMBSTONE_RATIO = 0.3
_COMPACT_MIN_TOMBSTONES = 32


def _where(items, mask, wanted=True):
    """Filter `items` by a parallel boolean `mask`.

    The three ingested-layer lists (chunks, slots, embeddings) have to be
    filtered by the same mask or they stop describing the same documents, so
    the filtering is written once rather than three times.
    """
    return [item for item, flag in zip(items, mask) if flag == wanted]


def _cosine_similarity(matrix, vector):
    """Cosine similarity of `vector` against every row of `matrix`."""
    denom = np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector)
    denom[denom == 0] = 1e-8  # avoid div-by-zero for empty/zero vectors
    return (matrix @ vector) / denom


def _as_evidence(doc, score, vector_sim, keyword_sim):
    """Shape one corpus chunk into the response dict the API returns."""
    return {
        "content": doc["content"],
        "source": doc.get("source", "unknown"),
        "type": doc.get("type", "documentation"),
        "confidence": round(float(score), 3),
        "metadata": {
            "vector_score": round(float(vector_sim), 3),
            "keyword_score": round(float(keyword_sim), 3),
            "id": doc.get("id"),
        },
    }


class Retriever:
    """Hybrid (vector + BM25 keyword) retriever over a chunked corpus.

    The live corpus is two layers concatenated together:

    - the docs_path corpus (self._docs_*): loaded once at startup, cached
      to disk by content hash (scout.index.compute_corpus_hash), so
      scout-ingest and the API stay in sync on the exact same corpus.
    - the ingested corpus (self._ingested_*): pages added later through
      add_chunks()/`/api/v1/ingest`, persisted so they survive a restart.

    self.corpus / self.corpus_embeddings / self._bm25_slots are derived
    from both layers and replaced as a group whenever either layer
    changes. Keeping them as two separate layers (rather than one flat
    list) is what makes de-duplication on re-ingest tractable: dropping a
    stale source only ever touches the ingested layer, never the docs_path
    corpus.

    **Slots.** self._bm25 addresses documents by slot, not by corpus
    position, because removing a document leaves a tombstone behind rather
    than shifting everything after it (see scout/bm25.py). self._bm25_slots
    is the mapping: for corpus position i, self._bm25_slots[i] is where
    that chunk lives in the keyword index. Scoring indexes one through the
    other, which is a numpy gather rather than a Python loop.

    Concurrency: FastAPI runs the sync endpoints in a threadpool, so a
    search can land mid-ingest. Two locks keep that safe:

    - _ingest_lock serializes add_chunks() against itself, covering the
      ingested layer and its on-disk store.
    - _index_lock covers the searchable view: the corpus, the embeddings,
      the slot mapping and the BM25 index itself, on both the read and the
      write side.

    _index_lock covering reads is a change from the original design, which
    kept readers lock-free by rebuilding a whole new BM25 index on every
    ingest and swapping it in. That rebuild re-tokenized the entire corpus
    and cost 3.4 seconds per ingest at 20k chunks, which is what
    docs/architecture/adr-001-incremental-index.md replaced with an index
    mutated in place. An index mutated in place cannot be read
    concurrently -- a search iterating a posting list while an ingest
    inserts into it raises RuntimeError -- so readers take the lock.

    That is affordable for the same reason the change was worth making:
    mutation is now proportional to what changed rather than to the corpus,
    and BM25 scoring is pure Python, which the GIL already serializes
    across threads. The work that genuinely parallelizes -- embedding a
    query, the numpy similarity scan, disk writes -- all happens outside
    the lock.

    The exception worth knowing about is /api/v1/search/batch, which scores
    up to max_batch_queries queries in one locked section rather than one.
    The GIL argument still applies to each query, but the hold time
    multiplies, and that endpoint is not rate limited, so a few concurrent
    batch callers can hold up an ingest for noticeably longer than a single
    search would. Scoring per query instead would shorten it at the cost of
    the batch's guarantee that every query in it sees the same corpus,
    which is the endpoint's reason to exist. Tracked in ROADMAP.md Phase 5.
    """

    def __init__(self, docs_path=None, top_k=None):
        config = load_config()
        self.docs_path = docs_path or config["docs_path"]
        self.top_k = top_k or config["top_k"]
        self.max_top_k = config["max_top_k"]
        self.vector_weight = config["vector_weight"]
        self.keyword_weight = config["keyword_weight"]

        self.embed_model = EmbeddingModel(config["vector_model"])

        self._ingest_lock = threading.Lock()
        self._index_lock = threading.Lock()

        self._docs_corpus, self._docs_embeddings = self._load_docs_layer()
        self._ingested_chunks, self._ingested_embeddings = self._load_ingested_layer()

        # Docs first, then ingested, so slot order matches corpus order and
        # a compaction can reset the mapping to a plain range.
        self._bm25 = BM25Index()
        self._docs_slots = self._bm25.add_documents(self._docs_corpus)
        self._ingested_slots = self._bm25.add_documents(self._ingested_chunks)
        with self._index_lock:
            self._swap_in_combined_corpus()

        logger.info(
            "retriever ready: %d docs chunks, %d ingested chunks",
            len(self._docs_corpus), len(self._ingested_chunks),
        )

    # -- corpus construction -------------------------------------------------

    def _empty_embeddings(self):
        return np.empty((0, self.embed_model.dimension))

    def _load_docs_layer(self):
        """Load and embed the markdown corpus under docs_path.

        An empty docs_path is a warning, not an error: an ingest-only
        deployment (the whole point of /api/v1/ingest) has no local
        markdown at all, and refusing to start there would make Scout
        unhostable without shipping a demo corpus alongside it.
        """
        corpus = chunk_docs(load_markdown_docs(self.docs_path))
        if not corpus:
            logger.warning(
                "no markdown documents under '%s'; starting with an empty docs "
                "corpus. Only ingested pages will be searchable.", self.docs_path,
            )
            return [], self._empty_embeddings()
        return corpus, self._embeddings_for(corpus)

    def _embeddings_for(self, corpus):
        """Return embeddings for `corpus`, reusing the on-disk cache when it
        still matches."""
        corpus_hash = compute_corpus_hash(corpus)
        cached = load_index()
        if cached and self._cache_is_usable(cached, corpus, corpus_hash):
            return cached[0]

        logger.info("embedding %d docs chunks (index cache miss)", len(corpus))
        embeddings = self.embed_model.encode([d["content"] for d in corpus])
        save_index(embeddings, corpus, corpus_hash)
        return embeddings

    def _cache_is_usable(self, cached, corpus, corpus_hash):
        """A cache hit needs the corpus hash, the row count and the vector
        width to all line up.

        The hash alone isn't enough: switching SCOUT_VECTOR_MODEL leaves a
        cache built by the previous model, and stacking those vectors
        against freshly embedded ones would either raise deep inside numpy
        or silently score against the wrong space.
        """
        embeddings, _metadata, cached_hash = cached
        if cached_hash != corpus_hash:
            return False
        if len(embeddings) != len(corpus):
            logger.warning("index cache row count doesn't match the corpus; re-embedding")
            return False
        if embeddings.shape[1] != self.embed_model.dimension:
            logger.warning(
                "index cache was built with a %d-dim model, current model is %d-dim; "
                "re-embedding", embeddings.shape[1], self.embed_model.dimension,
            )
            return False
        return True

    def _load_ingested_layer(self):
        """Load pages added through /api/v1/ingest in an earlier run."""
        ingested = load_ingested()
        if not ingested:
            return [], self._empty_embeddings()
        embeddings, chunks = ingested
        return chunks, embeddings

    def _swap_in_combined_corpus(self):
        """Rebuild the searchable view from the two layers.

        Caller must hold _index_lock: these three attributes are what a
        search reads together, and a search that saw a new corpus against
        stale embeddings would score chunks against the wrong vectors.
        Cheap enough to do under the lock -- a list concatenation, one
        numpy copy and one small integer array -- unlike the BM25 rebuild
        this used to also perform.
        """
        # Built into locals first: assigning self.corpus before the vstack
        # that follows it would, if that allocation failed, leave a corpus
        # longer than its embedding matrix and make every search raise.
        corpus = self._docs_corpus + self._ingested_chunks
        embeddings = np.vstack([self._docs_embeddings, self._ingested_embeddings])
        slots = np.asarray(self._docs_slots + self._ingested_slots, dtype=np.intp)

        self.corpus = corpus
        self.corpus_embeddings = embeddings
        self._bm25_slots = slots

    # -- querying ------------------------------------------------------------

    def _resolve_top_k(self, top_k):
        if top_k is None:
            return self.top_k
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        return min(top_k, self.max_top_k)

    def _snapshot(self, queries):
        """Take a consistent view of the searchable state, and score the
        keyword half of every query against it.

        BM25 scoring happens in here, inside the lock, because self._bm25
        is mutated in place by add_chunks() (see the class docstring).
        Gathering through _bm25_slots drops tombstoned slots and puts the
        scores in corpus order, so everything downstream can work in corpus
        positions and forget slots exist.
        """
        with self._index_lock:
            slots = self._bm25_slots
            return (
                self.corpus,
                self.corpus_embeddings,
                [self._bm25.scores_array(query)[slots] for query in queries],
            )

    def _fused_scores(self, query_vec, embeddings, keyword_raw):
        """Blend cosine similarity with normalized BM25 into one score per
        corpus position. Returns (fused, vector_sims, keyword_sims)."""
        vector_sims = _cosine_similarity(embeddings, query_vec)
        # BM25 scores are unbounded, so normalize to [0, 1] before fusing
        # with cosine similarity -- otherwise a strong keyword match could
        # swamp vector_weight/keyword_weight's intended balance.
        keyword_sims = normalize_array(keyword_raw)
        fused = self.vector_weight * vector_sims + self.keyword_weight * keyword_sims
        return fused, vector_sims, keyword_sims

    def _rank(self, query_vec, corpus, embeddings, keyword_raw, top_k):
        """Score one query against a snapshot and shape the winners."""
        fused, vector_sims, keyword_sims = self._fused_scores(
            query_vec, embeddings, keyword_raw
        )
        # Rank on the score array, then build result dicts only for the
        # chunks actually being returned. The previous version built a dict
        # for every chunk in the corpus on every query and threw away all
        # but top_k of them.
        ranked = np.argsort(-fused, kind="stable")[:top_k]
        return [
            _as_evidence(corpus[i], fused[i], vector_sims[i], keyword_sims[i])
            for i in ranked
        ]

    def search(self, query, top_k=None):
        """Return the top_k best-matching chunks as evidence dicts."""
        return self.search_many([query], top_k=top_k)[0]

    def search_many(self, queries, top_k=None):
        """Answer several queries at once, returning one result list each,
        in the order asked.

        Worth having as its own method rather than a loop over search():
        it takes one corpus snapshot for the whole batch, so every query in
        it is answered against the same index even if an ingest lands
        mid-batch, and it embeds all the queries in a single call to the
        model, which is where a sentence-transformer's batching actually
        pays off.
        """
        top_k = self._resolve_top_k(top_k)
        if not queries:
            return []
        # Read without the lock purely as a fast path, so an ingest-only
        # deployment that hasn't ingested anything yet doesn't run the
        # embedding model to answer with nothing. It's a single attribute
        # read, and the worst a race costs is one wasted encode.
        if not self.corpus:
            return [[] for _ in queries]

        query_vecs = self.embed_model.encode(list(queries))
        corpus, embeddings, keyword_scores = self._snapshot(queries)
        return self._rank_batch(query_vecs, corpus, embeddings, keyword_scores, top_k)

    def _rank_batch(self, query_vecs, corpus, embeddings, keyword_scores, top_k):
        """Rank every query in a batch against one snapshot.

        The corpus is re-checked here rather than trusted from the fast path
        above: between that read and the snapshot it can only have grown,
        but a corpus that is still empty at snapshot time would index into
        empty arrays.
        """
        if not corpus:
            return [[] for _ in keyword_scores]
        return [
            self._rank(query_vec, corpus, embeddings, keyword_raw, top_k)
            for query_vec, keyword_raw in zip(query_vecs, keyword_scores)
        ]

    # -- ingestion -----------------------------------------------------------

    def _without_sources(self, sources):
        """The ingested layer with `sources` removed, computed but not
        applied. Returns (chunks, slots, embeddings, slots_to_tombstone).

        Deliberately free of side effects. add_chunks() builds the whole new
        layer before assigning any of it, so that an allocation failure part
        way through leaves the retriever exactly as it was rather than with
        a chunk list longer than its embedding matrix -- a mismatch that
        would make every subsequent search raise, and would be persisted.
        """
        keep = [c["source"] not in sources for c in self._ingested_chunks]
        if all(keep):
            return self._ingested_chunks, self._ingested_slots, self._ingested_embeddings, []
        return (
            _where(self._ingested_chunks, keep),
            _where(self._ingested_slots, keep),
            self._ingested_embeddings[np.array(keep, dtype=bool)],
            _where(self._ingested_slots, keep, wanted=False),
        )

    def _publish(self, chunks, kept_slots, dead_slots, merged):
        """Make a prepared layer the live one, under _index_lock.

        Everything in here is proportional to what changed rather than to
        the size of the corpus, which is what makes holding the read lock
        across it acceptable. Compaction is the exception, and runs outside
        the lock for exactly that reason.
        """
        merged_chunks, merged_embeddings = merged
        with self._index_lock:
            self._bm25.remove_slots(dead_slots)
            self._ingested_slots = kept_slots + self._bm25.add_documents(chunks)
            self._ingested_chunks = merged_chunks
            self._ingested_embeddings = merged_embeddings
            self._swap_in_combined_corpus()

    def _compact_if_worthwhile(self):
        """Reclaim tombstoned BM25 slots once enough have accumulated.

        The compacted index is built before the lock is taken, because
        re-keying every posting list is proportional to the whole index
        (600ms at 20k chunks half-tombstoned) and doing that with readers
        excluded would stall every concurrent search for the duration.
        Reading self._bm25 here without the lock is safe: only ingest
        mutates it, and _ingest_lock is held.

        Compaction renumbers, invalidating every slot this class holds.
        That's recoverable in three lines only because it preserves
        insertion order and this corpus is laid out in that same order:
        docs first, then ingested. Both mappings become plain ranges again.
        """
        index = self._bm25
        if index.tombstones < _COMPACT_MIN_TOMBSTONES:
            return
        if index.tombstones < _COMPACT_TOMBSTONE_RATIO * index.size:
            return

        compacted = index.compacted()
        docs_count = len(self._docs_corpus)
        with self._index_lock:
            self._bm25 = compacted
            self._docs_slots = list(range(docs_count))
            self._ingested_slots = list(
                range(docs_count, docs_count + len(self._ingested_chunks))
            )
            self._swap_in_combined_corpus()
        logger.info("compacted the keyword index to %d slots", compacted.size)

    def add_chunks(self, chunks):
        """Embed `chunks`, replace any existing chunks that share a
        source, and persist the result.

        Used by the ingest pipeline (scout.web) so a freshly-fetched page
        is searchable immediately. Re-ingesting a URL that's already in the
        ingested store drops its old chunks first, so the store holds one
        version per source instead of piling up duplicates.

        Ordered so that the durable write happens before the page becomes
        searchable. The reverse would mean a failed write returns an error
        to the caller while the page is live in memory and absent from
        disk, so it would serve results until the next restart and then
        vanish without anything having reported a problem.

        Embedding and the disk write are the slow parts and both happen
        outside _index_lock. The write is still a full rewrite of the
        ingested store, which is the remaining O(everything) step and the
        next item in ROADMAP.md Phase 2.
        """
        if not chunks:
            return

        sources = {c["source"] for c in chunks}
        # Embedding is the slow part and touches no shared state, so it runs
        # before either lock is taken.
        new_embeddings = self.embed_model.encode([c["content"] for c in chunks])

        with self._ingest_lock:
            kept_chunks, kept_slots, kept_embeddings, dead_slots = self._without_sources(sources)
            merged = (
                kept_chunks + list(chunks),
                np.vstack([kept_embeddings, new_embeddings]),
            )
            save_ingested(merged[1], merged[0])
            self._publish(chunks, kept_slots, dead_slots, merged)
            self._compact_if_worthwhile()
        logger.info("indexed %d chunks from %d source(s)", len(chunks), len(sources))
