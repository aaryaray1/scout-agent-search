"""Hybrid (vector + BM25) retrieval over the combined corpus."""
import logging
import threading

import numpy as np

from .bm25 import BM25Index, normalize_array
from .config import load_config
from .embeddings import EmbeddingModel
from .index import compute_corpus_hash, load_index, save_index
from .ingest import chunk_docs, load_markdown_docs
from .store import make_store

logger = logging.getLogger(__name__)

# Tombstones cost memory, compaction costs a renumbering. Reclaim only once
# the dead weight is worth it, so a tiny corpus doesn't compact per ingest.
_COMPACT_TOMBSTONE_RATIO = 0.3
_COMPACT_MIN_TOMBSTONES = 32


def _where(items, mask, wanted=True):
    """Filter `items` by a parallel boolean `mask`.

    Written once because the three ingested-layer lists must be filtered by
    the same mask or they stop describing the same documents.
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
    """Hybrid retriever over two corpus layers: local markdown (_docs_*) and
    ingested pages (_ingested_*), with the searchable view derived from both.

    Concurrency-sensitive. Read docs/design/retrieval.md before editing.
    """

    def __init__(self, docs_path=None, top_k=None):
        config = load_config()
        self.docs_path = docs_path or config["docs_path"]
        self.top_k = top_k or config["top_k"]
        self.max_top_k = config["max_top_k"]
        self.vector_weight = config["vector_weight"]
        self.keyword_weight = config["keyword_weight"]

        self.embed_model = EmbeddingModel(config["vector_model"])
        # How ingested pages are persisted lives entirely behind this.
        self.store = make_store(config["chunk_store"])

        self._ingest_lock = threading.Lock()
        self._index_lock = threading.Lock()

        self._docs_corpus, self._docs_embeddings = self._load_docs_layer()
        self._ingested_chunks, self._ingested_embeddings = self._load_ingested_layer()

        # Docs first, then ingested, so slot order matches corpus order and a
        # compaction can reset both mappings to plain ranges.
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

        Empty is a warning, not an error: an ingest-only deployment has no
        local markdown, and refusing to start would make Scout unhostable
        without shipping a demo corpus.
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
        """Embeddings for `corpus`, reusing the on-disk cache if it matches."""
        corpus_hash = compute_corpus_hash(corpus)
        cached = load_index()
        if cached and self._cache_is_usable(cached, corpus, corpus_hash):
            return cached[0]

        logger.info("embedding %d docs chunks (index cache miss)", len(corpus))
        embeddings = self.embed_model.encode([d["content"] for d in corpus])
        save_index(embeddings, corpus, corpus_hash)
        return embeddings

    def _cache_is_usable(self, cached, corpus, corpus_hash):
        """A hit needs the hash, the row count and the vector width to line
        up -- a changed model leaves a cache in a different vector space."""
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
        chunks, embeddings = self.store.load()
        if not chunks:
            return [], self._empty_embeddings()
        return chunks, embeddings

    def _swap_in_combined_corpus(self):
        """Rebuild the searchable view from the two layers.

        Caller must hold _index_lock: a search that saw a new corpus against
        stale embeddings would score chunks against the wrong vectors.
        """
        # Built into locals first: assigning self.corpus before the vstack
        # would, if that allocation failed, leave a corpus longer than its
        # embedding matrix and make every search raise.
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
        """One consistent view of the searchable state, with the keyword half
        of every query scored against it.

        Scoring is inside the lock because the index is mutated in place; the
        slot gather drops tombstones and returns corpus order.
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
        # BM25 is unbounded, so normalize before fusing or a strong keyword
        # match swamps the configured weights.
        keyword_sims = normalize_array(keyword_raw)
        fused = self.vector_weight * vector_sims + self.keyword_weight * keyword_sims
        return fused, vector_sims, keyword_sims

    def _rank(self, query_vec, corpus, embeddings, keyword_raw, top_k):
        """Score one query against a snapshot and shape the winners."""
        fused, vector_sims, keyword_sims = self._fused_scores(
            query_vec, embeddings, keyword_raw
        )
        # Rank on the score array, then build dicts only for what is
        # returned, rather than one per chunk in the corpus.
        ranked = np.argsort(-fused, kind="stable")[:top_k]
        return [
            _as_evidence(corpus[i], fused[i], vector_sims[i], keyword_sims[i])
            for i in ranked
        ]

    def search(self, query, top_k=None):
        """Return the top_k best-matching chunks as evidence dicts."""
        return self.search_many([query], top_k=top_k)[0]

    def search_many(self, queries, top_k=None):
        """Answer several queries against one corpus snapshot, in order.

        One snapshot for the batch, and one pass through the embedding
        model, which is where a sentence-transformer's batching pays off.
        """
        top_k = self._resolve_top_k(top_k)
        if not queries:
            return []
        # Lock-free fast path so an empty deployment doesn't run the model to
        # answer with nothing. A race costs one wasted encode.
        if not self.corpus:
            return [[] for _ in queries]

        query_vecs = self.embed_model.encode(list(queries))
        corpus, embeddings, keyword_scores = self._snapshot(queries)
        return self._rank_batch(query_vecs, corpus, embeddings, keyword_scores, top_k)

    def _rank_batch(self, query_vecs, corpus, embeddings, keyword_scores, top_k):
        """Rank every query in a batch against one snapshot.

        The corpus is re-checked rather than trusted from the fast path: one
        still empty at snapshot time would index into empty arrays.
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
        applied: (chunks, slots, embeddings, slots_to_tombstone).

        Free of side effects so add_chunks() can build the whole new layer
        before assigning any of it.
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

        Everything here is proportional to what changed, which is what makes
        holding the read lock across it acceptable.
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

        Built outside the lock, since re-keying every posting list is
        O(index); reading self._bm25 unlocked is safe because only ingest
        mutates it and _ingest_lock is held.
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
        """Embed `chunks`, replace any sharing a source, and persist.

        Durable before searchable, and both slow steps outside _index_lock.
        """
        if not chunks:
            return

        sources = {c["source"] for c in chunks}
        # Embedding is slow and touches no shared state, so it runs first.
        new_embeddings = self.embed_model.encode([c["content"] for c in chunks])

        with self._ingest_lock:
            kept_chunks, kept_slots, kept_embeddings, dead_slots = self._without_sources(sources)
            merged = (
                kept_chunks + list(chunks),
                np.vstack([kept_embeddings, new_embeddings]),
            )
            self.store.upsert(chunks, new_embeddings)
            self._publish(chunks, kept_slots, dead_slots, merged)
            self._compact_if_worthwhile()
        logger.info("indexed %d chunks from %d source(s)", len(chunks), len(sources))
