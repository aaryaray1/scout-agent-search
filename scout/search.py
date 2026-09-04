"""Hybrid retrieval over the combined corpus."""
import logging
import threading

import numpy as np

from .bm25 import BM25Index, normalize
from .config import load_config
from .embeddings import EmbeddingModel
from .index import compute_corpus_hash, load_index, load_ingested, save_index, save_ingested
from .ingest import chunk_docs, load_markdown_docs

logger = logging.getLogger(__name__)


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

    self.corpus / self.corpus_embeddings / self._bm25 are derived from
    both layers, replaced as a group by _rebuild_combined_corpus()
    whenever either layer changes. Keeping them as two separate layers
    (rather than one flat list) is what makes de-duplication on re-ingest
    tractable: dropping a stale source only ever touches the ingested
    layer, never the docs_path corpus.

    Concurrency: FastAPI runs the sync endpoints in a threadpool, so a
    search can land mid-ingest. Two locks keep that safe without making
    readers wait on slow work:

    - _ingest_lock serializes add_chunks() against itself, covering the
      ingested layer and its on-disk store.
    - _swap_lock covers only the three-attribute handover in
      _rebuild_combined_corpus() and the matching read in _snapshot(), so
      a search can never observe a new corpus against stale embeddings or
      a stale BM25 index.

    Embedding and disk writes happen outside _swap_lock, so an ingest
    never blocks a concurrent search for longer than the swap itself.
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
        self._swap_lock = threading.Lock()

        self._docs_corpus, self._docs_embeddings = self._load_docs_layer()
        self._ingested_chunks, self._ingested_embeddings = self._load_ingested_layer()
        self._rebuild_combined_corpus()
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

    def _rebuild_combined_corpus(self):
        """Rebuild the searchable view from the two layers.

        Builds a whole new BM25 index rather than rebuilding the existing
        one in place, so a concurrent search either sees the old index or
        the new one, never one mid-build.
        """
        corpus = self._docs_corpus + self._ingested_chunks
        embeddings = np.vstack([self._docs_embeddings, self._ingested_embeddings])
        bm25 = BM25Index(corpus)

        with self._swap_lock:
            self.corpus = corpus
            self.corpus_embeddings = embeddings
            self._bm25 = bm25

    def _snapshot(self):
        """Take a consistent view of the three parallel structures, so the
        rest of a search can run outside the lock."""
        with self._swap_lock:
            return self.corpus, self.corpus_embeddings, self._bm25

    # -- querying ------------------------------------------------------------

    def _resolve_top_k(self, top_k):
        if top_k is None:
            return self.top_k
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        return min(top_k, self.max_top_k)

    def _fused_scores(self, query, embeddings, bm25):
        """Blend cosine similarity with normalized BM25 into one score per
        corpus position. Returns (fused, vector_sims, keyword_sims)."""
        query_vec = self.embed_model.encode([query])[0]
        vector_sims = _cosine_similarity(embeddings, query_vec)
        # BM25 scores are unbounded, so normalize to [0, 1] before fusing
        # with cosine similarity -- otherwise a strong keyword match could
        # swamp vector_weight/keyword_weight's intended balance.
        keyword_sims = np.asarray(normalize(bm25.scores(query)), dtype=float)
        fused = self.vector_weight * vector_sims + self.keyword_weight * keyword_sims
        return fused, vector_sims, keyword_sims

    def search(self, query, top_k=None):
        """Return the top_k best-matching chunks as evidence dicts."""
        top_k = self._resolve_top_k(top_k)
        corpus, embeddings, bm25 = self._snapshot()
        if not corpus:
            return []

        fused, vector_sims, keyword_sims = self._fused_scores(query, embeddings, bm25)
        # Rank on the score array, then build result dicts only for the
        # chunks actually being returned. The previous version built a dict
        # for every chunk in the corpus on every query and threw away all
        # but top_k of them.
        ranked = np.argsort(-fused, kind="stable")[:top_k]
        return [
            _as_evidence(corpus[i], fused[i], vector_sims[i], keyword_sims[i])
            for i in ranked
        ]

    # -- ingestion -----------------------------------------------------------

    def _drop_ingested_sources(self, sources):
        """Remove any already-ingested chunks whose source is in `sources`.

        Called before adding new chunks for a page, so re-ingesting a URL
        replaces its old content instead of accumulating duplicate chunks
        alongside it forever in the persisted store.
        """
        if not self._ingested_chunks:
            return
        keep = np.array(
            [c["source"] not in sources for c in self._ingested_chunks], dtype=bool
        )
        if keep.all():
            return
        self._ingested_chunks = [c for c, k in zip(self._ingested_chunks, keep) if k]
        self._ingested_embeddings = self._ingested_embeddings[keep]

    def add_chunks(self, chunks):
        """Embed `chunks`, replace any existing chunks that share a
        source, and persist the result.

        Used by the ingest pipeline (scout.web) so a freshly-fetched page
        is searchable immediately. Re-ingesting a URL that's already in the
        ingested store drops its old chunks first, so the store holds one
        version per source instead of piling up duplicates. This still
        isn't the real Phase 2 vector store (see ROADMAP.md): every call
        rewrites the whole ingested store to disk, which is fine at
        prototype scale but not how this should work once ingest volume
        grows.
        """
        if not chunks:
            return

        sources = {c["source"] for c in chunks}
        # Embedding is the slow part and touches no shared state, so it runs
        # before the lock is taken.
        new_embeddings = self.embed_model.encode([c["content"] for c in chunks])

        with self._ingest_lock:
            self._drop_ingested_sources(sources)
            self._ingested_chunks = self._ingested_chunks + list(chunks)
            self._ingested_embeddings = np.vstack(
                [self._ingested_embeddings, new_embeddings]
            )
            save_ingested(self._ingested_embeddings, self._ingested_chunks)
            self._rebuild_combined_corpus()
        logger.info("indexed %d chunks from %d source(s)", len(chunks), len(sources))
