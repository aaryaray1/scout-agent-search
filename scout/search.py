import numpy as np
from .embeddings import EmbeddingModel
from .ingest import load_markdown_docs, chunk_docs
from .bm25 import BM25Index, normalize
from .index import load_index, save_index, compute_corpus_hash, load_ingested, save_ingested
from .config import load_config


class Retriever:
    """Hybrid (vector + BM25 keyword) retriever over a chunked markdown
    corpus.

    The live corpus is two layers concatenated together:

    - the docs_path corpus (self._docs_*): loaded once at startup, cached
      to disk by content hash (scout.index.compute_corpus_hash), so
      scout-ingest and the API stay in sync on the exact same corpus.
    - the ingested corpus (self._ingested_*): pages added later through
      add_chunks()/`/api/v1/ingest`, persisted so they survive a restart.

    self.corpus / self.corpus_embeddings / self._bm25 are derived from
    both layers, recomputed by _rebuild_combined_corpus() whenever either
    layer changes. Keeping them as two separate layers (rather than one
    flat list) is what makes de-duplication on re-ingest tractable:
    dropping a stale source only ever touches the ingested layer, never
    the docs_path corpus.
    """

    def __init__(self, docs_path=None, top_k=None):
        config = load_config()
        self.docs_path = docs_path or config["docs_path"]
        self.top_k = top_k or config["top_k"]
        self.vector_weight = config["vector_weight"]
        self.keyword_weight = config["keyword_weight"]

        self.embed_model = EmbeddingModel(config["vector_model"])

        raw_docs = load_markdown_docs(self.docs_path)
        docs_corpus = chunk_docs(raw_docs)
        if not docs_corpus:
            raise ValueError(
                f"No documents found under '{self.docs_path}'. "
                "Add markdown files or point docs_path at a populated folder."
            )

        current_hash = compute_corpus_hash(docs_corpus)
        cached = load_index()
        if cached and cached[2] == current_hash:
            docs_embeddings = cached[0]
        else:
            docs_embeddings = self.embed_model.encode(
                [d["content"] for d in docs_corpus]
            )
            save_index(docs_embeddings, docs_corpus, current_hash)

        self._docs_corpus = docs_corpus
        self._docs_embeddings = docs_embeddings

        self._bm25 = BM25Index()
        self._load_persisted_ingest()

    def _load_persisted_ingest(self):
        """Load the ingested layer from disk (or start it empty) and fold
        it into the live corpus, so content added through /api/v1/ingest
        survives a restart instead of only living in memory for one
        process.
        """
        ingested = load_ingested()
        if ingested:
            embeddings, chunks = ingested
            self._ingested_chunks = chunks
            self._ingested_embeddings = embeddings
        else:
            self._ingested_chunks = []
            self._ingested_embeddings = np.empty((0, self._docs_embeddings.shape[1]))
        self._rebuild_combined_corpus()

    def _rebuild_combined_corpus(self):
        self.corpus = self._docs_corpus + self._ingested_chunks
        self.corpus_embeddings = np.vstack([self._docs_embeddings, self._ingested_embeddings])
        self._bm25.build(self.corpus)

    def _drop_ingested_sources(self, sources):
        """Remove any already-ingested chunks whose source is in `sources`.

        Called before adding new chunks for a page, so re-ingesting a URL
        replaces its old content instead of accumulating duplicate chunks
        alongside it forever in the persisted store.
        """
        if not self._ingested_chunks:
            return
        keep = [c["source"] not in sources for c in self._ingested_chunks]
        if all(keep):
            return
        self._ingested_chunks = [c for c, k in zip(self._ingested_chunks, keep) if k]
        self._ingested_embeddings = self._ingested_embeddings[keep]

    def search(self, query, top_k=None):
        top_k = top_k or self.top_k
        query_vec = self.embed_model.encode([query])[0]

        # Vectorized cosine similarity against the whole corpus at once.
        doc_norms = np.linalg.norm(self.corpus_embeddings, axis=1)
        query_norm = np.linalg.norm(query_vec)
        denom = doc_norms * query_norm
        denom[denom == 0] = 1e-8  # avoid div-by-zero for empty/zero vectors
        vector_sims = (self.corpus_embeddings @ query_vec) / denom

        # BM25 scores are unbounded, so normalize to [0, 1] before fusing
        # with cosine similarity -- otherwise a strong keyword match could
        # swamp vector_weight/keyword_weight's intended balance.
        key_sims = normalize(self._bm25.scores(query))

        results = []
        for doc, vector_sim, key_sim in zip(self.corpus, vector_sims, key_sims):
            final_score = (
                self.vector_weight * vector_sim +
                self.keyword_weight * key_sim
            )
            results.append({
                "content": doc["content"],
                "source": doc.get("source", "unknown"),
                "type": doc.get("type", "documentation"),
                "confidence": round(float(final_score), 3),
                "metadata": {
                    "vector_score": round(float(vector_sim), 3),
                    "keyword_score": round(float(key_sim), 3),
                    "id": doc.get("id"),
                },
            })

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:top_k]

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
        self._drop_ingested_sources(sources)

        new_embeddings = self.embed_model.encode([c["content"] for c in chunks])
        self._ingested_chunks.extend(chunks)
        self._ingested_embeddings = np.vstack([self._ingested_embeddings, new_embeddings])

        save_ingested(self._ingested_embeddings, self._ingested_chunks)
        self._rebuild_combined_corpus()
