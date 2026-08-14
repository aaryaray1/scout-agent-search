import numpy as np
from .embeddings import EmbeddingModel
from .ingest import load_markdown_docs, chunk_docs
from .keyword import tokenize, score_tokens
from .index import load_index, save_index, compute_corpus_hash, load_ingested, save_ingested
from .config import load_config


class Retriever:
    """Hybrid (vector + keyword) retriever over a chunked markdown corpus.

    Loads docs from `docs_path`, chunks them, and embeds them, reusing the
    on-disk cache built by `scout.cli` when the corpus hasn't changed
    (`scout.index.compute_corpus_hash`), so `scout-ingest` and the API stay
    in sync on the exact same corpus. Also reloads any pages previously
    added through add_chunks()/`/api/v1/ingest` from scout.index's ingested
    store, so those survive a restart too.
    """

    def __init__(self, docs_path=None, top_k=None):
        config = load_config()
        self.docs_path = docs_path or config["docs_path"]
        self.top_k = top_k or config["top_k"]
        self.vector_weight = config["vector_weight"]
        self.keyword_weight = config["keyword_weight"]

        self.embed_model = EmbeddingModel(config["vector_model"])

        raw_docs = load_markdown_docs(self.docs_path)
        self.corpus = chunk_docs(raw_docs)
        if not self.corpus:
            raise ValueError(
                f"No documents found under '{self.docs_path}'. "
                "Add markdown files or point docs_path at a populated folder."
            )

        # Tokenized once per document here, not per query: keyword_score
        # would otherwise re-tokenize every chunk's full text on every
        # search() call, which is wasted work since the corpus is static
        # between ingests.
        self._doc_tokens = [tokenize(d["content"]) for d in self.corpus]

        current_hash = compute_corpus_hash(self.corpus)

        cached = load_index()
        if cached:
            embeddings, metadata, cached_hash = cached
            if cached_hash == current_hash:
                self.corpus_embeddings = embeddings
                self._load_persisted_ingest()
                return

        self.corpus_embeddings = self.embed_model.encode(
            [d["content"] for d in self.corpus]
        )
        save_index(self.corpus_embeddings, self.corpus, current_hash)
        self._load_persisted_ingest()

    def _load_persisted_ingest(self):
        """Fold previously-ingested pages back into the live corpus on
        startup, so content added through /api/v1/ingest survives a
        restart instead of only living in memory for one process.
        """
        ingested = load_ingested()
        if ingested:
            self._ingested_embeddings, self._ingested_chunks = ingested
            self.corpus.extend(self._ingested_chunks)
            self._doc_tokens.extend(tokenize(c["content"]) for c in self._ingested_chunks)
            self.corpus_embeddings = np.vstack([self.corpus_embeddings, self._ingested_embeddings])
        else:
            self._ingested_chunks = []
            self._ingested_embeddings = np.empty((0, self.corpus_embeddings.shape[1]))

    def search(self, query, top_k=None):
        top_k = top_k or self.top_k
        query_vec = self.embed_model.encode([query])[0]
        query_tokens = tokenize(query)

        # Vectorized cosine similarity against the whole corpus at once.
        doc_norms = np.linalg.norm(self.corpus_embeddings, axis=1)
        query_norm = np.linalg.norm(query_vec)
        denom = doc_norms * query_norm
        denom[denom == 0] = 1e-8  # avoid div-by-zero for empty/zero vectors
        vector_sims = (self.corpus_embeddings @ query_vec) / denom

        results = []
        for doc, doc_tokens, vector_sim in zip(self.corpus, self._doc_tokens, vector_sims):
            key_sim = score_tokens(query_tokens, doc_tokens)
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
        """Embed and append `chunks` to the live index, and persist them.

        Used by the ingest pipeline (scout.web) so a freshly-fetched page is
        searchable immediately. Ingested chunks are written to
        scout.index's ingested store on every call, so they're reloaded by
        `_load_persisted_ingest()` the next time a Retriever starts up and
        survive a process restart. This isn't the real Phase 2 vector store
        (see ROADMAP.md): every call rewrites the whole ingested store to
        disk, which is fine at prototype scale but not how this should work
        once ingest volume grows.
        """
        if not chunks:
            return
        new_embeddings = self.embed_model.encode([c["content"] for c in chunks])
        self.corpus.extend(chunks)
        self._doc_tokens.extend(tokenize(c["content"]) for c in chunks)
        self.corpus_embeddings = np.vstack([self.corpus_embeddings, new_embeddings])

        self._ingested_chunks.extend(chunks)
        self._ingested_embeddings = np.vstack([self._ingested_embeddings, new_embeddings])
        save_ingested(self._ingested_embeddings, self._ingested_chunks)
