import numpy as np
from .embeddings import EmbeddingModel
from .ingest import load_markdown_docs, chunk_docs
from .keyword import keyword_score
from .index import load_index, save_index, compute_corpus_hash
from .config import load_config


class Retriever:
    """Hybrid (vector + keyword) retriever over a chunked markdown corpus.

    Loads docs from `docs_path`, chunks them, and embeds them -- reusing the
    on-disk cache built by `scout.cli` when the corpus hasn't changed
    (`scout.index.compute_corpus_hash`), so `scout-ingest` and the API stay
    in sync on the exact same corpus.
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

        current_hash = compute_corpus_hash(self.corpus)

        cached = load_index()
        if cached:
            embeddings, metadata, cached_hash = cached
            if cached_hash == current_hash:
                self.corpus_embeddings = embeddings
                return

        self.corpus_embeddings = self.embed_model.encode(
            [d["content"] for d in self.corpus]
        )
        save_index(self.corpus_embeddings, self.corpus, current_hash)

    def search(self, query, top_k=None):
        top_k = top_k or self.top_k
        query_vec = self.embed_model.encode([query])[0]

        # Vectorized cosine similarity against the whole corpus at once.
        doc_norms = np.linalg.norm(self.corpus_embeddings, axis=1)
        query_norm = np.linalg.norm(query_vec)
        denom = doc_norms * query_norm
        denom[denom == 0] = 1e-8  # avoid div-by-zero for empty/zero vectors
        vector_sims = (self.corpus_embeddings @ query_vec) / denom

        results = []
        for doc, vector_sim in zip(self.corpus, vector_sims):
            key_sim = keyword_score(query, doc["content"])
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
