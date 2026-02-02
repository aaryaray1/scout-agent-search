import json
import numpy as np
from .embeddings import EmbeddingModel
from .ingest import load_markdown_docs, chunk_docs
from .keyword import keyword_score

class Retriever:
    def __init__(self):
        raw_docs = load_markdown_docs()
        self.corpus = chunk_docs(raw_docs)

        self.embed_model = EmbeddingModel()
        self.corpus_embeddings = self.embed_model.encode(
            [d["content"] for d in self.corpus]
        )

    def search(self, query, top_k=3):
        query_vec = self.embed_model.encode([query])[0]
        results = []

        for i, doc_vec in enumerate(self.corpus_embeddings):
            doc = self.corpus[i]

            # Vector similarity
            vector_sim = np.dot(query_vec, doc_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
            )

            # Keyword overlap
            key_sim = keyword_score(query, doc["content"])

            # Weighted final score
            final_score = (
                0.65 * vector_sim +
                0.35 * key_sim
            )

            results.append({
                "content": doc["content"],
                "source": doc["source"],
                "type": doc.get("type", "documentation"),
                "confidence": round(float(final_score), 3),
                "metadata": {
                    "vector_score": round(float(vector_sim), 3),
                    "keyword_score": round(float(key_sim), 3),
                    "id": doc["id"]
                }
            })

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:top_k]

    def normalize_scores(scores):
        if not scores:
            return []

        min_s = min(scores)
        max_s = max(scores)

        if max_s == min_s:
            return [1.0 for _ in scores]

        return [(s - min_s) / (max_s - min_s) for s in scores]
