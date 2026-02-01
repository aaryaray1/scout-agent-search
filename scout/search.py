import json
import numpy as np
from .embeddings import EmbeddingModel

class Retriever:
    def __init__(self, corpus_path="data/corpus.json"):
        self.corpus_path = corpus_path
        self.corpus = json.load(open(corpus_path))
        self.embed_model = EmbeddingModel()
        self.corpus_embeddings = self.embed_model.encode([d["content"] for d in self.corpus])

    def search(self, query, top_k=3):
        # dummy scoring using simple semantic similarity
        query_vec = self.embed_model.encode([query])[0]
        scores = []
        for i, doc_vec in enumerate(self.corpus_embeddings):
            score = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scores[:top_k]:
            doc = self.corpus[idx]
            results.append({
                "content": doc["content"],
                "source": "dummy_corpus",
                "confidence": float(score),
                "metadata": {"id": doc["id"]}
            })
        return results
