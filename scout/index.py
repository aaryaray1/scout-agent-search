import os
import json
import hashlib
import numpy as np

INDEX_DIR = "data/index"


def _paths(index_dir):
    return {
        "emb": os.path.join(index_dir, "embeddings.npy"),
        "meta": os.path.join(index_dir, "meta.json"),
        "hash": os.path.join(index_dir, "corpus_hash.txt"),
        "ingested_emb": os.path.join(index_dir, "ingested_embeddings.npy"),
        "ingested_meta": os.path.join(index_dir, "ingested_meta.json"),
    }


def compute_corpus_hash(docs):
    h = hashlib.sha256()
    for d in docs:
        h.update(d["content"].encode("utf-8"))
    return h.hexdigest()


def save_index(embeddings, metadata, corpus_hash, index_dir=None):
    """Persist the docs_path corpus: a cache keyed by a content hash, so
    scout-ingest can skip rebuilding embeddings when nothing changed."""
    index_dir = index_dir or INDEX_DIR
    paths = _paths(index_dir)
    os.makedirs(index_dir, exist_ok=True)
    np.save(paths["emb"], embeddings)
    json.dump(metadata, open(paths["meta"], "w"), indent=2)
    open(paths["hash"], "w").write(corpus_hash)


def load_index(index_dir=None):
    index_dir = index_dir or INDEX_DIR
    paths = _paths(index_dir)
    if not (os.path.exists(paths["emb"]) and os.path.exists(paths["meta"])):
        return None
    embeddings = np.load(paths["emb"])
    metadata = json.load(open(paths["meta"]))
    corpus_hash = open(paths["hash"]).read()
    return embeddings, metadata, corpus_hash


def save_ingested(embeddings, metadata, index_dir=None):
    """Persist pages brought in through /api/v1/ingest.

    Unlike save_index, this isn't a hash-invalidated cache of a fixed
    source folder: it's the running record of everything ever ingested, so
    it survives a process restart (see ROADMAP.md Phase 2 for the real
    vector store this is standing in for).
    """
    index_dir = index_dir or INDEX_DIR
    paths = _paths(index_dir)
    os.makedirs(index_dir, exist_ok=True)
    np.save(paths["ingested_emb"], embeddings)
    json.dump(metadata, open(paths["ingested_meta"], "w"), indent=2)


def load_ingested(index_dir=None):
    index_dir = index_dir or INDEX_DIR
    paths = _paths(index_dir)
    if not (os.path.exists(paths["ingested_emb"]) and os.path.exists(paths["ingested_meta"])):
        return None
    metadata = json.load(open(paths["ingested_meta"]))
    if not metadata:
        return None
    embeddings = np.load(paths["ingested_emb"])
    return embeddings, metadata
