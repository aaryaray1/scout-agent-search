import os
import json
import hashlib
import numpy as np

INDEX_DIR = "data/index"
EMB_PATH = os.path.join(INDEX_DIR, "embeddings.npy")
META_PATH = os.path.join(INDEX_DIR, "meta.json")
HASH_PATH = os.path.join(INDEX_DIR, "corpus_hash.txt")


def ensure_index_dir():
    os.makedirs(INDEX_DIR, exist_ok=True)


def compute_corpus_hash(docs):
    h = hashlib.sha256()
    for d in docs:
        h.update(d["content"].encode("utf-8"))
    return h.hexdigest()


def save_index(embeddings, metadata, corpus_hash):
    ensure_index_dir()
    np.save(EMB_PATH, embeddings)
    json.dump(metadata, open(META_PATH, "w"), indent=2)
    open(HASH_PATH, "w").write(corpus_hash)


def load_index():
    if not (os.path.exists(EMB_PATH) and os.path.exists(META_PATH)):
        return None
    embeddings = np.load(EMB_PATH)
    metadata = json.load(open(META_PATH))
    corpus_hash = open(HASH_PATH).read()
    return embeddings, metadata, corpus_hash
