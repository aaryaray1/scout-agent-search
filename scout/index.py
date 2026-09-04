"""On-disk persistence for both corpus layers.

Two stores live side by side in the same directory:

- the docs_path cache (save_index/load_index): a rebuildable cache of the
  markdown corpus, keyed by a content hash. Losing it costs a re-embed.
- the ingested store (save_ingested/load_ingested): the running record of
  every page brought in through /api/v1/ingest. This is the only copy of
  that data, which is why the two stores handle corruption differently
  (see _read_json).

Every write goes through _atomic_write: content lands in a temp file in the
same directory and is then os.replace()d over the target, so a crash or a
restart mid-write leaves the previous good file in place instead of a
truncated one that would poison the next startup.
"""
import json
import logging
import os
import hashlib
import tempfile

import numpy as np

from .config import load_config

logger = logging.getLogger(__name__)

# Module-level so deployments can point it at a mounted volume via
# SCOUT_INDEX_DIR, and so tests can redirect it (see tests/conftest.py).
INDEX_DIR = load_config()["index_dir"]


class IndexCorruptError(Exception):
    """Raised when a store exists on disk but can't be read back."""


def _paths(index_dir):
    return {
        "emb": os.path.join(index_dir, "embeddings.npy"),
        "meta": os.path.join(index_dir, "meta.json"),
        "hash": os.path.join(index_dir, "corpus_hash.txt"),
        "ingested_emb": os.path.join(index_dir, "ingested_embeddings.npy"),
        "ingested_meta": os.path.join(index_dir, "ingested_meta.json"),
    }


def _resolve(index_dir):
    """Fall back to the module global so monkeypatching INDEX_DIR keeps
    working, rather than binding the default at import time."""
    return _paths(index_dir or INDEX_DIR)


def _atomic_write(path, write_fn, binary=False):
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    mode = "wb" if binary else "w"
    encoding = None if binary else "utf-8"
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, mode, encoding=encoding) as f:
            write_fn(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        # Leaving a stray .tmp behind would be harmless but untidy, and on a
        # mounted volume it accumulates.
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _write_npy(path, array):
    # np.save() appends ".npy" when handed a path but not when handed a file
    # object, which is what keeps the temp file's real name intact.
    _atomic_write(path, lambda f: np.save(f, array), binary=True)


def _write_json(path, obj):
    _atomic_write(path, lambda f: json.dump(obj, f, indent=2))


def _read_json(path, recoverable):
    """Read a JSON store, treating corruption differently per store.

    `recoverable` says whether the caller can rebuild what's in this file.
    The docs cache can (worst case a re-embed), so corruption there
    degrades to a cache miss. The ingested store can't -- it's the only
    copy of every page ever ingested -- so corruption there raises instead
    of quietly starting up empty and letting the next ingest overwrite
    still-recoverable data.
    """
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        if not recoverable:
            raise IndexCorruptError(
                f"could not read '{path}': {e}. Move or delete it to start "
                "with an empty ingested store."
            ) from e
        logger.warning("ignoring unreadable index cache '%s': %s", path, e)
        return None


def compute_corpus_hash(docs):
    """Fingerprint the corpus so a cached index can be invalidated.

    Covers source as well as content: two documents swapping filenames
    leaves the concatenated content identical, and hashing content alone
    would serve a cached index that attributes every chunk to the wrong
    source.
    """
    h = hashlib.sha256()
    for d in docs:
        h.update(d.get("source", "").encode("utf-8"))
        h.update(b"\0")
        h.update(d["content"].encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def save_index(embeddings, metadata, corpus_hash, index_dir=None):
    """Persist the docs_path corpus: a cache keyed by a content hash, so
    scout-ingest can skip rebuilding embeddings when nothing changed."""
    paths = _resolve(index_dir)
    _write_npy(paths["emb"], embeddings)
    _write_json(paths["meta"], metadata)
    _atomic_write(paths["hash"], lambda f: f.write(corpus_hash))


def load_index(index_dir=None):
    """Return (embeddings, metadata, corpus_hash), or None on a cache miss.

    An unreadable or half-written cache counts as a miss: the caller just
    re-embeds.
    """
    paths = _resolve(index_dir)
    required = (paths["emb"], paths["meta"], paths["hash"])
    if not all(os.path.exists(p) for p in required):
        return None

    metadata = _read_json(paths["meta"], recoverable=True)
    if metadata is None:
        return None
    try:
        embeddings = np.load(paths["emb"])
        with open(paths["hash"], encoding="utf-8") as f:
            corpus_hash = f.read()
    except (ValueError, OSError, UnicodeDecodeError) as e:
        logger.warning("ignoring unreadable index cache in '%s': %s", index_dir or INDEX_DIR, e)
        return None
    return embeddings, metadata, corpus_hash


def save_ingested(embeddings, metadata, index_dir=None):
    """Persist pages brought in through /api/v1/ingest.

    Unlike save_index, this isn't a hash-invalidated cache of a fixed
    source folder: it's the running record of everything ever ingested, so
    it survives a process restart (see ROADMAP.md Phase 2 for the real
    vector store this is standing in for).
    """
    paths = _resolve(index_dir)
    _write_npy(paths["ingested_emb"], embeddings)
    _write_json(paths["ingested_meta"], metadata)


def _load_ingested_embeddings(path, expected_rows):
    """Read the ingested embedding matrix and check it still lines up with
    the metadata it was written beside.

    Embeddings and chunks are parallel arrays; a store where they've
    drifted apart would keep working and silently attribute every score to
    the wrong chunk, which is worse than refusing to load.
    """
    try:
        embeddings = np.load(path)
    except (ValueError, OSError) as e:
        raise IndexCorruptError(f"could not read '{path}': {e}") from e

    if len(embeddings) != expected_rows:
        raise IndexCorruptError(
            f"ingested store is inconsistent: {len(embeddings)} embeddings "
            f"for {expected_rows} chunks in '{path}'"
        )
    return embeddings


def load_ingested(index_dir=None):
    """Return (embeddings, metadata) for the ingested store, or None when
    nothing has been ingested yet.

    Raises IndexCorruptError if the store exists but can't be read: it holds
    the only copy of ingested pages, so failing loudly beats starting empty.
    """
    paths = _resolve(index_dir)
    if not (os.path.exists(paths["ingested_emb"]) and os.path.exists(paths["ingested_meta"])):
        return None

    metadata = _read_json(paths["ingested_meta"], recoverable=False)
    if not metadata:
        return None
    return _load_ingested_embeddings(paths["ingested_emb"], len(metadata)), metadata
