"""On-disk persistence for both corpus layers, with every write atomic.

The docs cache is rebuildable so corruption degrades to a miss; the ingested
store is the only copy, so corruption raises. See docs/design/storage.md.
"""
import json
import logging
import os
import hashlib
import tempfile

import numpy as np

from .config import load_config

logger = logging.getLogger(__name__)

# Module-level so SCOUT_INDEX_DIR can point it at a volume and tests can
# redirect it (see tests/conftest.py).
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


def resolve_index_dir(index_dir=None):
    """Fall back to the module global rather than binding it at import time,
    so monkeypatching INDEX_DIR redirects scout.store too."""
    return index_dir or INDEX_DIR


def _resolve(index_dir):
    return _paths(resolve_index_dir(index_dir))


def _atomic_write(path, write_fn, binary=False):
    """Write via a temp file plus os.replace, so a crash mid-write leaves the
    previous good file rather than a truncated one."""
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
        # A stray .tmp is harmless but accumulates on a mounted volume.
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def write_npy(path, array):
    # np.save() appends ".npy" to a path but not to a file object, which is
    # what keeps the temp file's real name intact.
    _atomic_write(path, lambda f: np.save(f, array), binary=True)


def read_npy(path):
    """Load a .npy, raising IndexCorruptError rather than numpy's own errors
    so callers have one exception type to catch."""
    try:
        return np.load(path)
    except (ValueError, OSError) as e:
        raise IndexCorruptError(f"could not read '{path}': {e}") from e


def write_json(path, obj, indent=2):
    """Write `obj` as JSON, atomically.

    indent=None writes the compact form, for a machine-only file rewritten
    on a hot path: indentation roughly triples the bytes that get fsynced.
    """
    separators = None if indent else (",", ":")
    _atomic_write(
        path, lambda f: json.dump(obj, f, indent=indent, separators=separators)
    )


def read_json(path, recoverable):
    """Read a JSON store. `recoverable` says whether the caller can rebuild
    it: if not, corruption raises rather than reading as empty."""
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

    Covers source as well as content: two docs swapping filenames leaves the
    concatenated content identical.
    """
    h = hashlib.sha256()
    for d in docs:
        h.update(d.get("source", "").encode("utf-8"))
        h.update(b"\0")
        h.update(d["content"].encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def save_index(embeddings, metadata, corpus_hash, index_dir=None):
    """Persist the docs_path corpus cache, keyed by a content hash."""
    paths = _resolve(index_dir)
    write_npy(paths["emb"], embeddings)
    write_json(paths["meta"], metadata)
    _atomic_write(paths["hash"], lambda f: f.write(corpus_hash))


def load_index(index_dir=None):
    """Return (embeddings, metadata, corpus_hash), or None on a cache miss.

    An unreadable or half-written cache counts as a miss: the caller
    re-embeds.
    """
    paths = _resolve(index_dir)
    required = (paths["emb"], paths["meta"], paths["hash"])
    if not all(os.path.exists(p) for p in required):
        return None

    metadata = read_json(paths["meta"], recoverable=True)
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
    """Write the whole-store ingested format, used by JsonChunkStore."""
    paths = _resolve(index_dir)
    write_npy(paths["ingested_emb"], embeddings)
    write_json(paths["ingested_meta"], metadata)


def _load_ingested_embeddings(path, expected_rows):
    """Read the ingested matrix, refusing it if it no longer lines up with
    its metadata -- a drifted store scores every chunk against the wrong
    vector instead of failing."""
    embeddings = read_npy(path)
    if len(embeddings) != expected_rows:
        raise IndexCorruptError(
            f"ingested store is inconsistent: {len(embeddings)} embeddings "
            f"for {expected_rows} chunks in '{path}'"
        )
    return embeddings


def load_ingested(index_dir=None):
    """Return (embeddings, metadata) for the whole-store format, or None.

    Raises IndexCorruptError if it exists but can't be read: it holds the
    only copy of ingested pages, so failing loudly beats starting empty.
    """
    paths = _resolve(index_dir)
    if not (os.path.exists(paths["ingested_emb"]) and os.path.exists(paths["ingested_meta"])):
        return None

    metadata = read_json(paths["ingested_meta"], recoverable=False)
    if not metadata:
        return None
    return _load_ingested_embeddings(paths["ingested_emb"], len(metadata)), metadata
