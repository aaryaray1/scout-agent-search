"""Loading documents and splitting them into indexable chunks.

Both the local markdown corpus and pages pulled off the web converge on
the same doc shape here, so chunk_docs() is the single chunking path for
everything Scout indexes.
"""
import logging
import uuid
from pathlib import Path

from .config import load_config

logger = logging.getLogger(__name__)

# Fields chunk_docs() sets itself; everything else on a doc is an extra
# that gets carried onto each chunk.
_CORE_FIELDS = {"id", "content", "source", "type"}

DEFAULT_TYPE = "documentation"


def load_markdown_docs(path="data/docs"):
    """Read every *.md file under `path` into a doc dict."""
    docs = []
    for file in sorted(Path(path).glob("*.md")):
        docs.append({
            "source": file.name,
            "content": file.read_text(encoding="utf-8"),
            "type": DEFAULT_TYPE,
        })
    return docs


def chunk_text(text, chunk_size=None, overlap=None):
    """Split `text` into overlapping fixed-size word windows.

    Chunking by word count ignores document structure and can split
    mid-section; chunking by heading would preserve more meaning per chunk
    (tracked in ROADMAP.md Phase 0).
    """
    config = load_config()
    chunk_size = config["chunk_size"] if chunk_size is None else chunk_size
    overlap = config["chunk_overlap"] if overlap is None else overlap
    # The window advances by (chunk_size - overlap). A non-positive stride
    # would re-emit the same words forever, so refuse it rather than hang.
    stride = chunk_size - overlap
    if stride < 1:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
        )

    words = text.split()
    return [
        " ".join(words[start:start + chunk_size])
        for start in range(0, len(words), stride)
    ]


def doc_from_web_content(extracted, source_url):
    """Build a doc dict from scout.webextract.extract_content()'s output.

    Shaped to match load_markdown_docs()'s output so it flows through
    chunk_docs() unchanged: {"source", "content", "type", ...}.
    """
    return {
        "source": source_url,
        "content": extracted["content"],
        "type": "web",
        "title": extracted.get("title"),
        "url": source_url,
        "page_metadata": extracted.get("metadata", {}),
    }


def _chunks_for_doc(doc):
    """Split one doc into chunk dicts, carrying its extra fields along.

    Never writes back to `doc`: callers hand in their own dicts (the CLI
    reuses them for logging, tests assert on them), and a chunker that
    quietly stamps an id onto its input is a trap for the next caller.
    """
    doc_id = doc.get("id") or str(uuid.uuid4())
    doc_type = doc.get("type", DEFAULT_TYPE)
    extra_fields = {k: v for k, v in doc.items() if k not in _CORE_FIELDS}

    return [
        {
            "id": doc_id,
            "content": chunk,
            "source": doc["source"],
            "type": doc_type,
            "order": order,
            **extra_fields,
        }
        for order, chunk in enumerate(chunk_text(doc["content"]))
    ]


def chunk_docs(docs):
    """Flatten a list of docs into a flat list of indexable chunks."""
    chunks = []
    for doc in docs:
        chunks.extend(_chunks_for_doc(doc))
    return chunks
