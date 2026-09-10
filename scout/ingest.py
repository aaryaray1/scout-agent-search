"""Loading documents and splitting them into indexable chunks.

Local markdown and fetched web pages converge on the same doc shape here,
so chunk_docs() is the single chunking path. See docs/design/ingestion.md.
"""
import logging
import uuid
from pathlib import Path

from .config import load_config

logger = logging.getLogger(__name__)

# Set by chunk_docs(); anything else on a doc is carried onto each chunk.
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
    """Split `text` into overlapping fixed-size word windows."""
    config = load_config()
    chunk_size = config["chunk_size"] if chunk_size is None else chunk_size
    overlap = config["chunk_overlap"] if overlap is None else overlap
    # A non-positive stride re-emits the same words forever.
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
    """Shape extractor output like load_markdown_docs() output, so it flows
    through chunk_docs() unchanged."""
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

    Never writes back to `doc`: callers reuse their own dicts afterwards.
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
