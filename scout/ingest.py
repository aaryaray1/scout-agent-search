from pathlib import Path
import uuid

def load_markdown_docs(path="data/docs"):

    docs = []
    for file in Path(path).glob("*.md"):
        text = file.read_text(encoding="utf-8")
        docs.append({
            "source": file.name,
            "content": text,
            # Default type
            "type": "documentation"
        })
    return docs


def chunk_text(text, chunk_size=400, overlap=50):

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap

    return chunks


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


_CORE_FIELDS = {"id", "content", "source", "type"}


def chunk_docs(docs):

    chunks = []

    for doc in docs:
        # Generate unique ID
        if "id" not in doc:
            doc["id"] = str(uuid.uuid4())

        # Ensure 'type' exists
        doc_type = doc.get("type", "documentation")

        # Carry through any doc-level extras (title, url, page_metadata, ...)
        # onto every chunk so callers (e.g. the ingest API) don't lose them.
        extra_fields = {k: v for k, v in doc.items() if k not in _CORE_FIELDS}

        # Chunk content
        for order, chunk in enumerate(chunk_text(doc["content"])):
            chunks.append({
                "id": doc["id"],
                "content": chunk,
                "source": doc["source"],
                "type": doc_type,
                "order": order,
                **extra_fields,
            })

    return chunks
