"""The HTML to structured-chunk pipeline: fetch, extract, chunk.

Kept separate from its three stages so each stays independently testable.
See docs/design/ingestion.md.
"""
from .fetch import fetch_html
from .webextract import extract_content
from .ingest import doc_from_web_content, chunk_docs


def ingest_url(url):
    """Fetch `url` and return (title, metadata, chunks)."""
    html = fetch_html(url)
    return ingest_html(html, source_url=url)


def ingest_html(html, source_url):
    """Extract and chunk already-fetched HTML, skipping the network."""
    extracted = extract_content(html, url=source_url)
    doc = doc_from_web_content(extracted, source_url)
    chunks = chunk_docs([doc])
    return extracted["title"], extracted["metadata"], chunks
