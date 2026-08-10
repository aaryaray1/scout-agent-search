"""Orchestrates the HTML -> structured chunk pipeline: fetch, extract, chunk.

This is what the /api/v1/ingest endpoint calls. Kept separate from
scout.fetch / scout.webextract / scout.ingest so each stage (network,
extraction, chunking) stays independently testable.
"""
from .fetch import fetch_html
from .webextract import extract_content
from .ingest import doc_from_web_content, chunk_docs


def ingest_url(url):
    """Fetch `url`, extract its main content, and return it chunked.

    Returns (title, metadata, chunks). chunks match the same shape produced
    for the local markdown corpus, so they're ready for
    Retriever.add_chunks() or direct serialization.
    """
    html = fetch_html(url)
    return ingest_html(html, source_url=url)


def ingest_html(html, source_url):
    """Extract + chunk already-fetched HTML (skips the network fetch)."""
    extracted = extract_content(html, url=source_url)
    doc = doc_from_web_content(extracted, source_url)
    chunks = chunk_docs([doc])
    return extracted["title"], extracted["metadata"], chunks
