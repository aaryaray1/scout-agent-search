"""HTML -> structured content extraction.

Delegates boilerplate removal (nav, ads, cookie banners, etc.) to
trafilatura and asks it for markdown output, so the result slots straight
into scout.ingest.chunk_docs: the same chunking path the local markdown
corpus already goes through.
"""
import logging

import trafilatura

logger = logging.getLogger(__name__)

# Tables and multi-line code blocks survive as markdown by default; links
# are kept too so citations in the source page (e.g. "see the related
# docs") carry their real URL into the evidence instead of being flattened
# to plain text. Images are left out: an image URL alone isn't useful
# evidence for a text-based agent, and there's no rendering surface for it.
_EXTRACT_OPTIONS = {
    "output_format": "markdown",
    "include_links": True,
    "include_images": False,
    "favor_precision": True,
}

_EMPTY_METADATA_FIELDS = ("author", "date", "sitename")


def _extract_markdown(html: str, url: str) -> str:
    content = trafilatura.extract(html, url=url, **_EXTRACT_OPTIONS)
    if not content or not content.strip():
        raise ValueError("no extractable content found in HTML")
    return content


def _extract_metadata(html: str, url: str):
    """Return (title, metadata) for the page.

    trafilatura returns None when it can't read any metadata at all, which
    is common for fragments and error pages, so the shape of the metadata
    dict stays the same either way and callers never branch on it.
    """
    meta = trafilatura.extract_metadata(html, default_url=url)
    if meta is None:
        return None, {"url": url, **{f: None for f in _EMPTY_METADATA_FIELDS}}
    return meta.title, {
        "url": url,
        **{f: getattr(meta, f, None) for f in _EMPTY_METADATA_FIELDS},
    }


def extract_content(html: str, url: str = None) -> dict:
    """Extract main content + metadata from raw HTML.

    Returns {"title": str, "content": str (markdown), "metadata": dict}.
    Raises ValueError if trafilatura finds nothing extractable, e.g. the
    page is a login wall, an error page, or its content is JS-rendered
    (Scout does not execute JavaScript; see ROADMAP.md).
    """
    content = _extract_markdown(html, url)
    title, metadata = _extract_metadata(html, url)
    return {
        "title": title or url or "untitled",
        "content": content,
        "metadata": metadata,
    }
