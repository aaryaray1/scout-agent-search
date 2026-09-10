"""HTML to markdown + metadata, via trafilatura.

Markdown output so the result slots into the same chunking path the local
markdown corpus uses. See docs/design/ingestion.md.
"""
import logging

import trafilatura

logger = logging.getLogger(__name__)

# Links are kept so citations carry their real URL into the evidence;
# images are not, being useless to a text-only agent.
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
    """Return (title, metadata), keeping the metadata shape constant.

    trafilatura returns None for fragments and error pages, so callers
    never have to branch on it.
    """
    meta = trafilatura.extract_metadata(html, default_url=url)
    if meta is None:
        return None, {"url": url, **{f: None for f in _EMPTY_METADATA_FIELDS}}
    return meta.title, {
        "url": url,
        **{f: getattr(meta, f, None) for f in _EMPTY_METADATA_FIELDS},
    }


def extract_content(html: str, url: str = None) -> dict:
    """Extract main content + metadata as {"title", "content", "metadata"}.

    Raises ValueError when there is nothing extractable: a login wall, an
    error page, or JS-rendered content, which Scout does not execute.
    """
    content = _extract_markdown(html, url)
    title, metadata = _extract_metadata(html, url)
    return {
        "title": title or url or "untitled",
        "content": content,
        "metadata": metadata,
    }
