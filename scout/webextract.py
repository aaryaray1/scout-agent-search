"""HTML -> structured content extraction.

Delegates boilerplate removal (nav, ads, cookie banners, etc.) to
trafilatura and asks it for markdown output, so the result slots straight
into scout.ingest.chunk_docs: the same chunking path the local markdown
corpus already goes through.
"""
import trafilatura


def extract_content(html: str, url: str = None) -> dict:
    """Extract main content + metadata from raw HTML.

    Returns {"title": str, "content": str (markdown), "metadata": dict}.
    Raises ValueError if trafilatura finds nothing extractable, e.g. the
    page is a login wall, an error page, or its content is JS-rendered
    (Scout does not execute JavaScript; see ROADMAP.md).
    """
    content = trafilatura.extract(
        html,
        url=url,
        output_format="markdown",
        include_links=False,
        include_images=False,
        favor_precision=True,
    )
    if not content or not content.strip():
        raise ValueError("no extractable content found in HTML")

    meta = trafilatura.extract_metadata(html, default_url=url)
    title = (meta.title if meta else None) or url or "untitled"

    return {
        "title": title,
        "content": content,
        "metadata": {
            "url": url,
            "author": meta.author if meta else None,
            "date": meta.date if meta else None,
            "sitename": meta.sitename if meta else None,
        },
    }
