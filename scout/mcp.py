"""MCP server exposing Scout's search and ingest as agent tools.

Runs Scout in-process by default, so an agent gets web ingestion and search
with no server to deploy; `--url` points it at a shared Scout instead. See
docs/design/mcp.md.
"""
import argparse
import logging
import sys
from typing import Optional

from pydantic import BaseModel, Field

from .client import ScoutAPIError, ScoutClient
from .config import load_config
from .fetch import FetchError
from .logsetup import configure_logging
from .models import EVIDENCE_SCHEMA_VERSION
from .search import Retriever
from .web import ingest_url

logger = logging.getLogger(__name__)

# What an agent reads to decide whether to use this server at all, so it
# leads with what Scout is for rather than with what it is.
INSTRUCTIONS = """\
Scout turns web pages into searchable structured evidence.

Use `ingest` on any URL you would otherwise fetch and read in full. It
extracts the page, indexes it, and returns a short summary rather than the
page body. Then use `search` to pull back only the passages that answer your
question. Ingesting five pages and searching them costs a fraction of
reading five pages, and everything ingested stays searchable afterwards.
"""

# Enough of the first chunk to confirm the page is the one meant, without
# paying for the body.
PREVIEW_WORDS = 40


class Passage(BaseModel):
    """One piece of evidence, with the scoring breakdown dropped: an agent
    acts on the text and the source, and pays tokens for the rest."""

    content: str
    source: str
    confidence: float


class SearchResult(BaseModel):
    query: str
    results: list[Passage]


class IngestedPage(BaseModel):
    """What ingest returns instead of the page.

    Returning the body would cost the tokens the agent came to save; it is
    indexed, and `search` is how the part that matters comes back.
    """

    status: str = Field(description="ok, fetch_error, extract_error, or error.")
    url: str
    title: Optional[str] = None
    chunks_indexed: int = 0
    words: int = 0
    metadata: dict = Field(default_factory=dict)
    preview: str = ""
    error: Optional[str] = None


class IngestResult(BaseModel):
    results: list[IngestedPage]


def _preview(chunks):
    if not chunks:
        return ""
    words = chunks[0]["content"].split()[:PREVIEW_WORDS]
    return " ".join(words) + ("..." if len(words) == PREVIEW_WORDS else "")


def _summary(page_url, title, metadata, chunks):
    return IngestedPage(
        status="ok",
        url=page_url,
        title=title,
        chunks_indexed=len(chunks),
        words=sum(len(c["content"].split()) for c in chunks),
        metadata=metadata or {},
        preview=_preview(chunks),
    )


class LocalScout:
    """Scout as a library, matching the two ScoutClient methods this server
    uses so the tools don't care which backend is behind them."""

    def __init__(self):
        self.retriever = Retriever()

    def search(self, query, top_k=None):
        return self.retriever.search(query, top_k=top_k)

    def ingest_url(self, url):
        title, metadata, chunks = ingest_url(url)
        self.retriever.add_chunks(chunks)
        return _summary(url, title, metadata, chunks)


class RemoteScout:
    """A Scout server over HTTP, reshaped to the same summary LocalScout
    returns so both tools have one response shape."""

    def __init__(self, base_url, api_key=None):
        self.client = ScoutClient(base_url, api_key=api_key)

    def search(self, query, top_k=None):
        return self.client.search(query, top_k=top_k)

    def ingest_url(self, url):
        page = self.client.ingest_url(url)
        return _summary(page["url"], page["title"], page["metadata"], page["chunks"])


def _ingest_one(backend, url):
    """Ingest one URL, turning an expected failure into a status rather than
    failing every other URL in the call."""
    try:
        return backend.ingest_url(url)
    except FetchError as e:
        return IngestedPage(status="fetch_error", url=url, error=str(e))
    except ScoutAPIError as e:
        return IngestedPage(status="error", url=url, error=str(e.detail))
    except ValueError as e:
        # Nothing extractable: a login wall, an error page, or JS-rendered.
        return IngestedPage(status="extract_error", url=url, error=str(e))


def build_server(backend):
    """Wire the two tools onto an MCPServer over `backend`.

    Tools are sync: the SDK runs them in a worker thread, which is the
    concurrency the Retriever is already built for.
    """
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(
        name="scout",
        title="Scout",
        version=EVIDENCE_SCHEMA_VERSION,
        instructions=INSTRUCTIONS,
    )

    @server.tool()
    def search(query: str, top_k: int = 5) -> SearchResult:
        """Search everything ingested so far and return only the passages
        that match.

        Use this after `ingest`, or on its own to re-search pages ingested
        earlier in this session or a previous one. Ranking is hybrid semantic
        plus keyword, so it finds passages that mean the same thing as the
        query as well as ones sharing its words.

        Args:
            query: what you want to know, in natural language.
            top_k: how many passages to return. Keep it small.
        """
        # Passage is what drops the scoring breakdown evidence carries:
        # validating into it keeps the response to the declared shape.
        return SearchResult(query=query, results=backend.search(query, top_k=top_k))

    @server.tool()
    def ingest(urls: list[str]) -> IngestResult:
        """Fetch web pages, extract their real content, and index them for
        searching. Returns a short summary per page, not the page text.

        Use this instead of fetching a URL and reading it in full.
        Boilerplate is stripped, the content is indexed, and `search` then
        returns only the passages you need. Re-ingesting a URL replaces its
        previous version rather than duplicating it.

        Args:
            urls: the pages to ingest. Pass several at once when you have
                several; each one succeeds or fails on its own.
        """
        return IngestResult(results=[_ingest_one(backend, url) for url in urls])

    return server


def build_backend(url=None, api_key=None):
    return RemoteScout(url, api_key) if url else LocalScout()


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="scout-mcp",
        description="Run Scout as an MCP server over stdio.",
    )
    parser.add_argument(
        "--url",
        help="Base URL of a running Scout server. Omit to run Scout in this "
             "process, which needs no server but holds the index here.",
    )
    parser.add_argument("--api-key", help="X-API-Key for --url, if it needs one.")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    # Logging goes to stderr; stdout is the MCP transport, and anything
    # written there that is not a protocol message breaks the session.
    configure_logging(load_config()["log_level"])
    server = build_server(build_backend(args.url, args.api_key))
    logger.info("scout MCP server ready (%s)", args.url or "in-process")
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
