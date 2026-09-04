"""FastAPI surface: /api/v1/search and /api/v1/ingest.

Both endpoints are deliberately `def`, not `async def`. Retrieval is
CPU-bound (embedding a query, scoring the corpus) and ingest does a
blocking outbound fetch; declaring them sync lets Starlette run them in
its threadpool instead of stalling the event loop. Retriever is built for
exactly that concurrency -- see its docstring on the two locks.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request

from .config import load_config
from .fetch import FetchError
from .logsetup import configure_logging
from .models import (
    EVIDENCE_SCHEMA_VERSION,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
)
from .search import Retriever
from .web import ingest_html, ingest_url

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the Retriever on startup rather than at import time.

    Constructing it eagerly at module scope (the old approach) meant
    merely importing this module -- which any test file does -- loaded
    the embedding model and touched disk. Building it here instead means
    it only happens when the app actually starts, which is also the
    point at which test isolation (see tests/conftest.py) can redirect
    where it reads/writes.
    """
    config = load_config()
    configure_logging(config["log_level"])
    app.state.retriever = Retriever()
    yield


app = FastAPI(
    title="Scout Agent Search",
    version=EVIDENCE_SCHEMA_VERSION,
    description="Search that returns pre-structured JSON evidence, so agents never parse HTML.",
    lifespan=lifespan,
)


def get_retriever(request: Request) -> Retriever:
    """Hand endpoints the shared Retriever built during startup."""
    return request.app.state.retriever


@app.get("/")
def root():
    return {
        "service": "Scout",
        "status": "running",
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "docs": "/docs",
        "redoc": "/redoc",
    }


@app.get("/health")
def health(retriever: Retriever = Depends(get_retriever)):
    """Liveness plus enough state to tell an empty deployment from a
    populated one."""
    return {
        "status": "ok",
        "corpus_size": len(retriever.corpus),
        "schema_version": EVIDENCE_SCHEMA_VERSION,
    }


@app.post("/api/v1/search", response_model=SearchResponse)
def search_endpoint(q: SearchRequest, retriever: Retriever = Depends(get_retriever)):
    """Retrieve the best-matching evidence for a query.

    Empty and oversized queries are rejected by SearchRequest itself, so
    there is nothing left to validate here.
    """
    results = retriever.search(q.query, top_k=q.top_k)
    return {"query": q.query, "results": results}


def _extract_page(req: IngestRequest):
    """Run the right ingest path for the request and return
    (source_url, title, metadata, chunks)."""
    if req.url:
        title, metadata, chunks = ingest_url(req.url)
        return req.url, title, metadata, chunks
    title, metadata, chunks = ingest_html(req.html, req.source_url)
    return req.source_url, title, metadata, chunks


@app.post("/api/v1/ingest", response_model=IngestResponse)
def ingest_endpoint(req: IngestRequest, retriever: Retriever = Depends(get_retriever)):
    """Convert a web page (fetched by URL, or handed to us as raw HTML)
    straight into structured JSON: the conversion step agents otherwise
    have to do themselves. Also indexes the result so it's immediately
    searchable via /api/v1/search, and persists it so it survives a
    restart (see scout/index.py; still not the real Phase 2 vector
    store).
    """
    try:
        source_url, title, metadata, chunks = _extract_page(req)
    except FetchError as e:
        # Upstream wouldn't give us the page: that's a bad gateway, not a
        # bad request.
        logger.warning("ingest fetch failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e)) from e
    except ValueError as e:
        # Nothing extractable in the HTML: the caller's input is the problem.
        logger.info("ingest extraction rejected: %s", e)
        raise HTTPException(status_code=422, detail=str(e)) from e

    retriever.add_chunks(chunks)

    return {
        "url": source_url,
        "title": title,
        "type": "web",
        "metadata": metadata,
        "chunks": [
            {"id": c["id"], "content": c["content"], "order": c["order"]}
            for c in chunks
        ],
    }
