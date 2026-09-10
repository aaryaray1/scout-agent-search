"""FastAPI surface: /api/v1 search and ingest, single and batch.

Endpoints are `def`, not `async def`, so Starlette runs them in its
threadpool rather than stalling the event loop. See docs/design/api.md.
"""
import logging
import math
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request

from .auth import ApiKeyAuth, require_api_key
from .config import load_config
from .fetch import FetchError
from .logsetup import configure_logging
from .models import (
    EVIDENCE_SCHEMA_VERSION,
    BatchIngestRequest,
    BatchIngestResponse,
    BatchSearchRequest,
    BatchSearchResponse,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
)
from .ratelimit import TokenBucketLimiter
from .search import Retriever
from .web import ingest_html, ingest_url

logger = logging.getLogger(__name__)

# Documented on the ingest routes so an agent reads the failure mode out of
# /openapi.json instead of discovering it under load.
RATE_LIMITED = {429: {"description": "Per-caller ingest rate limit exceeded."}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build per-process state on startup rather than at import time, so
    importing this module doesn't load the embedding model or touch disk."""
    config = load_config()
    configure_logging(config["log_level"])
    app.state.auth = ApiKeyAuth(config["api_keys"])
    app.state.auth.log_status()
    app.state.limiter = TokenBucketLimiter(
        config["ingest_rate_limit"], config["ingest_rate_window"]
    )
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


def charge_ingest(request: Request, caller: str, cost: int = 1) -> None:
    """Spend `cost` against the caller's ingest budget, or raise 429.

    Called explicitly rather than as a dependency because a batch costs its
    own length, which a dependency cannot see.
    """
    retry_after = request.app.state.limiter.check(caller, cost)
    if retry_after is None:
        return
    logger.info("rate limited %s (cost %d, retry in %ss)", caller, cost, retry_after)
    raise HTTPException(
        status_code=429,
        detail=f"ingest rate limit exceeded; retry in {retry_after}s",
        # Retry-After is whole seconds; round up so a client obeying it
        # exactly doesn't come back a fraction too early.
        headers={"Retry-After": str(math.ceil(retry_after))},
    )


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
def health(request: Request, retriever: Retriever = Depends(get_retriever)):
    """Liveness, corpus size, and whether auth is on.

    Open, like `/`: a probe shouldn't need a credential, and neither reveals
    anything the port answering at all doesn't.
    """
    return {
        "status": "ok",
        "corpus_size": len(retriever.corpus),
        "auth": "enabled" if request.app.state.auth.enabled else "disabled",
        "schema_version": EVIDENCE_SCHEMA_VERSION,
    }


# -- search ------------------------------------------------------------------

@app.post("/api/v1/search", response_model=SearchResponse)
def search_endpoint(
    q: SearchRequest,
    retriever: Retriever = Depends(get_retriever),
    caller: str = Depends(require_api_key),
):
    """Retrieve the best-matching evidence for a query.

    Empty and oversized queries are rejected by the request model itself.
    """
    results = retriever.search(q.query, top_k=q.top_k)
    return {"query": q.query, "results": results}


@app.post("/api/v1/search/batch", response_model=BatchSearchResponse)
def search_batch_endpoint(
    q: BatchSearchRequest,
    retriever: Retriever = Depends(get_retriever),
    caller: str = Depends(require_api_key),
):
    """Answer several queries in one round trip, against one corpus
    snapshot, so results stay mutually consistent if an ingest lands
    mid-batch. Not rate limited: local CPU work, already batch-capped."""
    batches = retriever.search_many(q.queries, top_k=q.top_k)
    return {
        "results": [
            {"query": query, "results": results}
            for query, results in zip(q.queries, batches)
        ]
    }


# -- ingest ------------------------------------------------------------------

def _extract_page(req: IngestRequest):
    """Run the right ingest path and return
    (source_url, title, metadata, chunks)."""
    if req.url:
        title, metadata, chunks = ingest_url(req.url)
        return req.url, title, metadata, chunks
    title, metadata, chunks = ingest_html(req.html, req.source_url)
    return req.source_url, title, metadata, chunks


def _page_payload(source_url, title, metadata, chunks):
    """Shape one extracted page into the ingest response body."""
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


@app.post("/api/v1/ingest", response_model=IngestResponse, responses=RATE_LIMITED)
def ingest_endpoint(
    req: IngestRequest,
    request: Request,
    retriever: Retriever = Depends(get_retriever),
    caller: str = Depends(require_api_key),
):
    """Turn a web page into structured JSON, index it, and persist it.

    Takes a URL for Scout to fetch, or raw HTML the caller already has. The
    result is searchable immediately, with no rebuild step.
    """
    charge_ingest(request, caller)
    try:
        source_url, title, metadata, chunks = _extract_page(req)
    except FetchError as e:
        # Upstream wouldn't give us the page: a bad gateway, not a bad request.
        logger.warning("ingest fetch failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e)) from e
    except ValueError as e:
        # Nothing extractable: the caller's input is the problem.
        logger.info("ingest extraction rejected: %s", e)
        raise HTTPException(status_code=422, detail=str(e)) from e

    retriever.add_chunks(chunks)
    return _page_payload(source_url, title, metadata, chunks)


def _ingest_one(req: IngestRequest):
    """Ingest one batch item as (result, chunks), never raising for a
    per-item failure: the 502/422 split becomes a status string instead."""
    requested_url = req.url or req.source_url
    try:
        source_url, title, metadata, chunks = _extract_page(req)
    except FetchError as e:
        logger.warning("batch ingest fetch failed for %s: %s", requested_url, e)
        return {"status": "fetch_error", "url": requested_url, "error": str(e)}, []
    except ValueError as e:
        logger.info("batch ingest extraction rejected for %s: %s", requested_url, e)
        return {"status": "extract_error", "url": requested_url, "error": str(e)}, []
    page = _page_payload(source_url, title, metadata, chunks)
    return {"status": "ok", "url": source_url, "page": page}, chunks


@app.post(
    "/api/v1/ingest/batch", response_model=BatchIngestResponse, responses=RATE_LIMITED
)
def ingest_batch_endpoint(
    req: BatchIngestRequest,
    request: Request,
    retriever: Retriever = Depends(get_retriever),
    caller: str = Depends(require_api_key),
):
    """Ingest several pages in one call, indexing them as a single write.

    A failed page fails alone, with its own status, so an agent crawling ten
    links gets the eight that worked. Costs the rate limiter one unit per
    item, since that is how many outbound fetches it can trigger.
    """
    charge_ingest(request, caller, cost=len(req.items))

    results = []
    # Keyed by source so a URL repeated inside one batch is indexed once,
    # matching add_chunks()'s own one-version-per-source rule.
    chunks_by_source = {}
    for item in req.items:
        result, chunks = _ingest_one(item)
        results.append(result)
        if chunks:
            chunks_by_source[chunks[0]["source"]] = chunks

    retriever.add_chunks([c for chunks in chunks_by_source.values() for c in chunks])
    return {"results": results}
