"""FastAPI surface: /api/v1 search and ingest, single and batch.

Both endpoints are deliberately `def`, not `async def`. Retrieval is
CPU-bound (embedding a query, scoring the corpus) and ingest does a
blocking outbound fetch; declaring them sync lets Starlette run them in
its threadpool instead of stalling the event loop. Retriever is built for
exactly that concurrency -- see its docstring on the two locks.

Every /api/v1 route sits behind require_api_key (scout/auth.py). `/` and
/health deliberately don't: a liveness probe shouldn't need a credential,
and neither one reveals anything a caller couldn't learn from the port
answering at all.
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

# Documented on the ingest routes so an agent can read the failure mode
# out of /openapi.json instead of discovering it under load.
RATE_LIMITED = {429: {"description": "Per-caller ingest rate limit exceeded."}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the per-process state on startup rather than at import time.

    Constructing it eagerly at module scope (the old approach) meant
    merely importing this module -- which any test file does -- loaded
    the embedding model and touched disk. Building it here instead means
    it only happens when the app actually starts, which is also the
    point at which test isolation (see tests/conftest.py) can redirect
    where it reads/writes.
    """
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

    Called explicitly rather than declared as a dependency because the
    cost of a batch is the number of items in its body, which a
    dependency can't see. Charging a batch its full length is the point:
    otherwise /ingest/batch would be a way to make N outbound fetches for
    the price of one.
    """
    retry_after = request.app.state.limiter.check(caller, cost)
    if retry_after is None:
        return
    logger.info("rate limited %s (cost %d, retry in %ss)", caller, cost, retry_after)
    raise HTTPException(
        status_code=429,
        detail=f"ingest rate limit exceeded; retry in {retry_after}s",
        # Retry-After is defined in whole seconds; round up so a client
        # obeying it exactly doesn't come back a fraction too early.
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
    """Liveness plus enough state to tell an empty deployment from a
    populated one, and an open one from a guarded one.

    `auth` is reported because "did my API keys actually reach the
    container" is otherwise only answerable by getting rejected.
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

    Empty and oversized queries are rejected by SearchRequest itself, so
    there is nothing left to validate here.
    """
    results = retriever.search(q.query, top_k=q.top_k)
    return {"query": q.query, "results": results}


@app.post("/api/v1/search/batch", response_model=BatchSearchResponse)
def search_batch_endpoint(
    q: BatchSearchRequest,
    retriever: Retriever = Depends(get_retriever),
    caller: str = Depends(require_api_key),
):
    """Answer several queries in one round trip.

    For an agent fanning out over sub-questions this is one HTTP call
    instead of N, and every query is scored against the same corpus
    snapshot, so results within a batch stay mutually consistent even if
    an ingest lands while it runs. Not rate limited: like /search it's
    local CPU work, already bounded by the batch-length cap.
    """
    batches = retriever.search_many(q.queries, top_k=q.top_k)
    return {
        "results": [
            {"query": query, "results": results}
            for query, results in zip(q.queries, batches)
        ]
    }


# -- ingest ------------------------------------------------------------------

def _extract_page(req: IngestRequest):
    """Run the right ingest path for the request and return
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
    """Convert a web page (fetched by URL, or handed to us as raw HTML)
    straight into structured JSON: the conversion step agents otherwise
    have to do themselves. Also indexes the result so it's immediately
    searchable via /api/v1/search, and persists it so it survives a
    restart (see scout/index.py; still not the real Phase 2 vector
    store).
    """
    charge_ingest(request, caller)
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
    return _page_payload(source_url, title, metadata, chunks)


def _ingest_one(req: IngestRequest):
    """Ingest one batch item. Returns (result, chunks), and never raises
    for a per-item failure.

    The two exception types the single-page endpoint turns into 502 and
    422 become status strings here, since one bad page shouldn't decide
    the status of the whole batch.
    """
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

    Two things this does that a client looping over /ingest can't:

    - A failed page fails alone. Results come back in request order, each
      carrying its own status, so an agent crawling ten links gets the
      eight that worked instead of one error.
    - Every chunk goes in through one add_chunks() call, so the ingested
      store and the BM25 index are rewritten once for the batch instead
      of once per page (ROADMAP.md Phase 2 on why that rewrite is the
      expensive part).

    Costs the rate limiter one unit per item, since that's how many
    outbound fetches it can trigger.
    """
    charge_ingest(request, caller, cost=len(req.items))

    results = []
    # Keyed by source so a URL repeated inside one batch is indexed once,
    # matching add_chunks()'s own "one version per source" rule. Every
    # item still gets its own entry in the response.
    chunks_by_source = {}
    for item in req.items:
        result, chunks = _ingest_one(item)
        results.append(result)
        if chunks:
            chunks_by_source[chunks[0]["source"]] = chunks

    retriever.add_chunks([c for chunks in chunks_by_source.values() for c in chunks])
    return {"results": results}
