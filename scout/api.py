from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from .search import Retriever
from .models import SearchRequest, SearchResponse, IngestRequest, IngestResponse
from .fetch import FetchError
from .web import ingest_url, ingest_html


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
    app.state.retriever = Retriever()
    yield


app = FastAPI(title="Scout Agent Search", lifespan=lifespan)


@app.get("/")
def root():
    return {
        "service": "Scout",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "corpus_size": len(request.app.state.retriever.corpus)}


@app.post("/api/v1/search", response_model=SearchResponse)
def search_endpoint(q: SearchRequest, request: Request):
    if not q.query or not q.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")
    results = request.app.state.retriever.search(q.query, top_k=q.top_k)
    return {"query": q.query, "results": results}


@app.post("/api/v1/ingest", response_model=IngestResponse)
def ingest_endpoint(req: IngestRequest, request: Request):
    """Convert a web page (fetched by URL, or handed to us as raw HTML)
    straight into structured JSON: the conversion step agents otherwise
    have to do themselves. Also indexes the result so it's immediately
    searchable via /api/v1/search, and persists it so it survives a
    restart (see scout/index.py; still not the real Phase 2 vector
    store).
    """
    try:
        if req.url:
            title, metadata, chunks = ingest_url(req.url)
            source_url = req.url
        else:
            title, metadata, chunks = ingest_html(req.html, req.source_url)
            source_url = req.source_url
    except FetchError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    request.app.state.retriever.add_chunks(chunks)

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
