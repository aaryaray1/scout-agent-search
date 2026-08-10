from fastapi import FastAPI, HTTPException
from .search import Retriever
from .models import SearchRequest, SearchResponse, IngestRequest, IngestResponse
from .fetch import FetchError
from .web import ingest_url, ingest_html

app = FastAPI(title="Scout Agent Search")
retriever = Retriever()


@app.get("/")
def root():
    return {
        "service": "Scout",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health")
def health():
    return {"status": "ok", "corpus_size": len(retriever.corpus)}


@app.post("/api/v1/search", response_model=SearchResponse)
def search_endpoint(q: SearchRequest):
    if not q.query or not q.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")
    results = retriever.search(q.query, top_k=q.top_k)
    return {"query": q.query, "results": results}


@app.post("/api/v1/ingest", response_model=IngestResponse)
def ingest_endpoint(req: IngestRequest):
    """Convert a web page (fetched by URL, or handed to us as raw HTML)
    straight into structured JSON: the conversion step agents otherwise
    have to do themselves. Also indexes the result so it's immediately
    searchable via /api/v1/search (in-memory only; see ROADMAP.md Phase 2
    for durable storage of ingested pages).
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
