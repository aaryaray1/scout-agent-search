from fastapi import FastAPI, HTTPException
from .search import Retriever
from .models import SearchRequest, SearchResponse

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
