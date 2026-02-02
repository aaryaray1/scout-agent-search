from fastapi import FastAPI
from pydantic import BaseModel
from .search import Retriever

app = FastAPI(title="Scout Agent Search")
retriever = Retriever()

class Query(BaseModel):
    query: str
    agent_id: str = "default"


@app.get("/")
def root():
    return {
        "service": "Scout",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.post("/api/v1/search")
def search_endpoint(q: Query):
    results = retriever.search(q.query)
    return {"query": q.query, "results": results}
