from pydantic import BaseModel
from typing import List, Dict, Optional


class SearchRequest(BaseModel):
    query: str
    agent_id: str = "default"
    top_k: Optional[int] = None


class EvidenceMetadata(BaseModel):
    vector_score: float
    keyword_score: float
    id: Optional[str] = None


class Evidence(BaseModel):
    content: str
    source: str
    type: str = "documentation"
    confidence: float
    metadata: EvidenceMetadata


class SearchResponse(BaseModel):
    query: str
    results: List[Evidence]
