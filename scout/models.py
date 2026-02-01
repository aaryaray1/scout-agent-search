from pydantic import BaseModel
from typing import List, Dict

class Evidence(BaseModel):
    content: str
    source: str
    confidence: float
    metadata: Dict = {}

class SearchResult(BaseModel):
    query: str
    results: List[Evidence]
