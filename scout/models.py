from pydantic import BaseModel, model_validator
from typing import List, Dict, Optional

# Bump this on any breaking change to Evidence/SearchResponse/IngestResponse
# shape, so agents consuming these responses have something concrete to
# check compatibility against instead of guessing from field presence.
EVIDENCE_SCHEMA_VERSION = "1.0"


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
    schema_version: str = EVIDENCE_SCHEMA_VERSION
    query: str
    results: List[Evidence]


class IngestRequest(BaseModel):
    """Either `url` (Scout fetches it) or `html` + `source_url` (caller
    already has the HTML, e.g. from its own browser session)."""
    url: Optional[str] = None
    html: Optional[str] = None
    source_url: Optional[str] = None

    @model_validator(mode="after")
    def _require_url_or_html(self):
        if not self.url and not (self.html and self.source_url):
            raise ValueError(
                "provide either 'url', or both 'html' and 'source_url'"
            )
        return self


class IngestChunk(BaseModel):
    id: str
    content: str
    order: int


class IngestResponse(BaseModel):
    schema_version: str = EVIDENCE_SCHEMA_VERSION
    url: str
    title: str
    type: str = "web"
    metadata: Dict
    chunks: List[IngestChunk]
