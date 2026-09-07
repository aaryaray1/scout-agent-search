"""Request and response schemas.

These are the contract agents code against, so the bounds below are
deliberately declared on the models rather than checked inside the
endpoints: FastAPI publishes them in the OpenAPI schema, which means a
caller can discover the limits instead of discovering them by getting a
422 in production.

The caps also bound what a single request can cost Scout. They are not a
substitute for auth and rate limiting, which remain ROADMAP.md Phase 3.
"""
from typing import Annotated, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, StringConstraints, model_validator

from .config import load_config

# Bump this on any breaking change to Evidence/SearchResponse/IngestResponse
# shape, so agents consuming these responses have something concrete to
# check compatibility against instead of guessing from field presence.
EVIDENCE_SCHEMA_VERSION = "1.0"

_config = load_config()
MAX_TOP_K = _config["max_top_k"]
MAX_QUERY_CHARS = _config["max_query_chars"]
MAX_HTML_CHARS = _config["max_html_bytes"]
MAX_BATCH_QUERIES = _config["max_batch_queries"]
MAX_BATCH_INGEST = _config["max_batch_ingest"]
MAX_URL_CHARS = 2048

# Whitespace-only queries collapse to "" and fail min_length, so an empty
# query is rejected by the schema rather than by a hand-rolled check in the
# endpoint.
QueryStr = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUERY_CHARS),
]
UrlStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_URL_CHARS)]


class SearchRequest(BaseModel):
    query: QueryStr = Field(description="Natural-language query to retrieve evidence for.")
    agent_id: str = Field(default="default", max_length=128)
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=MAX_TOP_K,
        description="How many pieces of evidence to return. Defaults to the server's configured top_k.",
    )


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
    url: Optional[UrlStr] = None
    html: Optional[str] = Field(default=None, max_length=MAX_HTML_CHARS)
    source_url: Optional[UrlStr] = None

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


class BatchSearchRequest(BaseModel):
    """Several queries answered against one consistent view of the corpus.

    Batch length is capped for the same reason top_k is: it multiplies
    what one request costs.
    """
    queries: List[QueryStr] = Field(
        min_length=1,
        max_length=MAX_BATCH_QUERIES,
        description="Queries to answer. Results come back in this order.",
    )
    agent_id: str = Field(default="default", max_length=128)
    top_k: Optional[int] = Field(default=None, ge=1, le=MAX_TOP_K)


class BatchSearchResult(BaseModel):
    query: str
    results: List[Evidence]


class BatchSearchResponse(BaseModel):
    schema_version: str = EVIDENCE_SCHEMA_VERSION
    results: List[BatchSearchResult]


class BatchIngestRequest(BaseModel):
    items: List[IngestRequest] = Field(
        min_length=1,
        max_length=MAX_BATCH_INGEST,
        description="Pages to ingest. Each item takes the same shape as /ingest.",
    )


class BatchIngestResult(BaseModel):
    """One item's outcome. A batch never fails as a whole for one bad page.

    `status` carries the failure category that the single-page endpoint
    expresses as an HTTP status (502 vs 422), because per-item outcomes
    have nowhere else to put it: "fetch_error" means Scout couldn't get
    the page, "extract_error" means it got one with nothing in it.
    """
    status: Literal["ok", "fetch_error", "extract_error"] = Field(
        description="Whether the page was ingested, unreachable, or empty."
    )
    url: str
    page: Optional[IngestResponse] = None
    error: Optional[str] = None


class BatchIngestResponse(BaseModel):
    schema_version: str = EVIDENCE_SCHEMA_VERSION
    results: List[BatchIngestResult]
