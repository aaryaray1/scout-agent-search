"""HTTP-level tests for the FastAPI app.

Each test gets its own TestClient context, which triggers scout.api's
lifespan handler (building a fresh Retriever) while that test's
isolate_index_dir fixture (tests/conftest.py) is already active, so these
tests never touch the real project's data/index/.
"""
import pytest
from fastapi.testclient import TestClient

from scout.api import app
from scout.models import (
    EVIDENCE_SCHEMA_VERSION,
    MAX_HTML_CHARS,
    MAX_QUERY_CHARS,
    MAX_TOP_K,
)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["service"] == "Scout"


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["corpus_size"] > 0


def test_search_returns_schema_version(client):
    resp = client.post("/api/v1/search", json={"query": "rate limit exceeded"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["schema_version"] == "1.0"
    assert len(body["results"]) > 0


def test_search_rejects_empty_query(client):
    resp = client.post("/api/v1/search", json={"query": "   "})
    assert resp.status_code == 422


def test_ingest_returns_schema_version_and_is_immediately_searchable(client):
    html = (
        "<html><head><title>API Test Page</title></head><body><article>"
        "<p>Gadget teleportation errors are logged under code 5150.</p>"
        "</article></body></html>"
    )
    resp = client.post("/api/v1/ingest", json={
        "html": html,
        "source_url": "https://example.com/api-test-page",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["schema_version"] == "1.0"
    assert body["title"] == "API Test Page"
    assert len(body["chunks"]) > 0

    search_resp = client.post("/api/v1/search", json={
        "query": "gadget teleportation error 5150",
        "top_k": 1,
    })
    results = search_resp.json()["results"]
    assert results[0]["source"] == "https://example.com/api-test-page"


def test_ingest_requires_url_or_html(client):
    resp = client.post("/api/v1/ingest", json={})
    assert resp.status_code == 422


def test_ingest_rejects_unfetchable_url(client):
    resp = client.post("/api/v1/ingest", json={"url": "http://127.0.0.1/"})
    assert resp.status_code == 502


def test_health_reports_the_schema_version(client):
    body = client.get("/health").json()
    assert body["schema_version"] == EVIDENCE_SCHEMA_VERSION


@pytest.mark.parametrize("top_k", [0, -1, MAX_TOP_K + 1])
def test_search_rejects_out_of_range_top_k(client, top_k):
    """top_k=0 used to fall through to the server default and a negative
    top_k sliced from the end of the ranking. Both are now schema errors,
    which also means the bounds show up in the OpenAPI document."""
    resp = client.post("/api/v1/search", json={"query": "rate limit", "top_k": top_k})
    assert resp.status_code == 422


def test_search_rejects_an_oversized_query(client):
    resp = client.post("/api/v1/search", json={"query": "x" * (MAX_QUERY_CHARS + 1)})
    assert resp.status_code == 422


def test_search_accepts_top_k_at_the_ceiling(client):
    resp = client.post("/api/v1/search", json={"query": "rate limit", "top_k": MAX_TOP_K})
    assert resp.status_code == 200


def test_search_echoes_the_normalized_query(client):
    resp = client.post("/api/v1/search", json={"query": "  rate limit exceeded  "})
    assert resp.status_code == 200
    assert resp.json()["query"] == "rate limit exceeded"


def test_search_bounds_are_published_in_the_openapi_schema(client):
    """Agents should be able to discover the limits rather than find them
    by getting a 422 in production."""
    schema = client.get("/openapi.json").json()
    top_k = schema["components"]["schemas"]["SearchRequest"]["properties"]["top_k"]
    assert top_k["anyOf"][0]["maximum"] == MAX_TOP_K
    assert top_k["anyOf"][0]["minimum"] == 1


def test_ingest_rejects_html_with_nothing_extractable(client):
    resp = client.post("/api/v1/ingest", json={
        "html": "<html><body></body></html>",
        "source_url": "https://example.com/blank",
    })
    assert resp.status_code == 422


def test_ingest_rejects_an_oversized_html_payload(client):
    resp = client.post("/api/v1/ingest", json={
        "html": "x" * (MAX_HTML_CHARS + 1),
        "source_url": "https://example.com/huge",
    })
    assert resp.status_code == 422


def test_ingest_rejects_html_without_a_source_url(client):
    resp = client.post("/api/v1/ingest", json={"html": "<html><body><p>hi</p></body></html>"})
    assert resp.status_code == 422
