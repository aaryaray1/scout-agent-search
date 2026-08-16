"""HTTP-level tests for the FastAPI app.

Each test gets its own TestClient context, which triggers scout.api's
lifespan handler (building a fresh Retriever) while that test's
isolate_index_dir fixture (tests/conftest.py) is already active, so these
tests never touch the real project's data/index/.
"""
import pytest
from fastapi.testclient import TestClient
from scout.api import app


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
