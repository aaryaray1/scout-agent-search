"""Batch search and batch ingest.

The point of these endpoints is fan-out for agents, so the tests care
about the two properties that make them worth having over a client-side
loop: a consistent corpus snapshot across a batch of queries, and
per-item failure isolation across a batch of pages.
"""
import pytest
from fastapi.testclient import TestClient

from scout.api import app
from scout.models import MAX_BATCH_INGEST, MAX_BATCH_QUERIES


def page_html(title, body):
    return (
        f"<html><head><title>{title}</title></head><body><article>"
        f"<p>{body}</p></article></body></html>"
    )


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# -- batch search --------------------------------------------------------------

def test_batch_search_answers_every_query_in_order(client):
    resp = client.post("/api/v1/search/batch", json={
        "queries": ["rate limit exceeded", "authentication"],
        "top_k": 2,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["schema_version"] == "1.0"
    assert [r["query"] for r in body["results"]] == [
        "rate limit exceeded", "authentication",
    ]
    assert all(len(r["results"]) == 2 for r in body["results"])


def test_batch_search_matches_single_search(client):
    """The batch path has to be an optimization, not a different ranking."""
    single = client.post(
        "/api/v1/search", json={"query": "rate limit exceeded", "top_k": 3}
    ).json()["results"]
    batched = client.post("/api/v1/search/batch", json={
        "queries": ["rate limit exceeded"], "top_k": 3,
    }).json()["results"][0]["results"]

    assert batched == single


def test_batch_search_rejects_an_empty_batch(client):
    resp = client.post("/api/v1/search/batch", json={"queries": []})
    assert resp.status_code == 422


def test_batch_search_rejects_an_oversized_batch(client):
    resp = client.post(
        "/api/v1/search/batch", json={"queries": ["q"] * (MAX_BATCH_QUERIES + 1)}
    )
    assert resp.status_code == 422


def test_batch_search_applies_the_same_query_validation(client):
    """A blank query inside a batch is as invalid as a blank query alone."""
    resp = client.post("/api/v1/search/batch", json={"queries": ["fine", "   "]})
    assert resp.status_code == 422


def test_batch_limits_are_published_in_the_openapi_schema(client):
    schema = client.get("/openapi.json").json()
    queries = schema["components"]["schemas"]["BatchSearchRequest"]["properties"]["queries"]
    assert queries["maxItems"] == MAX_BATCH_QUERIES
    assert queries["minItems"] == 1


# -- batch ingest --------------------------------------------------------------

def test_batch_ingest_structures_every_page(client):
    resp = client.post("/api/v1/ingest/batch", json={"items": [
        {"html": page_html("Alpha", "Alpha covers widget calibration codes."),
         "source_url": "https://example.com/alpha"},
        {"html": page_html("Beta", "Beta covers sprocket alignment failures."),
         "source_url": "https://example.com/beta"},
    ]})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert [r["status"] for r in results] == ["ok", "ok"]
    assert [r["page"]["title"] for r in results] == ["Alpha", "Beta"]


def test_batch_ingested_pages_are_immediately_searchable(client):
    client.post("/api/v1/ingest/batch", json={"items": [
        {"html": page_html("Gamma", "Quantum flywheel desynchronization, error 8812."),
         "source_url": "https://example.com/gamma"},
    ]})
    results = client.post("/api/v1/search", json={
        "query": "quantum flywheel desynchronization 8812", "top_k": 1,
    }).json()["results"]

    assert results[0]["source"] == "https://example.com/gamma"


def test_one_bad_page_does_not_fail_the_batch(client):
    """An agent crawling ten links should get the eight that worked."""
    resp = client.post("/api/v1/ingest/batch", json={"items": [
        {"html": page_html("Good", "A page with genuine extractable content in it."),
         "source_url": "https://example.com/good"},
        {"html": "<html><body></body></html>", "source_url": "https://example.com/blank"},
        {"url": "http://127.0.0.1/"},
    ]})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert [r["status"] for r in results] == ["ok", "extract_error", "fetch_error"]
    assert results[1]["error"] and results[2]["error"]
    assert results[2]["url"] == "http://127.0.0.1/"


def test_a_url_repeated_within_a_batch_is_indexed_once(client):
    """add_chunks() de-duplicates against what's already stored, but two
    copies inside a single call would both land."""
    html = page_html("Twice", "Content that appears twice in one batch request.")
    resp = client.post("/api/v1/ingest/batch", json={"items": [
        {"html": html, "source_url": "https://example.com/twice"},
        {"html": html, "source_url": "https://example.com/twice"},
    ]})
    results = resp.json()["results"]
    assert [r["status"] for r in results] == ["ok", "ok"]

    # Both items report their own structured page, but the index holds one
    # copy: as many chunks as a single ingest of that page produces.
    expected = len(results[0]["page"]["chunks"])
    stored = [
        c for c in app.state.retriever._ingested_chunks
        if c["source"] == "https://example.com/twice"
    ]
    assert len(stored) == expected


def test_batch_ingest_rejects_an_oversized_batch(client):
    resp = client.post("/api/v1/ingest/batch", json={
        "items": [{"url": "https://example.com/"}] * (MAX_BATCH_INGEST + 1)
    })
    assert resp.status_code == 422


def test_batch_ingest_rejects_a_malformed_item(client):
    """Each item is validated as a full IngestRequest, so 'neither url nor
    html' is still a schema error rather than a per-item failure."""
    resp = client.post("/api/v1/ingest/batch", json={"items": [{}]})
    assert resp.status_code == 422


def test_batch_of_only_failures_still_answers(client):
    """add_chunks([]) is the path taken when nothing succeeded."""
    resp = client.post("/api/v1/ingest/batch", json={"items": [
        {"html": "<html><body></body></html>", "source_url": "https://example.com/x"},
    ]})
    assert resp.status_code == 200
    assert resp.json()["results"][0]["status"] == "extract_error"


def test_batch_ingest_statuses_are_published_in_the_openapi_schema(client):
    """The per-item status is the only place a batch says why something
    failed, so an agent should be able to read the possible values off the
    schema rather than out of a prose description."""
    schema = client.get("/openapi.json").json()
    status = schema["components"]["schemas"]["BatchIngestResult"]["properties"]["status"]
    assert set(status["enum"]) == {"ok", "fetch_error", "extract_error"}
