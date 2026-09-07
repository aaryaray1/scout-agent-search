"""The Python client, driven against the real app.

TestClient is itself an httpx.Client, so handing it to ScoutClient runs
every call through the actual ASGI stack -- routing, auth, validation,
retrieval -- without a socket or a server process. That's the whole
reason ScoutClient takes an injectable client: the same seam serves
tests, proxies and custom transports.
"""
import httpx
import pytest
from fastapi.testclient import TestClient

from scout.api import app
from scout.auth import API_KEY_HEADER
from scout.client import ScoutAPIError, ScoutClient, ScoutError
from scout.config import reset_config_cache

BASE_URL = "http://testserver"
KEY = "client-test-key"

HTML = (
    "<html><head><title>Client Page</title></head><body><article>"
    "<p>Hyperspindle overheating is reported as fault F-7741.</p>"
    "</article></body></html>"
)


@pytest.fixture
def scout():
    with TestClient(app) as transport:
        yield ScoutClient(base_url=BASE_URL, client=transport)


@pytest.fixture
def keyed_scout(monkeypatch):
    monkeypatch.setenv("SCOUT_API_KEYS", KEY)
    reset_config_cache()
    with TestClient(app) as transport:
        yield transport
    reset_config_cache()


def test_health(scout):
    body = scout.health()
    assert body["status"] == "ok"
    assert "corpus_size" in body


def test_search_returns_the_evidence_list(scout):
    results = scout.search("rate limit exceeded", top_k=2)
    assert len(results) == 2
    assert {"content", "source", "confidence", "metadata"} <= set(results[0])


def test_search_batch_returns_one_list_per_query(scout):
    batches = scout.search_batch(["rate limit exceeded", "authentication"], top_k=1)
    assert len(batches) == 2
    assert all(len(b) == 1 for b in batches)


def test_ingest_html_then_search_finds_it(scout):
    page = scout.ingest_html(HTML, "https://example.com/client-page")
    assert page["title"] == "Client Page"
    assert page["chunks"]

    results = scout.search("hyperspindle overheating F-7741", top_k=1)
    assert results[0]["source"] == "https://example.com/client-page"


def test_ingest_batch_reports_per_item_status(scout):
    results = scout.ingest_batch([
        {"html": HTML, "source_url": "https://example.com/batch-a"},
        {"html": "<html><body></body></html>", "source_url": "https://example.com/blank"},
    ])
    assert [r["status"] for r in results] == ["ok", "extract_error"]


def test_a_failed_page_in_a_batch_does_not_raise(scout):
    """Per-item failures are data, not exceptions: the batch succeeded."""
    results = scout.ingest_batch([{"url": "http://127.0.0.1/"}])
    assert results[0]["status"] == "fetch_error"


def test_server_errors_become_ScoutAPIError(scout):
    with pytest.raises(ScoutAPIError) as exc:
        scout.ingest_url("http://127.0.0.1/")
    assert exc.value.status_code == 502
    assert exc.value.detail


def test_validation_errors_carry_their_status(scout):
    with pytest.raises(ScoutAPIError) as exc:
        scout.search("   ")
    assert exc.value.status_code == 422


def test_the_api_key_is_sent_on_every_call(keyed_scout):
    with ScoutClient(base_url=BASE_URL, api_key=KEY, client=keyed_scout) as scout:
        assert scout.search("rate limit", top_k=1)


def test_a_missing_key_surfaces_as_a_401(keyed_scout):
    with ScoutClient(base_url=BASE_URL, client=keyed_scout) as scout:
        with pytest.raises(ScoutAPIError) as exc:
            scout.search("rate limit")
    assert exc.value.status_code == 401


def test_rate_limit_errors_expose_retry_after(monkeypatch):
    """429 is the one error a client is expected to act on rather than
    report, so Retry-After is parsed out instead of left in the message."""
    monkeypatch.setenv("SCOUT_INGEST_RATE_LIMIT", "1")
    reset_config_cache()
    with TestClient(app) as transport:
        scout = ScoutClient(base_url=BASE_URL, client=transport)
        scout.ingest_html(HTML, "https://example.com/first")
        with pytest.raises(ScoutAPIError) as exc:
            scout.ingest_html(HTML, "https://example.com/second")
    reset_config_cache()

    assert exc.value.status_code == 429
    assert exc.value.retry_after > 0


def test_an_unreachable_server_raises_ScoutError_not_httpx():
    """A caller should be able to catch one exception type from this
    module rather than one from its HTTP dependency."""
    def refuse(request):
        raise httpx.ConnectError("connection refused")

    with httpx.Client(transport=httpx.MockTransport(refuse)) as transport:
        scout = ScoutClient(base_url="http://scout.invalid", client=transport)
        with pytest.raises(ScoutError) as exc:
            scout.health()
    assert "could not reach Scout" in str(exc.value)


def test_a_non_json_error_body_is_still_reported():
    """A proxy or load balancer answering instead of Scout returns HTML,
    which shouldn't turn into a JSON parse error on the caller's side."""
    def gateway_error(request):
        return httpx.Response(503, text="<html><body>upstream down</body></html>")

    with httpx.Client(transport=httpx.MockTransport(gateway_error)) as transport:
        scout = ScoutClient(base_url="http://scout.invalid", client=transport)
        with pytest.raises(ScoutAPIError) as exc:
            scout.health()
    assert exc.value.status_code == 503
    assert "upstream down" in exc.value.detail


def test_a_newer_major_schema_version_warns(caplog):
    """schema_version exists so a caller can tell when the contract moved;
    the client surfaces that instead of silently mis-reading fields."""
    def future_server(request):
        return httpx.Response(200, json={"schema_version": "2.0", "status": "ok"})

    with httpx.Client(transport=httpx.MockTransport(future_server)) as transport:
        scout = ScoutClient(base_url="http://scout.invalid", client=transport)
        with caplog.at_level("WARNING", logger="scout.client"):
            scout.health()

    assert "schema_version 2.0" in caplog.text


def test_an_injected_client_is_not_closed_by_scout():
    """Whoever opened the connection pool owns it."""
    transport = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    ScoutClient(base_url=BASE_URL, client=transport).close()
    assert not transport.is_closed
    transport.close()
