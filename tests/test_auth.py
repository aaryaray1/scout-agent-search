"""API-key auth, at both the unit and the HTTP level.

The HTTP tests set SCOUT_API_KEYS before building the client, pinning down
that keys come from the environment at startup, not per request.
"""
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from scout.api import app
from scout.auth import API_KEY_HEADER, ApiKeyAuth, fingerprint
from scout.config import reset_config_cache

KEY = "test-key-one"
OTHER_KEY = "test-key-two"


@pytest.fixture
def keyed_client(monkeypatch):
    """A running app with two API keys configured."""
    monkeypatch.setenv("SCOUT_API_KEYS", f"{KEY},{OTHER_KEY}")
    reset_config_cache()
    with TestClient(app) as client:
        yield client
    reset_config_cache()


@pytest.fixture
def open_client():
    """A running app with no keys configured (the default)."""
    reset_config_cache()
    with TestClient(app) as client:
        yield client


# -- unit ---------------------------------------------------------------------

def test_auth_is_disabled_with_no_keys():
    auth = ApiKeyAuth([])
    assert not auth.enabled
    assert auth.identify(None, "1.2.3.4") == "ip:1.2.3.4"


def test_blank_keys_do_not_become_credentials():
    """A whitespace-only entry in config.json would otherwise be a live key
    that an empty-ish header matches."""
    auth = ApiKeyAuth(["  ", ""])
    assert not auth.enabled


def test_identity_is_the_key_fingerprint_not_the_key():
    auth = ApiKeyAuth([KEY])
    identity = auth.identify(KEY, "1.2.3.4")
    assert identity == f"key:{fingerprint(KEY)}"
    assert KEY not in identity


def test_non_ascii_key_is_rejected_not_crashed():
    """secrets.compare_digest raises TypeError on str inputs with non-ASCII
    characters, and the header is caller-controlled, so comparing as bytes
    is what keeps this a 401 instead of a 500."""
    auth = ApiKeyAuth([KEY])
    with pytest.raises(HTTPException) as exc:
        auth.identify("kéy", "1.2.3.4")
    assert exc.value.status_code == 401


# -- HTTP ---------------------------------------------------------------------

def test_search_requires_a_key_when_auth_is_enabled(keyed_client):
    resp = keyed_client.post("/api/v1/search", json={"query": "rate limit"})
    assert resp.status_code == 401
    assert API_KEY_HEADER in resp.json()["detail"]


def test_search_rejects_a_wrong_key(keyed_client):
    resp = keyed_client.post(
        "/api/v1/search", json={"query": "rate limit"}, headers={API_KEY_HEADER: "nope"}
    )
    assert resp.status_code == 401


def test_search_accepts_any_configured_key(keyed_client):
    for key in (KEY, OTHER_KEY):
        resp = keyed_client.post(
            "/api/v1/search", json={"query": "rate limit"}, headers={API_KEY_HEADER: key}
        )
        assert resp.status_code == 200


def test_ingest_requires_a_key_when_auth_is_enabled(keyed_client):
    resp = keyed_client.post("/api/v1/ingest", json={"url": "https://example.com/"})
    assert resp.status_code == 401


@pytest.mark.parametrize("path", ["/api/v1/search/batch", "/api/v1/ingest/batch"])
def test_batch_endpoints_are_guarded_too(keyed_client, path):
    """A new endpoint that forgets its dependency is the obvious way for
    auth to silently regress."""
    resp = keyed_client.post(path, json={})
    assert resp.status_code == 401


def test_health_and_root_stay_open(keyed_client):
    """A liveness probe shouldn't need a credential."""
    assert keyed_client.get("/health").status_code == 200
    assert keyed_client.get("/").status_code == 200


def test_health_reports_that_auth_is_on(keyed_client):
    assert keyed_client.get("/health").json()["auth"] == "enabled"


def test_open_deployment_serves_without_a_key(open_client):
    resp = open_client.post("/api/v1/search", json={"query": "rate limit"})
    assert resp.status_code == 200
    assert open_client.get("/health").json()["auth"] == "disabled"


def test_the_key_header_is_published_in_the_openapi_schema(keyed_client):
    """Same contract-discovery principle as the request-shape limits: an
    agent should be able to read that it needs a key."""
    schema = keyed_client.get("/openapi.json").json()
    schemes = schema["components"]["securitySchemes"]
    assert any(s.get("name") == API_KEY_HEADER for s in schemes.values())
