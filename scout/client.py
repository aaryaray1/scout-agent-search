"""Python client for a running Scout server.

The point of Scout is that an agent doesn't have to write the glue
between "I have a URL" and "I have structured evidence". A framework
hand-rolling `httpx.post(...)` against these endpoints has to rediscover
the auth header, the error shape and the batch payloads every time, so
that glue lives here once.

Responses come back as plain dicts rather than the Pydantic models in
scout/models.py, deliberately. Parsing into the server's own models would
mean a client one version behind the server rejects a response carrying a
field it hasn't heard of yet -- exactly the coupling `schema_version` is
supposed to let callers manage themselves. The version is checked, not
enforced: a mismatched major version warns, because the caller is in a
better position than this module to decide whether it can still work.
"""
import logging

import httpx

from .auth import API_KEY_HEADER
from .models import EVIDENCE_SCHEMA_VERSION

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
# Ingest can spend a full outbound fetch timeout (10s) plus extraction and
# embedding before it answers, so the client's patience is set well above
# the server's own fetch timeout rather than at the usual few seconds.
DEFAULT_TIMEOUT = 60.0


class ScoutError(Exception):
    """Base class for every error this client raises."""


class ScoutAPIError(ScoutError):
    """The server answered, and said no.

    `status_code` and `detail` are pulled apart rather than flattened into
    a message so a caller can branch on them: 401 means fix your key, 429
    means back off (see `retry_after`), 502 means the page was
    unreachable and retrying may well work.
    """

    def __init__(self, status_code, detail, retry_after=None):
        super().__init__(f"Scout returned {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail
        self.retry_after = retry_after


def _detail_of(response):
    """Pull FastAPI's `detail` out of an error body, falling back to text.

    A non-JSON body means something other than Scout answered -- a proxy,
    a load balancer, an error page -- which is worth surfacing verbatim
    instead of hiding behind a parse failure.
    """
    try:
        body = response.json()
    except ValueError:
        return response.text.strip()[:500] or response.reason_phrase
    if isinstance(body, dict) and "detail" in body:
        return body["detail"]
    return str(body)


def _retry_after_of(response):
    raw = response.headers.get("retry-after", "")
    return int(raw) if raw.isdigit() else None


class ScoutClient:
    """Synchronous client for Scout's HTTP API.

    Usable as a context manager, which is the recommended form since it
    closes the underlying connection pool::

        with ScoutClient("http://localhost:8000", api_key="...") as scout:
            page = scout.ingest_url("https://example.com/docs")
            hits = scout.search("how do I authenticate")
    """

    def __init__(
        self,
        base_url=DEFAULT_BASE_URL,
        api_key=None,
        timeout=DEFAULT_TIMEOUT,
        client=None,
    ):
        """`client` accepts a pre-built httpx.Client.

        That's what makes this testable without a live server (an
        ASGITransport pointed at the app), and it's also the hook a caller
        needs for a proxy, a custom retry transport or mTLS. A client
        passed in is not closed by this one, on the principle that
        whoever opened it owns it.
        """
        self.base_url = base_url.rstrip("/")
        headers = {API_KEY_HEADER: api_key} if api_key else {}
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout)
        self._headers = headers

    # -- plumbing ------------------------------------------------------------

    def close(self):
        if self._owns_client:
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def _check_schema_version(self, body):
        """Warn when the server's evidence schema has moved on.

        Compares major versions only: "1.1" adding a field is something a
        tolerant caller survives, "2.0" reshaping evidence is not.
        """
        served = str(body.get("schema_version", ""))
        if served and served.split(".")[0] != EVIDENCE_SCHEMA_VERSION.split(".")[0]:
            logger.warning(
                "Scout server speaks schema_version %s, this client was built "
                "against %s; response fields may have moved",
                served, EVIDENCE_SCHEMA_VERSION,
            )
        return body

    def _request(self, method, path, json=None):
        try:
            response = self._client.request(
                method, f"{self.base_url}{path}", json=json, headers=self._headers
            )
        except httpx.HTTPError as e:
            raise ScoutError(f"could not reach Scout at {self.base_url}: {e}") from e

        if response.is_error:
            raise ScoutAPIError(
                response.status_code, _detail_of(response), _retry_after_of(response)
            )
        return self._check_schema_version(response.json())

    # -- endpoints -----------------------------------------------------------

    def health(self):
        """Server liveness, corpus size, and whether auth is on."""
        return self._request("GET", "/health")

    def search(self, query, top_k=None, agent_id="default"):
        """Return the evidence list for one query."""
        payload = {"query": query, "agent_id": agent_id}
        if top_k is not None:
            payload["top_k"] = top_k
        return self._request("POST", "/api/v1/search", json=payload)["results"]

    def search_batch(self, queries, top_k=None, agent_id="default"):
        """Return one evidence list per query, in the order asked."""
        payload = {"queries": list(queries), "agent_id": agent_id}
        if top_k is not None:
            payload["top_k"] = top_k
        body = self._request("POST", "/api/v1/search/batch", json=payload)
        return [item["results"] for item in body["results"]]

    def ingest_url(self, url):
        """Have Scout fetch `url` and return it as structured JSON."""
        return self._request("POST", "/api/v1/ingest", json={"url": url})

    def ingest_html(self, html, source_url):
        """Structure HTML the caller already has (from its own browser
        session, say), attributed to `source_url`."""
        return self._request(
            "POST", "/api/v1/ingest", json={"html": html, "source_url": source_url}
        )

    def ingest_batch(self, items):
        """Ingest several pages in one call.

        `items` are dicts in /ingest's own shape: {"url": ...} or
        {"html": ..., "source_url": ...}. Returns one result per item, in
        order, each with its own `status` -- a page that failed doesn't
        raise, because the others succeeded.
        """
        body = self._request(
            "POST", "/api/v1/ingest/batch", json={"items": list(items)}
        )
        return body["results"]
