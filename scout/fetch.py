"""HTTP fetching for the ingest pipeline, with SSRF and size guards.

Caller-chosen URLs are Scout's main attack surface; docs/design/ingestion.md
covers what is guarded and what is not (notably DNS rebinding).
"""
import ipaddress
import logging
import socket
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

USER_AGENT = "ScoutBot/0.1 (+https://github.com/scout-agent-search; agent-search ingestion)"
DEFAULT_TIMEOUT = 10.0
MAX_RESPONSE_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_REDIRECTS = 5
ALLOWED_SCHEMES = {"http", "https"}


class FetchError(Exception):
    """Raised when a URL can't be fetched or is rejected before fetching."""


class SSRFBlockedError(FetchError):
    """Raised when a URL resolves to a non-public address."""


class ResponseTooLargeError(FetchError):
    """Raised when a response exceeds the byte cap."""


def _is_blocked_ip(ip_str: str) -> bool:
    ip = ipaddress.ip_address(ip_str)
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _assert_safe_host(host: str) -> None:
    """Resolve `host` and reject it if any resolved address is non-public."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise FetchError(f"could not resolve host '{host}': {e}") from e

    for *_, sockaddr in infos:
        ip_str = sockaddr[0]
        if _is_blocked_ip(ip_str):
            raise SSRFBlockedError(
                f"'{host}' resolves to non-public address {ip_str}; refusing to fetch"
            )


def _validate_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise FetchError(
            f"unsupported URL scheme '{parsed.scheme}'; only http/https are allowed"
        )
    if not parsed.hostname:
        raise FetchError(f"could not parse a hostname out of '{url}'")
    _assert_safe_host(parsed.hostname)
    return url


def _assert_html_content_type(response) -> None:
    """Reject a PDF, an image or a JSON error page before extraction, which
    would otherwise fail generically further down the pipeline."""
    content_type = response.headers.get("content-type", "")
    if not content_type:
        return  # some servers omit it; nothing to validate against
    main_type = content_type.split(";")[0].strip().lower()
    if "html" not in main_type:
        raise FetchError(
            f"expected an HTML response, got content-type '{main_type}'"
        )


def _build_client(timeout: float) -> httpx.Client:
    """Build the client for one fetch, with redirects left to _follow_redirects
    so every hop is re-validated. Factored out for httpx.MockTransport."""
    return httpx.Client(
        follow_redirects=False,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
    )


def _send(client: httpx.Client, url: str) -> httpx.Response:
    """GET `url`, leaving the body unread so status and headers can be
    checked before committing memory to it."""
    return client.send(client.build_request("GET", url), stream=True)


def _follow_redirects(client: httpx.Client, url: str) -> httpx.Response:
    """Walk the redirect chain, re-validating each hop against the SSRF
    guard, and return the first non-redirect response."""
    response = _send(client, url)
    for _ in range(MAX_REDIRECTS):
        if not response.is_redirect:
            return response

        location = response.headers.get("location")
        next_url = str(response.url.join(location)) if location else None
        response.close()
        if not next_url:
            raise FetchError(f"redirect from '{url}' had no Location header")

        _validate_url(next_url)
        response = _send(client, next_url)

    response.close()
    raise FetchError(f"'{url}' exceeded {MAX_REDIRECTS} redirects")


def _read_capped_body(response: httpx.Response, max_bytes: int) -> str:
    """Read the body, stopping as soon as it passes `max_bytes`.

    Streamed rather than read whole: Content-Length is optional and a
    hostile server can omit it, so a cap applied afterwards protects nothing.
    """
    declared = response.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > max_bytes:
        raise ResponseTooLargeError(
            f"response too large ({declared} bytes > {max_bytes} limit)"
        )

    body = bytearray()
    for chunk in response.iter_bytes():
        body.extend(chunk)
        if len(body) > max_bytes:
            raise ResponseTooLargeError(
                f"response exceeded the {max_bytes} byte limit"
            )
    return bytes(body).decode(response.charset_encoding or "utf-8", errors="replace")


def fetch_html(url: str, timeout: float = DEFAULT_TIMEOUT, max_bytes: int = MAX_RESPONSE_BYTES) -> str:
    """Fetch `url` and return its response body as text."""
    _validate_url(url)
    logger.info("fetching %s", url)

    try:
        with _build_client(timeout) as client:
            response = _follow_redirects(client, url)
            try:
                response.raise_for_status()
                _assert_html_content_type(response)
                return _read_capped_body(response, max_bytes)
            finally:
                response.close()
    except httpx.HTTPError as e:
        raise FetchError(f"failed to fetch '{url}': {e}") from e
