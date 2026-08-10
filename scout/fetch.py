"""HTTP fetching for Scout's ingest pipeline.

This fetches arbitrary URLs on behalf of callers, which is real attack
surface (SSRF): a caller could ask Scout to request internal services,
cloud metadata endpoints, etc. The guards below are a first pass, not a
complete answer. See ROADMAP.md's "Security notes" for known gaps, notably
DNS rebinding: the host is validated at resolution time, not at the moment
of connection.
"""
import ipaddress
import socket
from urllib.parse import urlparse

import httpx

USER_AGENT = "ScoutBot/0.1 (+https://github.com/scout-agent-search; agent-search ingestion)"
DEFAULT_TIMEOUT = 10.0
MAX_RESPONSE_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_REDIRECTS = 5
ALLOWED_SCHEMES = {"http", "https"}


class FetchError(Exception):
    """Raised when a URL can't be fetched or is rejected before fetching."""


class SSRFBlockedError(FetchError):
    """Raised when a URL resolves to a non-public address."""


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

    for _family, _type, _proto, _canonname, sockaddr in infos:
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


def fetch_html(url: str, timeout: float = DEFAULT_TIMEOUT, max_bytes: int = MAX_RESPONSE_BYTES) -> str:
    """Fetch `url` and return its response body as text.

    Redirects are followed manually (rather than via httpx's built-in
    follow_redirects) so every hop gets re-validated against the SSRF guard
    instead of trusting whatever the server points at next.
    """
    _validate_url(url)

    headers = {"User-Agent": USER_AGENT}
    try:
        with httpx.Client(follow_redirects=False, timeout=timeout, headers=headers) as client:
            response = client.get(url)

            redirects = 0
            while response.is_redirect and redirects < MAX_REDIRECTS:
                next_url = response.headers.get("location")
                if not next_url:
                    break
                next_url = str(httpx.URL(response.url).join(next_url))
                _validate_url(next_url)
                response = client.get(next_url)
                redirects += 1

            response.raise_for_status()

            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > max_bytes:
                raise FetchError(
                    f"response too large ({content_length} bytes > {max_bytes} limit)"
                )
            if len(response.content) > max_bytes:
                raise FetchError(
                    f"response too large ({len(response.content)} bytes > {max_bytes} limit)"
                )

            return response.text
    except httpx.HTTPError as e:
        raise FetchError(f"failed to fetch '{url}': {e}") from e
