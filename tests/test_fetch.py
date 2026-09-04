"""Tests for the fetch layer's SSRF, redirect, content-type and size guards.

Everything below the SSRF tests runs against an httpx.MockTransport swapped
in through fetch._build_client, so the whole redirect/limit surface is
exercised without touching the network.
"""
import httpx
import pytest

from scout import fetch as fetch_module
from scout.fetch import (
    MAX_REDIRECTS,
    FetchError,
    ResponseTooLargeError,
    SSRFBlockedError,
    fetch_html,
)


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/",
    "http://localhost/",
    "http://[::1]/",
    "http://169.254.169.254/latest/meta-data/",  # cloud metadata endpoint
    "http://10.0.0.5/",
    "http://192.168.1.1/",
    "http://[::ffff:169.254.169.254]/",  # IPv4-mapped IPv6 form of the above
])
def test_fetch_blocks_non_public_hosts(url):
    with pytest.raises(SSRFBlockedError):
        fetch_html(url)


@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "ftp://example.com/file",
    "javascript:alert(1)",
])
def test_fetch_rejects_disallowed_schemes(url):
    with pytest.raises(FetchError):
        fetch_html(url)


def test_fetch_rejects_unresolvable_host():
    with pytest.raises(FetchError):
        fetch_html("http://this-host-should-not-resolve.invalid/")


@pytest.fixture
def serve(monkeypatch):
    """Serve responses from a handler function instead of the network.

    example.com is treated as a resolvable public host so tests don't need
    DNS; every other host still goes through the real SSRF check, which is
    what lets the redirect test below prove an internal hop gets blocked.
    """
    real_assert_safe_host = fetch_module._assert_safe_host

    def fake_assert_safe_host(host):
        if host == "example.com":
            return
        real_assert_safe_host(host)

    def install(handler):
        monkeypatch.setattr(fetch_module, "_assert_safe_host", fake_assert_safe_host)
        monkeypatch.setattr(
            fetch_module,
            "_build_client",
            lambda timeout: httpx.Client(
                transport=httpx.MockTransport(handler),
                follow_redirects=False,
                timeout=timeout,
            ),
        )
    return install


def _html(body=b"<html><body>hi</body></html>", content_type="text/html"):
    headers = {"content-type": content_type} if content_type else {}
    return httpx.Response(200, headers=headers, content=body)


def test_fetch_rejects_non_html_content_type(serve):
    serve(lambda request: _html(content_type="application/pdf"))
    with pytest.raises(FetchError, match="content-type"):
        fetch_html("https://example.com/whitepaper.pdf")


def test_fetch_allows_html_content_type(serve):
    serve(lambda request: _html(content_type="text/html; charset=utf-8"))
    assert "hi" in fetch_html("https://example.com/page")


def test_fetch_allows_missing_content_type(serve):
    serve(lambda request: _html(content_type=None))
    assert "hi" in fetch_html("https://example.com/page")


def test_fetch_decodes_declared_charset(serve):
    serve(lambda request: _html(
        body="<html><body>café</body></html>".encode("latin-1"),
        content_type="text/html; charset=latin-1",
    ))
    assert "café" in fetch_html("https://example.com/page")


def test_fetch_follows_redirects_to_the_final_page(serve):
    def handler(request):
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "/final"})
        return _html(b"<html><body>arrived</body></html>")

    serve(handler)
    assert "arrived" in fetch_html("https://example.com/start")


def test_fetch_revalidates_each_redirect_hop_against_the_ssrf_guard(serve):
    """A public URL that redirects to an internal one must still be blocked:
    this is why redirects are followed manually rather than by httpx."""
    serve(
        lambda request: httpx.Response(
            302, headers={"location": "http://169.254.169.254/latest/meta-data/"}
        )
    )
    with pytest.raises(SSRFBlockedError):
        fetch_html("https://example.com/start")


def test_fetch_gives_up_on_a_redirect_loop(serve):
    serve(lambda request: httpx.Response(302, headers={"location": "/loop"}))
    with pytest.raises(FetchError, match=f"exceeded {MAX_REDIRECTS} redirects"):
        fetch_html("https://example.com/loop")


def test_fetch_rejects_a_redirect_with_no_location(serve):
    serve(lambda request: httpx.Response(302))
    with pytest.raises(FetchError, match="no Location header"):
        fetch_html("https://example.com/start")


def test_fetch_rejects_an_oversized_declared_content_length(serve):
    serve(lambda request: httpx.Response(
        200,
        headers={"content-type": "text/html", "content-length": "999999"},
        content=b"x" * 999999,
    ))
    with pytest.raises(ResponseTooLargeError):
        fetch_html("https://example.com/big", max_bytes=1000)


def test_fetch_stops_reading_an_oversized_body_that_declares_no_length(serve):
    """The cap has to hold for a server that omits Content-Length, and it
    has to stop reading rather than buffer the whole body first -- otherwise
    a hostile page exhausts memory before the limit is ever consulted."""
    served = {"bytes": 0}

    def endless():
        while True:
            served["bytes"] += 8192
            yield b"x" * 8192

    serve(lambda request: httpx.Response(
        200, headers={"content-type": "text/html"}, content=endless()
    ))

    with pytest.raises(ResponseTooLargeError):
        fetch_html("https://example.com/endless", max_bytes=50_000)

    # Bounded by the cap plus at most one chunk, not by the sender.
    assert served["bytes"] < 100_000
