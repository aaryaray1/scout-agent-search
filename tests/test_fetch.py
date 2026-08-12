import httpx
import pytest
from scout import fetch as fetch_module
from scout.fetch import fetch_html, FetchError, SSRFBlockedError


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/",
    "http://localhost/",
    "http://[::1]/",
    "http://169.254.169.254/latest/meta-data/",  # cloud metadata endpoint
    "http://10.0.0.5/",
    "http://192.168.1.1/",
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


def _mock_get(monkeypatch, content_type, body=b"<html><body>hi</body></html>"):
    """Stub out DNS resolution and the actual HTTP GET so content-type
    handling can be tested without any real network access."""
    monkeypatch.setattr(fetch_module, "_assert_safe_host", lambda host: None)

    def fake_get(self, url, *args, **kwargs):
        headers = {"content-type": content_type} if content_type else {}
        return httpx.Response(200, headers=headers, content=body, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.Client, "get", fake_get)


def test_fetch_rejects_non_html_content_type(monkeypatch):
    _mock_get(monkeypatch, content_type="application/pdf")
    with pytest.raises(FetchError, match="content-type"):
        fetch_html("https://example.com/whitepaper.pdf")


def test_fetch_allows_html_content_type(monkeypatch):
    _mock_get(monkeypatch, content_type="text/html; charset=utf-8")
    assert "hi" in fetch_html("https://example.com/page")


def test_fetch_allows_missing_content_type(monkeypatch):
    _mock_get(monkeypatch, content_type=None)
    assert "hi" in fetch_html("https://example.com/page")
