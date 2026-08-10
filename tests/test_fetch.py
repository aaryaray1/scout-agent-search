import pytest
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
