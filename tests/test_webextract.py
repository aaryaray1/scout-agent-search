import pytest
from scout.webextract import extract_content

SAMPLE_HTML = """
<html>
<head>
  <title>Understanding Error 1008</title>
  <meta name="author" content="Jane Doe">
</head>
<body>
  <nav>Home | Docs | Pricing</nav>
  <article>
    <h1>Understanding Error 1008</h1>
    <p>Error 1008 means the account does not have a sufficient balance to
    process the request. This is one of the most common billing errors
    reported by API consumers integrating with the platform.</p>
    <h2>Resolution</h2>
    <p>Add funds to the account and retry the request once the balance has
    been updated in the billing dashboard.</p>
  </article>
  <footer>Copyright 2026</footer>
</body>
</html>
"""


def test_extract_content_pulls_title_and_text():
    result = extract_content(SAMPLE_HTML, url="https://example.com/errors/1008")
    assert "1008" in result["title"]
    assert "sufficient balance" in result["content"].lower()
    # boilerplate should not leak into the extracted content
    assert "Copyright" not in result["content"]
    assert result["metadata"]["url"] == "https://example.com/errors/1008"


def test_extract_content_raises_on_empty_html():
    with pytest.raises(ValueError):
        extract_content("<html><body></body></html>", url="https://example.com/blank")
