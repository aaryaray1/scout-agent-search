from scout.web import ingest_html

SAMPLE_HTML = """
<html>
<head><title>Rate Limits</title></head>
<body>
  <article>
    <h1>Rate Limits</h1>
    <p>Error 1020 occurs when request volume exceeds the allowed rate
    limit. Reduce request frequency and implement exponential backoff to
    recover from this error safely and reliably.</p>
  </article>
</body>
</html>
"""


def test_ingest_html_produces_search_ready_chunks():
    title, metadata, chunks = ingest_html(SAMPLE_HTML, source_url="https://example.com/rate-limits")

    assert title == "Rate Limits"
    assert metadata["url"] == "https://example.com/rate-limits"
    assert len(chunks) > 0

    chunk = chunks[0]
    assert chunk["source"] == "https://example.com/rate-limits"
    assert chunk["type"] == "web"
    assert "rate limit" in chunk["content"].lower()
    assert chunk["order"] == 0
    assert "id" in chunk
