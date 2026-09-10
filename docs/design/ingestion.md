# Ingestion

Covers `scout/fetch.py`, `scout/webextract.py`, `scout/web.py` and
`scout/ingest.py`. This is the pipeline behind `/api/v1/ingest`, and the
actual point of the project: HTML in, structured JSON out, done once instead
of once per agent.

```
fetch_html  ->  extract_content  ->  doc_from_web_content  ->  chunk_docs
(fetch.py)      (webextract.py)      (ingest.py)              (ingest.py)
```

`web.py` ties them together as `ingest_url()` and `ingest_html()`. The stages
are separate modules so network, extraction and chunking stay independently
testable.

## Fetching, and the attack surface it opens

Fetching arbitrary URLs on a caller's behalf is the main risk Scout takes on.
A caller could otherwise point it at internal services or cloud metadata
endpoints.

**Scheme allowlist.** `http` and `https` only.

**SSRF guard.** The hostname is resolved and *every* returned address is
checked; any private, loopback, link-local, multicast, reserved or
unspecified address rejects the fetch. IPv4-mapped IPv6 forms are covered by
`ipaddress`, so `::ffff:169.254.169.254` is caught too.

**Redirects are handled manually.** `follow_redirects` stays off and each hop
is re-validated through the same guard, rather than trusting where a server
points next. The budget is 5 hops; exceeding it, or a redirect with no
`Location` header, raises an error naming what happened instead of surfacing
as whatever httpx said next.

**Known gap: DNS rebinding.** The host is validated at resolution time, not
at connection time, so a host that resolves safely during the check and
re-resolves to a private address for the actual connection is not covered.
Low risk while self-hosted; must close before any multi-tenant exposure.
Tracked in ROADMAP Phase 1.

## Resource limits

**Response size is capped while streaming.** This did not used to work: the
cap compared `len(response.content)` against the limit, by which point the
entire body was already in memory, and otherwise trusted `Content-Length`,
which a hostile server can simply omit or understate. `_read_capped_body()`
now streams in chunks and stops at the limit, dropping the connection. A test
serves an endless body and asserts the reader stops near the cap rather than
at the sender's discretion.

**Timeout.** 10s, so a slow server cannot hold a threadpool worker
indefinitely.

**Content type is validated before extraction.** A PDF, an image, or a JSON
error page served with a 200 fails fast with a specific error instead of a
generic "no extractable content" further down the pipeline. Servers that omit
the header are allowed through - there is nothing to validate against.

Bodies are never executed or rendered. Extraction is text-only parsing; a
headless browser is a deliberate, sandboxed later decision, not a default.

## Extraction

`trafilatura` does boilerplate removal (nav, ads, cookie banners) and returns
markdown, which slots straight into the same chunking path the local markdown
corpus goes through.

Tables and fenced code blocks survive by default. Two options are set
deliberately:

- `include_links=True` - citations in the source page carry their real URL
  into the evidence instead of being flattened to plain text. This was off
  originally, which was the one piece of structure actually being lost.
- `include_images=False` - an image URL alone is not useful evidence for a
  text-based agent, and there is no rendering surface for it.

`extract_metadata` returns `None` for fragments and error pages, so the
metadata dict keeps the same shape either way and callers never branch on it.

Nothing extractable raises `ValueError`, which the API turns into a 422: the
page is a login wall, an error page, or its content is JS-rendered. Scout
does not execute JavaScript (ROADMAP Phase 1).

## Chunking

`chunk_text()` splits into fixed-size overlapping word windows
(`chunk_size` 400, `chunk_overlap` 50). The window advances by
`chunk_size - overlap`; a non-positive stride would re-emit the same words
forever and fill memory, so it is rejected at config load and again here,
naming the setting.

Fixed windows ignore document structure and can split mid-section. Chunking
by heading would preserve more meaning per chunk, and is ROADMAP Phase 6 -
measured on the eval harness rather than assumed to be better.

`chunk_docs()` is the single chunking path for both local markdown and
fetched pages, which is why `doc_from_web_content()` exists: it shapes
extractor output into the same dict `load_markdown_docs()` produces.

`_chunks_for_doc()` never writes back to the doc it is handed. It used to
stamp a generated `id` onto the caller's own dict as a side effect, which is
a trap for the next caller - the CLI reuses those dicts for logging and tests
assert on them.

Fields other than `id`/`content`/`source`/`type` are carried through onto
every chunk, which is how page title and metadata reach the evidence.
