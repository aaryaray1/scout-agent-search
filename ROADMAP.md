# Scout Roadmap

This captures where Scout is and what needs to be built next. It's ordered
roughly by dependency — later phases build on earlier ones — not strictly by
priority, though Phase 1 is the actual point of the project.

## Phase 0 - Local search prototype (done)

A working hybrid vector+keyword retriever over a local markdown corpus, with
a FastAPI search endpoint and a CLI to build/refresh the index. This exists
mainly to prove out the retrieval and scoring logic before the harder problem
(ingestion) is tackled. Current known limitations, carried into later phases:

- In-memory linear scan over `numpy` arrays - fine for a demo corpus, won't
  hold up past a few thousand chunks.
- Keyword scoring is raw token-overlap (`scout/keyword.py`), not BM25/TF-IDF
  - cheap but weak; a `whoosh`-backed inverted index would score plausibility
  and rare-term weighting much better.
- Score fusion weights (`vector_weight`/`keyword_weight` in `config.json`)
  are guessed, not tuned against any labeled relevance set.
- Fixed-size word chunking (`chunk_text`) ignores document structure (it can
  split mid-section); chunking by heading/section would preserve more
  meaning per chunk.

## Phase 1 — HTML → JSON ingestion (the actual differentiator)

**First pass scaffolded.** `POST /api/v1/ingest` takes either `{"url": ...}`
(Scout fetches it) or `{"html": ..., "source_url": ...}` (caller already has
the HTML) and returns structured JSON immediately, indexing it in-memory so
it's searchable via `/api/v1/search` in the same process without a rebuild
step. The pieces:

1. **Fetcher** (`scout/fetch.py`) — `httpx`-based, manual redirect handling
   (each hop re-validated), 10s timeout, 5MB response cap, and an SSRF guard
   that resolves the hostname and rejects private/loopback/link-local/
   reserved addresses before connecting.
2. **Extractor** (`scout/webextract.py`) — delegates to `trafilatura` for
   boilerplate removal, asks for markdown output plus title/author/date/
   sitename metadata.
3. **Normalizer** (`scout/ingest.py::doc_from_web_content`) — wraps
   extracted content in the same doc shape `load_markdown_docs()` produces,
   so it flows through the existing `chunk_docs()` unchanged.
4. **Orchestration** (`scout/web.py`) — ties fetch → extract → chunk
   together as `ingest_url()` / `ingest_html()`.
5. **Retriever.add_chunks()** — embeds and appends new chunks to the live
   in-memory corpus/embeddings array.

What's still open before this is production-ready:

- **Ingested pages are in-memory only** — they vanish on restart. Durable
  storage is Phase 2.
- **No structure beyond markdown text** — headings survive as `#`/`##`,
  but tables, code blocks, and link references are currently discarded
  (`include_links=False`, `include_images=False` in `webextract.py`). Worth
  revisiting once there's a concrete consumer need for them.
- **SSRF guard checks at DNS-resolution time, not connection time** — a
  DNS-rebinding attack (hostname resolves safely during the check, then
  re-resolves to a private IP for the actual connection) isn't covered.
  Low risk for a self-hosted tool today, but should be closed before this
  is ever exposed multi-tenant.
- **JS-rendered pages return nothing** — `trafilatura` only sees the HTML
  as served; sites that render content client-side will hit the "no
  extractable content" error. A headless-browser fallback is a deliberate
  future decision, not a default, given the sandboxing it'd require.
- **No content-type/HTML validation before parsing** — a non-HTML response
  (e.g. a PDF or JSON error page served with a 200) currently just fails
  extraction with a generic `ValueError`; a clearer error would help.
- **`scout/models.py` evidence schema still isn't versioned** — it now
  covers both search and ingest responses, but nothing stops it drifting
  again as the code evolves.

## Phase 2 — Storage that scales past a demo corpus

- Swap the flat `numpy` array + JSON metadata cache for a real vector store.
  `qdrant-client` is already sitting in git history as a dependency, either bring it back deliberately
  (e.g. via `docker-compose` for local dev) or
  document a different choice.
- Add a `whoosh` (or similar) inverted index for the keyword half of hybrid
  search, replacing the current set-intersection scoring.
- Incremental indexing: right now any corpus change forces a full
  re-embed. Ingested pages should be upsertable by source URL/hash without
  rebuilding everything.
- De-duplication: re-ingesting the same URL (or near-duplicate content)
  shouldn't create redundant chunks.

## Phase 3 — API and agent-facing ergonomics

- Auth (even a simple API-key header) before this is exposed beyond
  localhost.
- Rate limiting / request size limits on `/ingest`, since it does outbound
  fetches on the caller's behalf (SSRF considerations apply — see below).
- A minimal Python client SDK (`scout-client`) so agent frameworks don't
  hand-roll HTTP calls against the API.
- Batch endpoints (`/api/v1/ingest/batch`, multi-query search) for agents
  that want to fan out.

## Phase 4 — Open-source readiness

- Pick and add a LICENSE (this is a decision for the maintainer, not
  something to default silently).
- CONTRIBUTING.md and issue/PR templates.
- CI (GitHub Actions): run `pytest` on push/PR. Currently tests need the
  `all-MiniLM-L6-v2` model available locally/cached — CI will need either a
  cached model step or a lightweight fake embedding backend for fast,
  network-independent test runs.
- Versioned releases / changelog once the API surface stabilizes.

## Security notes to design in from the start of Phase 1

Fetching arbitrary URLs on behalf of callers is the main new attack surface
Scout will take on:

- **SSRF**: block requests to internal/private IP ranges and cloud metadata
  endpoints (`169.254.169.254`, etc.) unless explicitly allow-listed.
- **Resource limits**: cap response size and fetch timeout so a malicious or
  huge page can't exhaust memory/CPU.
- **Content sanitization**: don't execute or render fetched HTML/JS —
  extraction should be text-only parsing, never a headless browser unless
  that's a deliberate, sandboxed later addition for JS-rendered pages.
