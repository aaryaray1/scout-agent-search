# Scout Roadmap

Ordered by dependency, not strictly by priority. Phase 1 is the actual point of the project.

## Phase 0 - local search prototype (done)

A hybrid vector + keyword retriever over a local markdown corpus, a FastAPI search endpoint, a CLI to build/refresh the index. Exists mainly to prove out retrieval and scoring before ingestion. Known limitations, carried into later phases:

- In-memory linear scan over `numpy` arrays - fine for a demo corpus, won't hold up past a few thousand chunks.
- Score fusion weights (`vector_weight`/`keyword_weight` in `config.json`) are guessed, not tuned against a labeled relevance set.
- Fixed-size word chunking (`chunk_text`) ignores document structure and can split mid-section. Chunking by heading would preserve more meaning per chunk.

Closed since first written:

- `Retriever()` used to reload the embedding model from disk on every instantiation. `scout/embeddings.py` now caches the loaded `SentenceTransformer` per model name at module level, so it's loaded once per process regardless of how many `Retriever()`/`EmbeddingModel()` instances get created. The test suite dropped from repeatedly reloading it in almost every test to loading it once.
- Keyword scoring was raw token overlap (`scout/keyword.py`, since removed): set intersection with no notion of term rarity or frequency. Replaced with real BM25 (`scout/bm25.py`), implemented directly rather than pulling in a search library, since the formula is compact and this way it's fully under test (including against hand-computed reference values). Raw BM25 scores are unbounded, so they're min-max normalized to [0, 1] before fusing with cosine similarity, keeping `vector_weight`/`keyword_weight` meaningful.

## Phase 1 - HTML to JSON ingestion (the actual differentiator)

First pass scaffolded. `POST /api/v1/ingest` takes `{"url": ...}` (Scout fetches it) or `{"html": ..., "source_url": ...}` (caller already has the HTML), returns structured JSON immediately, and indexes it in-memory so it's searchable via `/api/v1/search` without a rebuild step.

- **Fetcher** (`scout/fetch.py`): httpx-based, manual redirect handling with each hop re-validated, 10s timeout, 5MB response cap, SSRF guard that resolves the hostname and rejects private/loopback/link-local/reserved addresses before connecting.
- **Extractor** (`scout/webextract.py`): delegates to `trafilatura` for boilerplate removal, asks for markdown output plus title/author/date/sitename metadata.
- **Normalizer** (`scout/ingest.py::doc_from_web_content`): wraps extracted content in the same doc shape `load_markdown_docs()` produces, so it flows through `chunk_docs()` unchanged.
- **Orchestration** (`scout/web.py`): ties fetch, extract, and chunk together as `ingest_url()` / `ingest_html()`.
- **Retriever.add_chunks()**: embeds new chunks, drops any existing ingested chunks with a matching source, and persists the result, so re-ingesting a URL updates it in place instead of duplicating it.

Open before this is production-ready:

- SSRF guard checks at DNS resolution time, not connection time. A DNS-rebinding attack (host resolves safely during the check, then re-resolves to a private IP for the actual connection) isn't covered. Low risk while self-hosted, should be closed before any multi-tenant exposure.
- JS-rendered pages return nothing, since `trafilatura` only sees the HTML as served. A headless-browser fallback is a deliberate future decision, not a default, given the sandboxing it needs.
- `schema_version` (below) is a single string on the response envelope, not a real compatibility contract: nothing validates it, negotiates it, or documents what changes between versions. Fine as a foundation, not sufficient on its own.

Closed since the first pass:

- Content-type validation now runs in `scout/fetch.py` before extraction, so a non-HTML response (a PDF, a JSON error page served with a 200) fails fast with a specific error instead of a generic `ValueError` further down the pipeline.
- Ingested pages now survive a restart. `add_chunks()` persists new chunks and embeddings through `scout/index.py::save_ingested`, and `Retriever.__init__` reloads them via `load_ingested()` on startup. Verified live: ingested a page, killed the server process, restarted it, and the content was still searchable with no re-ingest. This is a stepping stone, not the real Phase 2 store: every ingest rewrites the entire ingested store to disk.
- Re-ingesting the same URL now replaces its old chunks instead of piling up duplicates alongside them. `Retriever` tracks the docs_path corpus and the ingested corpus as two separate layers (`_docs_*` / `_ingested_*`) concatenated into the live `self.corpus`; `add_chunks()` drops any existing ingested chunks matching the incoming chunks' `source` before adding the new ones. Verified live: ingested the same URL twice with different content, `corpus_size` stayed the same both times, and only the latest version was searchable, including after a restart. Only covers exact same-URL re-ingestion, not near-duplicate content from different URLs (Phase 2).
- "No structure beyond markdown text" turned out to be only half true. Checked empirically: `trafilatura` already preserves tables (as markdown tables) and multi-line code (as fenced blocks) by default; only link URLs were actually being dropped, because `webextract.py` explicitly passed `include_links=False`. Flipped to `True`, so a citation like "see the related docs" now keeps its real URL instead of being flattened to plain text. `include_images=False` stays as a deliberate choice, not an oversight: an image URL alone isn't useful evidence for a text-based agent.
- `Evidence`/`SearchResponse`/`IngestResponse` now carry a `schema_version` field (`scout/models.py`, currently `"1.0"`), giving agents something concrete to check instead of guessing compatibility from field presence. See the versioning caveat above for what this doesn't yet do.
- `scout/api.py` used to build its `Retriever` as a module-level singleton at import time, which meant merely importing the module (as any test file must) loaded the embedding model and touched disk, with no way to isolate that from a test's own storage. Refactored to build it in a FastAPI lifespan handler instead, and added `tests/test_api.py`, the first tests that exercise the actual HTTP layer rather than calling into `Retriever` directly.

## Phase 2 - storage that scales past a demo corpus

- Swap the flat `numpy` array and JSON metadata cache (now including the ingested-pages store added in Phase 1) for a real vector store. `qdrant-client` sat in requirements.txt early in the project's history, which suggests it was the original intent. Either bring it back deliberately (e.g. via `docker-compose` for local dev) or document a different choice.
- True incremental indexing. The docs_path corpus still forces a full re-embed on any change, and every ingest rewrites the entire ingested store to disk, rebuilds the combined in-memory corpus, and rebuilds the BM25 index from scratch, even though only one source's worth of chunks actually changed.
- Near-duplicate detection across different URLs. Exact same-URL re-ingestion is now handled (Phase 1); two different URLs serving the same or near-identical content still produce separate chunks.

Closed since first written: real BM25 keyword scoring, tracked above under Phase 0 (it replaced Phase 0's token-overlap scoring, so that's where the detail lives) rather than pulling in `whoosh` as originally floated here.

## Phase 3 - API and agent-facing ergonomics

- Auth (even a simple API-key header) before this is exposed beyond localhost.
- Rate limiting and request size limits on `/ingest`, since it does outbound fetches on the caller's behalf.
- A minimal Python client SDK so agent frameworks don't hand-roll HTTP calls against the API.
- Batch endpoints (`/api/v1/ingest/batch`, multi-query search) for agents that want to fan out.

## Phase 4 - open source readiness

- CONTRIBUTING.md and issue/PR templates.
- CI (GitHub Actions) running `pytest` on push/PR. Tests currently need the `all-MiniLM-L6-v2` model available locally or cached; CI needs either a cached-model step or a lightweight fake embedding backend for fast, network-independent runs.
- Versioned releases and a changelog once the API surface stabilizes.

## Security notes for Phase 1

Fetching arbitrary URLs on a caller's behalf is the main new attack surface Scout takes on.

- SSRF: block requests to internal/private IP ranges and cloud metadata endpoints (`169.254.169.254`, etc.) unless explicitly allow-listed.
- Resource limits: cap response size and fetch timeout so a malicious or huge page can't exhaust memory or CPU.
- Content sanitization: don't execute or render fetched HTML/JS. Extraction should stay text-only parsing; a headless browser is a deliberate, sandboxed later addition, not a default.
