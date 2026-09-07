# Scout Roadmap

Ordered by dependency, not strictly by priority. Phase 1 is the actual point of the project.

## Phase 0 - local search prototype (done)

A hybrid vector + keyword retriever over a local markdown corpus, a FastAPI search endpoint, a CLI to build/refresh the index. Exists mainly to prove out retrieval and scoring before ingestion. Known limitations, carried into later phases:

- In-memory linear scan over `numpy` arrays - fine for a demo corpus, won't hold up past a few thousand chunks.
- Score fusion weights (`vector_weight`/`keyword_weight`) are guessed, not tuned against a labeled relevance set.
- Fixed-size word chunking (`chunk_text`) ignores document structure and can split mid-section. Chunking by heading would preserve more meaning per chunk.

Closed since first written:

- `Retriever()` used to reload the embedding model from disk on every instantiation. `scout/embeddings.py` now caches the loaded `SentenceTransformer` per model name at module level, so it's loaded once per process regardless of how many `Retriever()`/`EmbeddingModel()` instances get created.
- Keyword scoring was raw token overlap (`scout/keyword.py`, since removed): set intersection with no notion of term rarity or frequency. Replaced with real BM25 (`scout/bm25.py`), implemented directly rather than pulling in a search library, since the formula is compact and this way it's fully under test (including against hand-computed reference values). Raw BM25 scores are unbounded, so they're min-max normalized to [0, 1] before fusing with cosine similarity, keeping `vector_weight`/`keyword_weight` meaningful.
- `search()` used to build a result dict for every chunk in the corpus on every query, then sort all of them and discard everything past `top_k`. It now ranks the score array first and materializes only the chunks it returns. The linear scan is unchanged - that's Phase 2 - but the per-query allocation no longer scales with corpus size.
- `top_k` had no lower bound. `top_k or self.top_k` turned an explicit `0` into the server default, and a negative value sliced from the end of the ranking. `top_k` is now bounded on the request model (1..`max_top_k`) and validated in `Retriever.search()` for direct library callers.
- `chunk_text` looped forever if `chunk_overlap` reached `chunk_size`, since the window's stride is `chunk_size - overlap`. Both are config values, so a plausible misconfiguration hung the process and filled memory. Config validation now rejects it at load, naming the setting.
- `chunk_docs()` stamped a generated `id` onto the caller's own doc dicts as a side effect. It no longer mutates its input.
- `compute_corpus_hash()` hashed content only, so renaming a markdown file left the cache valid and every chunk kept reporting its old `source`. Source is now part of the hash.

## Phase 1 - HTML to JSON ingestion (the actual differentiator)

`POST /api/v1/ingest` takes `{"url": ...}` (Scout fetches it) or `{"html": ..., "source_url": ...}` (caller already has the HTML), returns structured JSON immediately, and indexes it so it's searchable via `/api/v1/search` without a rebuild step.

- **Fetcher** (`scout/fetch.py`): httpx-based, manual redirect handling with each hop re-validated, 10s timeout, streamed 5MB response cap, SSRF guard that resolves the hostname and rejects private/loopback/link-local/reserved addresses before connecting.
- **Extractor** (`scout/webextract.py`): delegates to `trafilatura` for boilerplate removal, asks for markdown output plus title/author/date/sitename metadata.
- **Normalizer** (`scout/ingest.py::doc_from_web_content`): wraps extracted content in the same doc shape `load_markdown_docs()` produces, so it flows through `chunk_docs()` unchanged.
- **Orchestration** (`scout/web.py`): ties fetch, extract, and chunk together as `ingest_url()` / `ingest_html()`.
- **Retriever.add_chunks()**: embeds new chunks, drops any existing ingested chunks with a matching source, and persists the result, so re-ingesting a URL updates it in place instead of duplicating it.

Open before this is production-ready:

- SSRF guard checks at DNS resolution time, not connection time. A DNS-rebinding attack (host resolves safely during the check, then re-resolves to a private IP for the actual connection) isn't covered. Low risk while self-hosted, should be closed before any multi-tenant exposure.
- JS-rendered pages return nothing, since `trafilatura` only sees the HTML as served. A headless-browser fallback is a deliberate future decision, not a default, given the sandboxing it needs.
- `schema_version` is a single string on the response envelope, not a real compatibility contract: nothing validates it, negotiates it, or documents what changes between versions. Fine as a foundation, not sufficient on its own.

Closed since the first pass:

- Content-type validation now runs in `scout/fetch.py` before extraction, so a non-HTML response (a PDF, a JSON error page served with a 200) fails fast with a specific error instead of a generic `ValueError` further down the pipeline.
- Ingested pages survive a restart, via `scout/index.py::save_ingested` / `load_ingested()`.
- Re-ingesting the same URL replaces its old chunks instead of piling up duplicates. `Retriever` tracks the docs_path corpus and the ingested corpus as two separate layers (`_docs_*` / `_ingested_*`) concatenated into the live `self.corpus`. Only covers exact same-URL re-ingestion, not near-duplicate content from different URLs (Phase 2).
- "No structure beyond markdown text" turned out to be only half true. `trafilatura` already preserves tables and fenced code blocks by default; only link URLs were being dropped, because `webextract.py` passed `include_links=False`. Flipped to `True`. `include_images=False` stays a deliberate choice: an image URL alone isn't useful evidence for a text-based agent.
- `Evidence`/`SearchResponse`/`IngestResponse` carry a `schema_version` field (`scout/models.py`, currently `"1.0"`). See the versioning caveat above for what this doesn't yet do.
- `scout/api.py` used to build its `Retriever` as a module-level singleton at import time, which meant merely importing the module (as any test file must) loaded the embedding model and touched disk. Refactored into a FastAPI lifespan handler, with `tests/test_api.py` exercising the actual HTTP layer.
- **The response size cap didn't cap anything.** It compared `len(response.content)` against the limit - by which point the entire body was already in memory - and otherwise trusted `Content-Length`, which a hostile server can simply omit. `_read_capped_body()` now streams the response and stops reading at the limit. Covered by a test that serves an endless body and asserts the reader stops near the cap rather than at the sender's discretion.
- Exceeding the redirect budget, and a redirect with no `Location` header, now raise errors that name what happened instead of surfacing as whatever httpx said next.
- **Scout couldn't be deployed as the ingest-only service this phase describes.** `Retriever.__init__` raised if `docs_path` held no markdown, so hosting it meant shipping a demo corpus alongside it. An empty docs corpus is now a startup warning; only ingested pages are searchable, and `/health` reports `corpus_size: 0`.

## Phase 2 - storage that scales past a demo corpus

- Swap the flat `numpy` array and JSON metadata cache for a real vector store. `qdrant-client` sat in requirements.txt early in the project's history, which suggests it was the original intent. Either bring it back deliberately (e.g. via `docker-compose` for local dev) or document a different choice.
- True incremental indexing. The docs_path corpus still forces a full re-embed on any change, and every ingest rewrites the entire ingested store to disk, rebuilds the combined in-memory corpus, and rebuilds the BM25 index from scratch, even though only one source's worth of chunks actually changed. `/api/v1/ingest/batch` amortizes that rewrite across a batch (Phase 3) rather than fixing it: the cost is still O(everything ingested so far) per call.
- Near-duplicate detection across different URLs. Exact same-URL re-ingestion is handled (Phase 1); two different URLs serving the same or near-identical content still produce separate chunks.
- Retrieval is still a linear scan over every embedding. Ranking no longer allocates per corpus chunk, but the scan itself is what an ANN index in a real vector store would replace.

Closed since first written:

- Real BM25 keyword scoring, tracked above under Phase 0 (it replaced Phase 0's token-overlap scoring, so that's where the detail lives) rather than pulling in `whoosh` as originally floated here.
- **Writes are now crash-safe.** Both stores were written with bare `open()` calls whose handles were never closed and whose encoding defaulted to whatever the host happened to use. A crash or restart mid-write left a truncated file that the next startup would choke on. Every write now goes through a temp file, `fsync`, and an atomic `os.replace`, with UTF-8 declared explicitly. `pytest` treats `ResourceWarning` as an error, so a reintroduced leaked handle fails the suite.
- **The two stores now fail differently, on purpose.** The docs cache is rebuildable, so corruption there degrades to a cache miss and a re-embed. The ingested store is the only copy of every page ever ingested, so corruption there raises `IndexCorruptError` rather than starting up empty and letting the next ingest overwrite still-recoverable data. A store whose embedding count and chunk count disagree is rejected the same way, since it would otherwise keep working while attributing every score to the wrong chunk.
- **Cache validity is checked on more than the hash.** Changing `SCOUT_VECTOR_MODEL` used to leave a cache built by the previous model; stacking those vectors against freshly embedded ones would either raise inside numpy or silently score against the wrong space. Row count and vector width are now checked alongside the hash.

## Phase 3 - API and agent-facing ergonomics

Open:

- **The rate limiter is per-process, so it doesn't survive horizontal scaling.** Running four uvicorn workers means four buckets and four times the configured limit, and it resets on restart. A shared counter (Redis) is the fix, and it belongs with the Phase 2 storage decision rather than as a second, separate piece of infrastructure.
- **Keys are static.** They're read once at startup from `SCOUT_API_KEYS`, so rotating one means a restart, and there's no per-key scoping (a key that may search but not ingest) or revocation short of redeploying. Right shape for a single-operator service; not enough to hand keys to third parties.
- Unauthenticated deployments bucket the rate limit by client IP, which behind a proxy is one bucket for everyone. `X-Forwarded-For` is deliberately not trusted, since anyone can send it; a deployment that terminates TLS at a proxy it controls needs to pass the real address some other way.
- `schema_version` still isn't negotiated. The client now warns on a major-version mismatch (`scout/client.py`), which is a check, not a contract: nothing lets a caller *request* a version, and there's no documented list of what changes between them.

Closed since first written:

- **Auth exists, and it was the main blocker to public hosting.** `scout/auth.py` gates every `/api/v1` route behind an `X-API-Key` header matched against `SCOUT_API_KEYS`. Keys are compared as bytes in constant time against every configured key (returning on the first match would leak key ordering to anyone timing it; comparing as `str` raises `TypeError` on non-ASCII input, which is caller-controlled and would have been a 500 rather than a 401). The key itself never reaches a log line or the rate limiter: callers are identified by a 12-character hash of it. With no keys set there's no auth at all, which is correct for a laptop and wrong for anything reachable, so startup logs which of the two it is and `/health` reports it. `/` and `/health` stay open so a liveness probe doesn't need a credential.
- **Ingest is rate limited per caller** (`scout/ratelimit.py`): a refilling token bucket, keyed on the API key when auth is on and the client IP when it isn't. A token bucket rather than a fixed window because a window boundary lets a caller send 2x the limit across it, and because a continuously refilling bucket can answer `Retry-After` exactly instead of "some time before the window flips". A rejected request spends nothing, so a client retrying in a loop doesn't push its own deadline out. Idle buckets are evicted, since the alternative is an unbounded map of every caller identity ever seen - for an unauthenticated deployment, every client IP.
- **Batch endpoints for fan-out.** `/api/v1/search/batch` answers every query in a batch against one corpus snapshot and embeds the whole batch in a single pass through the model, so results within a batch stay mutually consistent even if an ingest lands mid-batch. `/api/v1/ingest/batch` isolates per-item failure - the two exception types the single-page endpoint turns into 502 and 422 become per-item `status` values, so an agent crawling ten links gets the eight that worked - and indexes every successful page in one `add_chunks()` call, which amortizes the whole-store rewrite noted under Phase 2 across the batch instead of paying it per page. A batch costs the rate limiter one unit per item; charging it as one request would have made `/ingest/batch` a way to buy N outbound fetches for the price of one.
- **A Python client** (`scout/client.py`) with `search`/`search_batch`/`ingest_url`/`ingest_html`/`ingest_batch`, an injectable `httpx.Client` (which is what makes it testable against the real ASGI app, and the hook a caller needs for a proxy or custom transport), and `ScoutAPIError` carrying `status_code`, `detail` and `retry_after` separately so a caller can branch on them. It returns plain dicts rather than parsing into the server's own Pydantic models: doing that would mean a client one version behind rejects a response carrying a field it hasn't heard of, which is the exact coupling `schema_version` exists to avoid.

- Request-shape limits are declared on the Pydantic models rather than checked inside endpoint bodies, so `top_k` bounds, query length and payload size appear in `/openapi.json`. An agent can read the limits instead of discovering them through a 422. A test asserts the bounds are actually published.
- **The retriever wasn't safe under the concurrency FastAPI actually gives it.** Sync endpoints run in a threadpool, so a search can land mid-ingest, and `_rebuild_combined_corpus()` reassigned the corpus, the embeddings and the BM25 index one at a time while rebuilding BM25 in place. A search landing in that window scored chunks against mismatched embeddings. The three structures are now built off to the side and swapped as a unit under a lock held only for the handover, with embedding and disk I/O outside it. `tests/test_concurrency.py` covers it; both tests there were checked against a deliberately broken retriever and fail on it.
- Configuration is environment-overridable (`SCOUT_*`), so a deployment can be retargeted without editing files inside the image, and settings that would break the pipeline are rejected at load.
- Logging replaced `print()`, with Scout's loggers at the configured level and dependencies pinned to WARNING - otherwise turning Scout up to INFO also turns on sentence-transformers and httpx chatter, which buried Scout's own output.

## Phase 4 - open source readiness

- CONTRIBUTING.md and issue/PR templates.
- CI (GitHub Actions) running `pytest` on push/PR. The suite no longer touches the network, but it still needs the `all-MiniLM-L6-v2` weights available locally or cached; CI needs either a cached-model step or a lightweight fake embedding backend for fast, hermetic runs.
- Versioned releases and a changelog once the API surface stabilizes.

## Security notes

Fetching arbitrary URLs on a caller's behalf is the main attack surface Scout takes on.

- SSRF: internal/private IP ranges and cloud metadata endpoints (`169.254.169.254`, etc.) are blocked, including their IPv4-mapped IPv6 forms, and every redirect hop is re-validated rather than trusting where a server points next. DNS rebinding remains open (see Phase 1).
- Resource limits: response size and fetch timeout are capped so a malicious or huge page can't exhaust memory or CPU. The size cap is enforced while streaming, not after the body is in memory.
- Content sanitization: fetched HTML/JS is never executed or rendered. Extraction stays text-only parsing; a headless browser is a deliberate, sandboxed later addition, not a default.
- Authentication: an `X-API-Key` header checked in constant time, covering every `/api/v1` route (Phase 3). Off by default, and startup says so out loud rather than leaving an operator to infer it.
- Per-caller rate limiting on ingest, charged per page so a batch can't buy N fetches for the price of one (Phase 3). Per-process, so it doesn't hold across multiple workers.
- Not yet covered: key rotation without a restart, per-key scopes, and a rate limit shared across workers.
