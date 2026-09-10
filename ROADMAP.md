# Scout Roadmap

Ordered by dependency, not strictly by priority. Phase 1 is the actual point of the project.

Phases 0, 1 and 3 are done. Phase 2's storage work is done and only near-duplicate detection is left in it. Phase 7 has its headline item, the MCP server. Phases 5 and 6 are what remains of the finish line: hosted somewhere reachable, and retrieval quality demonstrated rather than asserted. Phase 4 sits behind them deliberately - being usable comes before being contributable to.

Where a decision here was made against measurements rather than intuition, the measurements live in [docs/architecture/](docs/architecture/) and can be reproduced with `scripts/bench_index.py`. How each part actually works is in [docs/design/](docs/design/).

## Phase 0 - local search prototype (done)

A hybrid vector + keyword retriever over a local markdown corpus, a FastAPI search endpoint, a CLI to build/refresh the index. Exists mainly to prove out retrieval and scoring before ingestion. Known limitations, carried into later phases:

- In-memory linear scan over `numpy` arrays. Written here as "won't hold up past a few thousand chunks", which measuring later showed to be wrong: it's 11ms at 20k chunks, in a matmul that releases the GIL, and it was the *cheapest* thing on the list. Kept as a limitation because it is still linear, but see Phase 2 for where the cost actually was.
- Score fusion weights (`vector_weight`/`keyword_weight`) are guessed, not tuned against a labeled relevance set. Phase 6.
- Fixed-size word chunking (`chunk_text`) ignores document structure and can split mid-section. Chunking by heading would preserve more meaning per chunk. Phase 6.

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

This phase was originally written as "swap the flat `numpy` array and JSON metadata cache for a real vector store", on the strength of `qdrant-client` having sat in requirements.txt early in the project's history. Measuring first (`scripts/bench_index.py`) said that was the wrong target: at 20k chunks the vector scan a vector store would replace costs 11ms per query, while rebuilding the keyword index - which every single ingest paid - cost 3,441ms. The decision and its evidence are recorded in [docs/architecture/adr-001-incremental-index.md](docs/architecture/adr-001-incremental-index.md). The short version is that the index becomes incremental in-process, a vector database is deferred behind a seam rather than adopted, and the revisit trigger is written down.

Open:

- **The docs_path corpus still forces a full re-embed on any change.** Unchanged by the above: it is a hash-invalidated cache of a fixed folder, so a single edited markdown file re-embeds all of it.
- Near-duplicate detection across different URLs. Exact same-URL re-ingestion is handled (Phase 1); two different URLs serving the same or near-identical content still produce separate chunks. Content hashing catches the exact case cheaply; near-duplicates need an embedding-similarity threshold at ingest time.
- Retrieval is still a linear scan over every embedding, and deliberately so for now - 11ms at 20k chunks, in a numpy matmul that releases the GIL. This is the number that has to grow before a vector store earns its operational cost.
- **`/api/v1/search/batch` holds the index lock for the whole batch.** Keyword scoring for every query in a batch happens in one locked section, so hold time scales with batch length (up to `max_batch_queries`, default 10) while a single search does not. That endpoint is deliberately not rate limited, since it is local CPU work, so concurrent batch callers can delay an ingest more than a single search would. Scoring per query would shorten the hold but would give up the guarantee that every query in a batch sees one corpus snapshot, which is why the endpoint exists; the real fix is a reader-writer lock, and it belongs with the Phase 5 hosting work where concurrency actually gets exercised.
- **Building the keyword index cold is now more expensive, not less.** An inverted index writes one entry per (document, term) pair into thousands of posting lists where the old structure incremented one flat counter. That is a deliberate trade - it is paid once at startup instead of on every ingest - but it means a 20k-chunk ingested corpus spends about 3.4s on startup rebuilding the keyword index. Persisting the postings alongside the segments would remove it. The segmented store landed without doing so, deliberately: postings would have to be invalidated and rewritten on every merge, and startup is the one cost a long-running service pays least often.
- **A segmented store write is flat in chunks but linear in distinct live sources.** The manifest carries one entry per source and is rewritten on every ingest: 21ms at 20k chunks across 2,500 sources, against 725ms for the full rewrite it replaced. A far shallower slope, and small enough not to be the constraint, but not flat. It is the number to watch if the ADR-001 revisit trigger is ever approached.

Closed since first written:

- **The keyword index is incremental.** `scout/bm25.py` was a list of per-document term counters that could only be built, never updated, so `Retriever` threw the whole thing away and re-tokenized the entire corpus on every ingest. It is now an inverted index (`term -> {slot: term frequency}`) with `add_documents`, `remove_slots` and `compact`. An ingest costs the terms in the page being ingested rather than the size of the corpus: 1-2ms at any corpus size, against 128ms/794ms/3,441ms at 1k/5k/20k chunks before. Queries got faster too, since scoring now walks posting lists instead of every document: 3.3ms rather than 39.4ms at 20k chunks.
- **Removal is by tombstone, and slots are the reason.** Dropping a document from the middle of the index would shift the position of every document after it, which would silently re-point the retriever's whole mapping at the wrong chunks. Removed documents keep their slot and score 0.0; `Retriever._bm25_slots` maps corpus position to slot, and `compact()` reclaims tombstones once they pass a threshold, renumbering without re-tokenizing anything. The tests for this were checked against five deliberately broken versions (statistics not updated on add, live count not decremented on remove, slots deleted rather than tombstoned, postings not renumbered on compact, postings not popped on remove) and each one is caught.
- **The retriever's concurrency model changed with it.** The old design kept searches lock-free by building an entire new BM25 index off to the side and swapping it in - that swap was the thing being made cheap, and it is exactly what an incremental index cannot offer, since a search iterating a posting list while an ingest inserts into it raises `RuntimeError`. `_swap_lock` became `_index_lock` and now covers keyword scoring as well as the handover. That is affordable because of the change itself: mutation is proportional to what changed, and BM25 scoring is pure Python, which the GIL already serializes across threads. Query embedding, the numpy similarity scan and disk writes all stay outside the lock. Copy-on-write was the alternative and was rejected: copying the postings is O(total postings), which is the cost the change exists to remove.
- **There is a benchmark, so these numbers can be re-checked rather than believed.** `scripts/bench_index.py`. It draws words from a Zipf distribution rather than uniformly, because uniform draws produce chunks with about 390 distinct terms where real prose has about 237, which inflates indexing cost and deflates query cost at the same time. An earlier uniform version of this benchmark pointed at a different conclusion.

- **The ingested store appends instead of rewriting.** Every ingest used to rewrite every page ever ingested: 238ms at 20k chunks, growing linearly, and the last O(everything) step on the ingest path once the keyword index became incremental. `scout/store/segmented.py` writes one immutable segment per ingest plus a manifest entry recording which sources it supersedes, and merges the live set back into one segment once dead weight or segment count passes a threshold. Same benchmark run: 22ms/62ms/725ms for the full rewrite at 1k/5k/20k chunks, against 3.9ms/3.9ms/21ms segmented.

  Crash safety comes from the write ordering rather than from a lock. Segments land first and are inert until the manifest names them, so a crash leaves either the old manifest and an unreferenced segment - cleaned up on the next load - or the new one. There is no window where a partially written store reads as a complete one. A pre-segment store is imported automatically on first start, since an upgrade that silently began empty would let the next ingest supersede the only copy of every page already there.

- **`Retriever` talks to the storage interface, not to `scout.index`.** The `ChunkStore` seam was scaffolded and unused; it is now the only path to the ingested store, selected by the `chunk_store` config value. `JsonChunkStore` stays as the reference implementation and the rollback path, and `tests/test_store.py` runs the whole contract against both backends, so a case that passes on the old one and fails on the new one is a regression rather than a new expectation.

- Real BM25 keyword scoring, tracked above under Phase 0 (it replaced Phase 0's token-overlap scoring, so that's where the detail lives) rather than pulling in `whoosh` as originally floated here.
- **Writes are now crash-safe.** Both stores were written with bare `open()` calls whose handles were never closed and whose encoding defaulted to whatever the host happened to use. A crash or restart mid-write left a truncated file that the next startup would choke on. Every write now goes through a temp file, `fsync`, and an atomic `os.replace`, with UTF-8 declared explicitly. `pytest` treats `ResourceWarning` as an error, so a reintroduced leaked handle fails the suite.
- **The two stores now fail differently, on purpose.** The docs cache is rebuildable, so corruption there degrades to a cache miss and a re-embed. The ingested store is the only copy of every page ever ingested, so corruption there raises `IndexCorruptError` rather than starting up empty and letting the next ingest overwrite still-recoverable data. A store whose embedding count and chunk count disagree is rejected the same way, since it would otherwise keep working while attributing every score to the wrong chunk.
- **Cache validity is checked on more than the hash.** Changing `SCOUT_VECTOR_MODEL` used to leave a cache built by the previous model; stacking those vectors against freshly embedded ones would either raise inside numpy or silently score against the wrong space. Row count and vector width are now checked alongside the hash.

## Phase 3 - API and agent-facing ergonomics

Open. Everything left in this phase turned out to be scale-shaped rather than feature-shaped, so it has moved to the phase that owns that scale:

- The per-process rate limiter, static keys (no rotation without a restart, no per-key scopes), and untrusted `X-Forwarded-For` behind a proxy are all Phase 5. They are the difference between running locally and running somewhere reachable, not gaps in the API surface itself.
- Negotiated `schema_version` is Phase 7, alongside the integrations that make version skew something a caller actually experiences.

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

Deprioritized relative to Phases 5-7. Scout being usable matters more right now than Scout being contributable to, and a stable API surface is a prerequisite for versioning it anyway.

- CONTRIBUTING.md and issue/PR templates.
- CI (GitHub Actions) running `pytest` on push/PR. The suite no longer touches the network, but it still needs the `all-MiniLM-L6-v2` weights available locally or cached; CI needs either a cached-model step or a lightweight fake embedding backend for fast, hermetic runs. The fake backend is worth having regardless: the suite spends most of its wall time in the embedding model.
- Versioned releases and a changelog once the API surface stabilizes.

## Phase 5 - actually hosted

Everything here is a gap between "runs on a laptop" and "runs somewhere with a URL".

- A Dockerfile and a documented deploy target. Scout is already configurable entirely through `SCOUT_*` environment variables and stores all durable state under one directory, so this is packaging rather than redesign.
- **Key rotation without a restart.** Keys are read once at startup from `SCOUT_API_KEYS` (Phase 3), so rotating one means a redeploy.
- **Per-key scopes.** A key that may search but not ingest is the obvious first split, since ingest is the endpoint that spends Scout's network on hosts the caller chooses.
- **A rate limit shared across workers.** The token bucket is in-process, so N uvicorn workers means N times the configured limit. This wants the same external store a vector database would, which is why ADR-001 records that the two decisions should land together rather than adding two pieces of infrastructure separately.
- **Real client addresses behind a proxy.** Unauthenticated deployments bucket the rate limit by client IP, which behind a proxy is one bucket for everyone. `X-Forwarded-For` is deliberately not trusted, since anyone can send it; a deployment terminating TLS at a proxy it controls needs a configured trusted-proxy list rather than blanket trust.

## Phase 6 - retrieval quality, demonstrated

Scout is fast and returns well-shaped evidence. Nothing so far shows the evidence is *good*, and two of the settings that decide it were picked by hand.

- A labeled relevance set: queries paired with the chunks that should answer them. Small and honest beats large and synthetic.
- An eval harness reporting recall@k and MRR, runnable from the CLI, so a change to chunking or fusion has a number attached rather than an opinion.
- Tune `vector_weight`/`keyword_weight` against that set. They are currently 0.65/0.35 because those looked reasonable.
- Heading-aware chunking to replace the fixed word window (Phase 0), measured on the same harness rather than assumed to be better.

## Phase 7 - agent-native integration

The pitch is search built for agents, and the MCP server is what makes it true without anyone wiring it up by hand.

Open:

- A LangChain / LlamaIndex retriever adapter, for frameworks that expect their own interface.
- **Search restricted to named sources.** "What does this page say about X" is the shape an agent reaches for after ingesting, and today it can only search the whole corpus and hope the right page ranks. A `sources` filter on `search` would also make a `read(url, query)` MCP tool a few lines rather than a new subsystem.
- **A structured-output contract shared with the HTTP API.** The MCP tools declare their response shape as pydantic models (`Passage`, `IngestedPage`), which is a second set of models alongside `scout/models.py`. They agree today because both are small; nothing enforces it.
- **Negotiated `schema_version`.** Carried over from Phase 3: the field exists on every response and the Python client warns on a major-version mismatch, but nothing lets a caller *request* a version and there is no documented list of what changes between them. That is a check, not a contract, and shipping integrations is the point at which it starts to matter.

Closed since first written:

- **Scout is an MCP server.** `scout/mcp.py`, installed as `scout-mcp` from the `[mcp]` extra, exposes `search` and `ingest` to any MCP-capable agent. It runs Scout in-process by default, so an agent gets web ingestion and search with nothing to deploy, and `--url` points it at a shared Scout instead; both backends return the same shape, so the tools never branch on which is behind them.

  The design decision the server turns on is that **`ingest` returns a summary, not the page**. Echoing the body back would cost exactly the tokens an agent came to save, making it `fetch` with extra steps. The response is bounded regardless of page size - title, chunk and word counts, metadata, and a 40-word preview so the agent can tell it got the page it meant - and the body comes back through `search`, a few hundred tokens at a time. `search` drops the scoring breakdown the HTTP API returns for the same reason.

  Both tools publish an output schema, because they return pydantic models rather than bare dicts. An agent reading the tool listing sees the response shape rather than parsing JSON out of a text block, which is the thing Scout exists to stop agents doing. See [docs/design/mcp.md](docs/design/mcp.md).

## Security notes

Fetching arbitrary URLs on a caller's behalf is the main attack surface Scout takes on.

- SSRF: internal/private IP ranges and cloud metadata endpoints (`169.254.169.254`, etc.) are blocked, including their IPv4-mapped IPv6 forms, and every redirect hop is re-validated rather than trusting where a server points next. DNS rebinding remains open (see Phase 1).
- Resource limits: response size and fetch timeout are capped so a malicious or huge page can't exhaust memory or CPU. The size cap is enforced while streaming, not after the body is in memory.
- Content sanitization: fetched HTML/JS is never executed or rendered. Extraction stays text-only parsing; a headless browser is a deliberate, sandboxed later addition, not a default.
- Authentication: an `X-API-Key` header checked in constant time, covering every `/api/v1` route (Phase 3). Off by default, and startup says so out loud rather than leaving an operator to infer it.
- Per-caller rate limiting on ingest, charged per page so a batch can't buy N fetches for the price of one (Phase 3). Per-process, so it doesn't hold across multiple workers.
- Not yet covered: key rotation without a restart, per-key scopes, and a rate limit shared across workers. All three are Phase 5.
