# Scout Roadmap

This captures where Scout is and what needs to be built next. It's ordered
roughly by dependency — later phases build on earlier ones — not strictly by
priority, though Phase 1 is the actual point of the project.

## Phase 0 — Local search prototype (done)

A working hybrid vector+keyword retriever over a local markdown corpus, with
a FastAPI search endpoint and a CLI to build/refresh the index. This exists
mainly to prove out the retrieval and scoring logic before the harder problem
(ingestion) is tackled. Current known limitations, carried into later phases:

- In-memory linear scan over `numpy` arrays — fine for a demo corpus, won't
  hold up past a few thousand chunks.
- Keyword scoring is raw token-overlap (`scout/keyword.py`), not BM25/TF-IDF
  — cheap but weak; a `whoosh`-backed inverted index would score plausibility
  and rare-term weighting much better.
- Score fusion weights (`vector_weight`/`keyword_weight` in `config.json`)
  are guessed, not tuned against any labeled relevance set.
- Fixed-size word chunking (`chunk_text`) ignores document structure (it can
  split mid-section); chunking by heading/section would preserve more
  meaning per chunk.

## Phase 1 — HTML → JSON ingestion (the actual differentiator)

This is the missing piece that makes Scout what it's meant to be, rather
than a generic RAG demo. Concretely:

1. **Fetcher**: given a URL, retrieve the page (`httpx`/`requests`), with
   sane timeouts, redirect handling, and a User-Agent that identifies Scout.
2. **Extractor**: turn raw HTML into clean structured content — title,
   headings/section tree, body text, code blocks, tables, and link/image
   references — stripping nav/ads/boilerplate. Libraries worth evaluating:
   `trafilatura`, `readability-lxml`, or a hand-rolled `BeautifulSoup` pass
   if more control over structure is needed.
3. **Normalizer**: map extracted content into the same evidence-chunk shape
   `scout/ingest.py` already produces (`id`, `content`, `source`, `type`),
   so it flows through the existing chunk → embed → index pipeline
   unchanged. This is also where a document `type` beyond `"documentation"`
   (`api_reference`, `changelog`, `forum_post`, etc.) would get inferred.
4. **New API surface**: a `POST /api/v1/ingest` endpoint that takes a URL (or
   raw HTML) and returns the structured JSON directly — this is the "skip
   the conversion step" path, distinct from `/search`, which queries
   previously-indexed content. Whether ingest is synchronous or
   queues a background job depends on how heavy extraction+embedding turns
   out to be; start synchronous, revisit if latency is bad.
5. **Evidence schema as a real contract**: `scout/models.py` should become
   the versioned, documented schema agents build against — right now it's
   just shaped to match whatever `search()` happens to return.

## Phase 2 — Storage that scales past a demo corpus

- Swap the flat `numpy` array + JSON metadata cache for a real vector store.
  `qdrant-client` is already sitting in git history as a dependency, which
  suggests this was the original intent — either bring it back deliberately
  (e.g. via `docker-compose` for local dev) or drop the assumption and
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
