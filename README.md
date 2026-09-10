# Scout

Search built for agents, not browsers.

## Why

An agent that pulls information from the web gets raw HTML back and has to parse it into something usable before it can reason over it. Every agent repeats that step, on every fetch. Scout does it once and returns structured JSON instead of markup.

Reading a page costs an agent the whole page. Scout indexes it and hands back only the passages that answer the question, so ten pages cost roughly what one used to.

## Use it from an agent

Scout ships as an MCP server, so an agent gets web ingestion and search as tools with no service to deploy:

```bash
pip install "scout-agent-search[mcp]"
```

```json
{"mcpServers": {"scout": {"command": "scout-mcp"}}}
```

That gives the agent two tools:

- **`ingest(urls)`** fetches pages, strips the boilerplate, indexes them, and returns a short summary per page rather than the page body. The summary is the same size whether the page was 300 words or 30,000.
- **`search(query, top_k)`** returns only the passages that answer the question, from everything ingested so far, including in earlier sessions.

Point it at a shared Scout instead of running one in-process with `scout-mcp --url https://scout.example.com`. Both are the same two tools over the same response shape. [docs/design/mcp.md](docs/design/mcp.md) covers the design.

## What's here

- Local hybrid search (vector + BM25 keyword) over a markdown corpus: `scout/ingest.py`, `scout/embeddings.py`, `scout/bm25.py`, `scout/index.py`, `scout/search.py`. Served as `POST /api/v1/search`.
- HTML ingestion: fetch a URL or accept raw HTML, extract clean content with `trafilatura`, chunk it, index it. `scout/fetch.py`, `scout/webextract.py`, `scout/web.py`. Served as `POST /api/v1/ingest`. Ingested pages persist to disk and survive a restart, and re-ingesting the same URL replaces its content instead of duplicating it.
- Batch endpoints for agents that fan out: `POST /api/v1/search/batch` and `POST /api/v1/ingest/batch`. A batch of queries is answered against one corpus snapshot; a batch of pages is indexed in one write, and a page that fails fails alone.
- API-key auth (`scout/auth.py`) and per-caller ingest rate limiting (`scout/ratelimit.py`) on the `/api/v1` surface.
- An MCP server exposing search and ingest as agent tools, in-process or against a running Scout: `scout/mcp.py`.
- A Python client so agent frameworks don't hand-roll HTTP: `scout/client.py`.

Neither half of an ingest costs the size of the corpus any more. The keyword index is incremental, so an ingest updates the page that changed rather than re-tokenizing everything, and the durable store appends one immutable segment per ingest instead of rewriting every page ever ingested. Measurements and the reasoning: [docs/architecture/adr-001-incremental-index.md](docs/architecture/adr-001-incremental-index.md) and [docs/design/storage.md](docs/design/storage.md).

Ingestion is a first pass, not a finished pipeline. See [ROADMAP.md](ROADMAP.md) for what's missing (JS-rendered pages, near-duplicate detection, a rate limit shared across workers).

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"

scout-ingest                  # build the index from data/docs
uvicorn scout.api:app --reload
```

Search:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "rate limit exceeded", "top_k": 2}'
```

```json
{
  "schema_version": "1.0",
  "query": "rate limit exceeded",
  "results": [
    {
      "content": "...",
      "source": "minimax_errors.md",
      "type": "documentation",
      "confidence": 0.575,
      "metadata": { "vector_score": 0.346, "keyword_score": 1.0, "id": "..." }
    }
  ]
}
```

Ingest a live page instead of pre-built docs:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/some/docs/page"}'
```

Returns structured content directly and indexes it for immediate search. Ingested pages persist to disk (`data/index/`) and are reloaded on the next startup; re-ingesting the same URL replaces its content instead of duplicating it.

## Fan-out

An agent working through a page of links, or decomposing a question into sub-questions, shouldn't pay a round trip per item.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ingest/batch \
  -H "Content-Type: application/json" \
  -d '{"items": [{"url": "https://example.com/a"}, {"url": "https://example.com/b"}]}'
```

```json
{
  "schema_version": "1.0",
  "results": [
    { "status": "ok", "url": "https://example.com/a", "page": { "title": "A", "chunks": [] } },
    { "status": "fetch_error", "url": "https://example.com/b", "error": "..." }
  ]
}
```

One bad page fails alone: results come back in request order, each with its own `status` (`ok`, `fetch_error`, `extract_error`), so eight of ten links working returns eight pages rather than one error. Everything that succeeded is indexed in a single write rather than one per page.

`POST /api/v1/search/batch` takes `{"queries": [...]}` and answers every query against the same corpus snapshot, embedding the whole batch in one pass through the model.

## Python client

```python
from scout.client import ScoutClient

with ScoutClient("http://localhost:8000", api_key="...") as scout:
    page = scout.ingest_url("https://example.com/docs/errors")
    hits = scout.search("what does error 429 mean", top_k=3)
    pages = scout.ingest_batch([{"url": u} for u in urls])
```

Responses come back as plain dicts, not the server's Pydantic models: a client that parsed into them would reject a response from a server one schema version ahead, which is exactly what `schema_version` exists to let callers manage. A mismatched major version logs a warning. Errors raise `ScoutAPIError`, which keeps `status_code`, `detail` and (for a 429) `retry_after` separate so a caller can branch on them.

## Configuration

Settings resolve in three layers, each overriding the one before it:

1. built-in defaults (`scout/config.py`)
2. `scout/config.json`
3. `SCOUT_`-prefixed environment variables

The environment layer wins so a deployment can be reconfigured without editing files inside the image.

| Setting | Env var | Default | What it does |
| --- | --- | --- | --- |
| `vector_model` | `SCOUT_VECTOR_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model used for embeddings |
| `docs_path` | `SCOUT_DOCS_PATH` | `data/docs` | folder of markdown documents to index at startup |
| `index_dir` | `SCOUT_INDEX_DIR` | `data/index` | where the embedding cache and ingested store are written |
| `chunk_store` | `SCOUT_CHUNK_STORE` | `segmented` | how ingested pages are persisted; `json` rewrites the whole store per ingest |
| `top_k` | `SCOUT_TOP_K` | `3` | default result count when a request doesn't specify one |
| `vector_weight` / `keyword_weight` | `SCOUT_VECTOR_WEIGHT` / `SCOUT_KEYWORD_WEIGHT` | `0.65` / `0.35` | score-fusion balance |
| `chunk_size` / `chunk_overlap` | `SCOUT_CHUNK_SIZE` / `SCOUT_CHUNK_OVERLAP` | `400` / `50` | chunking window, in words |
| `max_top_k` | `SCOUT_MAX_TOP_K` | `50` | ceiling on `top_k` a caller may request |
| `max_query_chars` | `SCOUT_MAX_QUERY_CHARS` | `2000` | ceiling on query length |
| `max_html_bytes` | `SCOUT_MAX_HTML_BYTES` | `5242880` | ceiling on an inline `html` payload |
| `max_batch_queries` | `SCOUT_MAX_BATCH_QUERIES` | `10` | ceiling on queries in one `/search/batch` |
| `max_batch_ingest` | `SCOUT_MAX_BATCH_INGEST` | `5` | ceiling on pages in one `/ingest/batch` |
| `api_keys` | `SCOUT_API_KEYS` | (empty) | comma-separated API keys; empty means no auth |
| `ingest_rate_limit` | `SCOUT_INGEST_RATE_LIMIT` | `30` | ingests allowed per caller per window; `0` disables |
| `ingest_rate_window` | `SCOUT_INGEST_RATE_WINDOW` | `60` | the rate-limit window, in seconds |
| `log_level` | `SCOUT_LOG_LEVEL` | `INFO` | level for Scout's own loggers (dependencies stay at WARNING) |

Settings that would break the pipeline are rejected at load rather than failing somewhere less obvious. A `chunk_overlap` at or above `chunk_size`, for instance, leaves the chunker with a stride of zero and would loop forever, so it raises at startup naming the offending setting.

## Deployment

**Ingest-only.** Scout starts with no local markdown corpus at all: point `SCOUT_DOCS_PATH` at an empty folder and every document arrives through `/api/v1/ingest`. Startup logs a warning and `/health` reports `corpus_size: 0`.

```bash
SCOUT_DOCS_PATH=/var/empty SCOUT_INDEX_DIR=/data/scout \
  uvicorn scout.api:app --host 0.0.0.0 --port 8000
```

**State.** Everything durable lives under `index_dir`: the docs-corpus embedding cache (rebuildable) and the ingested-page store (not). Mount it on a volume. The two are treated differently on corruption: a damaged docs cache degrades to a cache miss and re-embeds, while a damaged ingested store raises rather than starting up empty and letting the next ingest overwrite recoverable data. Every write goes through a temp file and an atomic rename, so a crash mid-write leaves the previous good file in place.

The ingested store is a run of append-only segments plus a manifest. An ingest writes one new segment and never touches an existing one, so its cost is the page rather than the store; superseded pages are reclaimed by a merge once enough accumulate. Crash safety falls out of the write ordering rather than needing a lock: segments land first and are inert until the manifest names them. An older `data/index/ingested_*.json` store is imported automatically on first start.

**Concurrency.** The endpoints are sync, so Starlette runs them in its threadpool and a search can land mid-ingest. `Retriever` guards the searchable view - corpus, embeddings, slot mapping and keyword index - with one lock, held on both the read and the write side, because the keyword index is updated in place rather than rebuilt and swapped. That is affordable because an update costs the page being ingested rather than the whole corpus, and because keyword scoring is pure Python and already serialized by the GIL. Query embedding, the similarity scan and disk writes all happen outside the lock. See `tests/test_concurrency.py`.

**Auth.** Set `SCOUT_API_KEYS` to one or more comma-separated keys and every `/api/v1` route requires an `X-API-Key` header matching one of them. Keys are compared in constant time and never logged: callers are identified downstream by a short hash of the key they presented.

```bash
SCOUT_API_KEYS="$(openssl rand -hex 24)" uvicorn scout.api:app --host 0.0.0.0
```

With no keys set there is no auth at all, which is right for a laptop and wrong for anything reachable, so startup logs which of the two it is and `/health` reports `"auth": "enabled" | "disabled"`, since "did my keys reach the container" shouldn't only be answerable by getting rejected. `/` and `/health` stay open so a liveness probe doesn't need a credential.

**Rate limiting.** `/api/v1/ingest` and `/api/v1/ingest/batch` are limited per caller (per key, or per client IP when auth is off), since ingest is the endpoint that spends Scout's network on hosts the caller chooses. A batch costs one unit per item, so it can't be used to make N outbound fetches for the price of one. Exceeding it returns 429 with a `Retry-After` header. The bucket is in-process, so N uvicorn workers means N times the configured limit; a shared store is ROADMAP Phase 5 work.

**Limits.** Request-shape caps are declared on the Pydantic models, so they appear in `/openapi.json` (including the batch ceilings and the API-key security scheme) and an agent can discover them instead of finding them via a 422. Outbound fetches are capped by streaming the response and stopping at the byte limit, rather than trusting `Content-Length`.

## Tests

```bash
pytest
```

240 tests, no network access required. The suite covers the SSRF/redirect/size guards against a mock transport, storage corruption and atomicity, config layering, concurrent search-during-ingest, incremental keyword-index updates and their slot mapping, both storage backends against one shared contract, auth and rate limiting, batch fan-out, the Python client against the real ASGI app, the MCP tools through the MCP server itself, and the HTTP layer end to end.

Tests for the guards are written against a deliberately broken version first: an auth check that always matches, a limiter that never evicts, a batch charged as one request, a keyword index that renumbers slots instead of tombstoning them, a store that commits a segment before writing it. A test that still passes against the bug it describes is worse than no test.

Complexity is tracked with `radon`:

```bash
radon cc -s -n B scout/     # anything scoring worse than A
```

Nothing in `scout/` currently exceeds CC 6, against a refactor-now threshold of 11.

Index performance is tracked with a benchmark rather than an assertion:

```bash
python scripts/bench_index.py
```

It reports what an ingest and a query cost at increasing corpus sizes. The numbers quoted in [ROADMAP.md](ROADMAP.md) and the ADRs come from it, so they can be re-checked instead of believed.

## Layout

```
scout/          package: ingest, embed, bm25, index, search, fetch, webextract, web,
                api, auth, ratelimit, client, mcp, cli, config, logsetup
scout/store/    storage backends for ingested pages, behind one interface
data/docs/      markdown source documents (demo corpus)
data/index/     generated embedding cache + persisted ingested pages, git-ignored
docs/design/    how each part works and why
docs/architecture/  decision records, with the measurements behind them
scripts/        benchmarks
tests/          pytest suite
```

Code carries short comments and points at [docs/design/](docs/design/), which is where the reasoning lives.

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Roadmap

See [ROADMAP.md](ROADMAP.md).
