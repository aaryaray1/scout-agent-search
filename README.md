# Scout

Search built for agents, not browsers.

## Why

An agent that pulls information from the web gets raw HTML back and has to parse it into something usable before it can reason over it. Every agent repeats that step, on every fetch. Scout does it once and returns structured JSON instead of markup.

## What's here

- Local hybrid search (vector + BM25 keyword) over a markdown corpus: `scout/ingest.py`, `scout/embeddings.py`, `scout/bm25.py`, `scout/index.py`, `scout/search.py`. Served as `POST /api/v1/search`.
- HTML ingestion: fetch a URL or accept raw HTML, extract clean content with `trafilatura`, chunk it, index it. `scout/fetch.py`, `scout/webextract.py`, `scout/web.py`. Served as `POST /api/v1/ingest`. Ingested pages persist to disk and survive a restart, and re-ingesting the same URL replaces its content instead of duplicating it.

Ingestion is a first pass, not a finished pipeline. See [ROADMAP.md](ROADMAP.md) for what's missing (a real vector store past demo scale, JS-rendered pages, auth).

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
| `top_k` | `SCOUT_TOP_K` | `3` | default result count when a request doesn't specify one |
| `vector_weight` / `keyword_weight` | `SCOUT_VECTOR_WEIGHT` / `SCOUT_KEYWORD_WEIGHT` | `0.65` / `0.35` | score-fusion balance |
| `chunk_size` / `chunk_overlap` | `SCOUT_CHUNK_SIZE` / `SCOUT_CHUNK_OVERLAP` | `400` / `50` | chunking window, in words |
| `max_top_k` | `SCOUT_MAX_TOP_K` | `50` | ceiling on `top_k` a caller may request |
| `max_query_chars` | `SCOUT_MAX_QUERY_CHARS` | `2000` | ceiling on query length |
| `max_html_bytes` | `SCOUT_MAX_HTML_BYTES` | `5242880` | ceiling on an inline `html` payload |
| `log_level` | `SCOUT_LOG_LEVEL` | `INFO` | level for Scout's own loggers (dependencies stay at WARNING) |

Settings that would break the pipeline are rejected at load rather than failing somewhere less obvious. A `chunk_overlap` at or above `chunk_size`, for instance, leaves the chunker with a stride of zero and would loop forever, so it raises at startup naming the offending setting.

## Deployment

**Ingest-only.** Scout starts with no local markdown corpus at all: point `SCOUT_DOCS_PATH` at an empty folder and every document arrives through `/api/v1/ingest`. Startup logs a warning and `/health` reports `corpus_size: 0`.

```bash
SCOUT_DOCS_PATH=/var/empty SCOUT_INDEX_DIR=/data/scout \
  uvicorn scout.api:app --host 0.0.0.0 --port 8000
```

**State.** Everything durable lives under `index_dir`: the docs-corpus embedding cache (rebuildable) and the ingested-page store (not). Mount it on a volume. The two are treated differently on corruption: a damaged docs cache degrades to a cache miss and re-embeds, while a damaged ingested store raises rather than starting up empty and letting the next ingest overwrite recoverable data. Every write goes through a temp file and an atomic rename, so a crash mid-write leaves the previous good file in place.

**Concurrency.** The endpoints are sync, so Starlette runs them in its threadpool and a search can land mid-ingest. `Retriever` swaps its corpus, embeddings and BM25 index as a single unit under a lock held only for the handover; embedding and disk writes happen outside it. See `tests/test_concurrency.py`.

**Limits.** Request-shape caps are declared on the Pydantic models, so they appear in `/openapi.json` and an agent can discover them instead of finding them via a 422. Outbound fetches are capped by streaming the response and stopping at the byte limit, rather than trusting `Content-Length`. Auth and rate limiting are **not** implemented, see ROADMAP Phase 3 before exposing this beyond a trusted network.

## Tests

```bash
pytest
```

106 tests, no network access required. The suite covers the SSRF/redirect/size guards against a mock transport, storage corruption and atomicity, config layering, concurrent search-during-ingest, and the HTTP layer end to end.

Complexity is tracked with `radon`:

```bash
radon cc -s -n B scout/     # anything scoring worse than A
```

Nothing in `scout/` currently exceeds CC 6, against a refactor-now threshold of 11.

## Layout

```
scout/          package: ingest, embed, bm25, index, search, fetch, webextract, web, api, cli, config, logsetup
data/docs/      markdown source documents (demo corpus)
data/index/     generated embedding cache + persisted ingested pages, git-ignored
tests/          pytest suite
```

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Roadmap

See [ROADMAP.md](ROADMAP.md).
