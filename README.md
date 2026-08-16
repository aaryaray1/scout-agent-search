# Scout

Search built for agents, not browsers.

## Why

An agent that pulls information from the web gets raw HTML back and has to parse it into something usable before it can reason over it. Every agent repeats that step, on every fetch. Scout does it once and returns structured JSON instead of markup.

## What's here

- Local hybrid search (vector + BM25 keyword) over a markdown corpus: `scout/ingest.py`, `scout/embeddings.py`, `scout/bm25.py`, `scout/index.py`, `scout/search.py`. Served as `POST /api/v1/search`.
- HTML ingestion: fetch a URL or accept raw HTML, extract clean content with `trafilatura`, chunk it, index it. `scout/fetch.py`, `scout/webextract.py`, `scout/web.py`. Served as `POST /api/v1/ingest`. Ingested pages persist to disk and survive a restart, and re-ingesting the same URL replaces its content instead of duplicating it.

Ingestion is a first pass, not a finished pipeline. See [ROADMAP.md](ROADMAP.md) for what's missing (a real vector store past demo scale, JS-rendered pages, richer structure).

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
      "confidence": 0.574,
      "metadata": { "vector_score": 0.344, "keyword_score": 1.0, "id": "..." }
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

Defaults (embedding model, docs path, `top_k`, score-fusion weights) live in `scout/config.json`, loaded via `scout/config.py`.

## Tests

```bash
pytest
```

## Layout

```
scout/          package: ingest, embed, bm25, index, search, fetch, webextract, web, api, cli
data/docs/      markdown source documents (demo corpus)
data/index/     generated embedding cache + persisted ingested pages, git-ignored
tests/          pytest suite
```

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Roadmap

See [ROADMAP.md](ROADMAP.md).
