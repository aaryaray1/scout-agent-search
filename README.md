# Scout

Scout is a search engine built **for agents, not browsers**.

## The problem

When an agent needs information from the web, it typically fetches a page as
raw HTML and then has to parse, clean, and restructure that HTML into
something a model can actually reason over (usually JSON) before it's useful.
That conversion step is repeated by every agent, on every fetch, and it costs
real time and tokens.

Scout's goal is to remove that step entirely: agents query Scout and get
back **pre-structured JSON evidence** — ranked, sourced, and ready to use —
instead of a blob of markup they have to parse themselves.

## What's implemented today

The current prototype is the retrieval half of that pipeline: a hybrid
(vector + keyword) search engine over a corpus of markdown documents.

- **Ingestion** ([scout/ingest.py](scout/ingest.py)) — loads markdown files, chunks them with overlap.
- **Embeddings** ([scout/embeddings.py](scout/embeddings.py)) — sentence-transformer vector encoding.
- **Indexing** ([scout/index.py](scout/index.py)) — caches embeddings to disk, keyed by a hash of the corpus so re-ingesting unchanged docs is a no-op.
- **Search** ([scout/search.py](scout/search.py)) — cosine similarity + keyword overlap, fused into one confidence score, returned as evidence objects.
- **API** ([scout/api.py](scout/api.py)) — a small FastAPI service exposing `/api/v1/search`.

A first pass at the actual point of the project — turning a live web page
into structured JSON instead of leaving that to the agent — now exists too:

- **Fetch** ([scout/fetch.py](scout/fetch.py)) — SSRF-guarded HTTP fetcher.
- **Extract** ([scout/webextract.py](scout/webextract.py)) — HTML → clean markdown + metadata via `trafilatura`.
- **Ingest API** ([scout/api.py](scout/api.py)) — `POST /api/v1/ingest` takes a `url` (or raw `html` + `source_url`), returns structured JSON immediately, and indexes it in-memory so it's searchable right away.

It's a first pass, not a finished pipeline — see [ROADMAP.md](ROADMAP.md) for
what's still open (durable storage for ingested pages, JS-rendered pages,
richer structure beyond markdown).

## Quickstart

```bash
# from the repo root
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"

# build the search index from data/docs
scout-ingest

# run the API
uvicorn scout.api:app --reload
```

Then query it:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "rate limit exceeded", "top_k": 2}'
```

Response shape:

```json
{
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

Or hand it a live page instead of pre-built docs — this is the part meant to
replace an agent's own HTML → JSON step:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/some/docs/page"}'
```

which returns structured content directly and makes it searchable
immediately (see [ROADMAP.md](ROADMAP.md) for current limits, notably: this
index is in-memory only and won't survive a restart yet).

## Configuration

Defaults (embedding model, docs path, `top_k`, score-fusion weights) live in
[scout/config.json](scout/config.json) and are loaded via
[scout/config.py](scout/config.py).

## Running tests

```bash
pytest
```

## Project layout

```
scout/          the package: ingest, embed, index, search, api, cli
data/docs/      markdown source documents (the demo corpus)
data/index/     generated embedding cache (git-ignored, rebuilt by scout-ingest)
tests/          pytest suite
```

## License

Not yet chosen - this repo isn't licensed for reuse until a LICENSE file is
added.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for what's next, starting with the HTML→JSON
ingestion pipeline that's the actual point of this project.
