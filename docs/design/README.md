# Design notes

Why Scout is built the way it is. The code carries short comments and points
here; these files carry the reasoning, the bugs that motivated a decision,
and the measurements behind it.

| File | Covers |
| --- | --- |
| [retrieval.md](retrieval.md) | `search.py`, `bm25.py`, `embeddings.py` - hybrid scoring, slots, the concurrency model |
| [storage.md](storage.md) | `index.py`, `store/` - the two corpus layers, segments, crash safety, corruption policy |
| [ingestion.md](ingestion.md) | `fetch.py`, `webextract.py`, `web.py`, `ingest.py` - fetching, SSRF, extraction, chunking |
| [api.md](api.md) | `api.py`, `models.py`, `auth.py`, `ratelimit.py`, `client.py` - the HTTP surface and what guards it |
| [configuration.md](configuration.md) | `config.py` - the layering and every setting |

Decisions large enough to have alternatives live in
[../architecture/](../architecture/) as ADRs instead.

[ROADMAP.md](../../ROADMAP.md) is the forward plan; these files describe what
is there now.
