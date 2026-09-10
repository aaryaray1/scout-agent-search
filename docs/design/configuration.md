# Configuration

Covers `scout/config.py`.

## Layering

```
built-in defaults  <  scout/config.json  <  SCOUT_* environment variables
```

The environment layer wins because hosting a service means changing its
settings without editing files inside the image. Every key is overridable:
`top_k` becomes `SCOUT_TOP_K`, `index_dir` becomes `SCOUT_INDEX_DIR`, and so
on.

Environment values arrive as strings, so every non-string setting declares
how to read itself back. A value that will not parse is logged and ignored
rather than crashing startup on a typo in one variable.

The merged config is resolved once per process and handed out as a copy, so
callers on a hot path (chunking every document) do not re-read
`config.json`, and a caller mutating what it gets back cannot corrupt anyone
else's view. List settings are copied too, so that promise holds a level
deeper than a plain `dict()` copy would make it.

`reset_config_cache()` exists for tests and for a process that edits its own
environment after import. A deployed server resolves config once at startup.

## Settings

| Key | Default | What it does |
| --- | --- | --- |
| `vector_model` | `all-MiniLM-L6-v2` | sentence-transformers model name |
| `docs_path` | `data/docs` | folder of local markdown to index at startup |
| `index_dir` | `data/index` | where every durable file lives |
| `chunk_store` | `segmented` | `segmented` or `json`, see [storage.md](storage.md) |
| `top_k` | 3 | default result count |
| `vector_weight` | 0.65 | dense half of the fused score |
| `keyword_weight` | 0.35 | sparse half of the fused score |
| `chunk_size` | 400 | words per chunk |
| `chunk_overlap` | 50 | words shared between neighbouring chunks |
| `max_top_k` | 50 | ceiling on a request's `top_k` |
| `max_query_chars` | 2000 | ceiling on query length |
| `max_html_bytes` | 5 MB | ceiling on an inline HTML payload |
| `max_batch_queries` | 10 | queries per `/search/batch` |
| `max_batch_ingest` | 5 | pages per `/ingest/batch` |
| `api_keys` | `[]` | empty means no auth at all |
| `ingest_rate_limit` | 30 | ingest tokens per window, 0 disables |
| `ingest_rate_window` | 60 | window in seconds |
| `log_level` | `INFO` | Scout's own loggers only |

`SCOUT_API_KEYS` is comma-separated, because secrets arrive through the
environment far more often than through a committed file. An empty or
whitespace-only value yields no keys, which is a deliberate "no auth", not a
parse failure.

## Validation

Settings that would break the pipeline are rejected at load, naming the
setting, rather than failing somewhere less obvious later:

- **`chunk_overlap >= chunk_size`** is the one that genuinely mattered.
  `chunk_text()` advances by `chunk_size - overlap`, so an overlap at or
  above chunk size never advances: the process hangs and fills memory. This
  turns a hang into a startup error.
- **`top_k < 1`** would give an always-empty result.
- **`max_batch_* < 1`** would make a batch endpoint reject every request as a
  confusing 422.
- **`ingest_rate_window < 1`** would divide by zero at request time.
- **`ingest_rate_limit < 0`** is meaningless; 0 is the documented way to
  disable the limiter.
- **An unknown `chunk_store`** fails at startup rather than on the first
  ingest, with the ingested store already in whichever state the previous run
  left it.

The valid store names are a literal in `config.py` rather than imported from
`scout.store`. Config is the bottom of the import graph - `scout.store` reads
it - so importing back up would be a cycle.
