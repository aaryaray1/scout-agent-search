# The HTTP surface

Covers `scout/api.py`, `scout/models.py`, `scout/auth.py`,
`scout/ratelimit.py`, `scout/client.py` and `scout/logsetup.py`.

## Endpoints

| Route | Auth | Rate limited | What it is for |
| --- | --- | --- | --- |
| `GET /` | no | no | Service identity and where the docs are |
| `GET /health` | no | no | Liveness, corpus size, whether auth is on |
| `POST /api/v1/search` | yes | no | Evidence for one query |
| `POST /api/v1/search/batch` | yes | no | Several queries, one corpus snapshot |
| `POST /api/v1/ingest` | yes | yes | One page (by URL, or raw HTML) to JSON |
| `POST /api/v1/ingest/batch` | yes | yes, per item | Several pages, one index write |

`/` and `/health` are deliberately open: a liveness probe should not need a
credential, and neither reveals anything a caller could not learn from the
port answering at all. `/health` reports whether auth is enabled, because
"did my API keys actually reach the container" is otherwise only answerable
by getting rejected.

## Two structural choices

**Endpoints are `def`, not `async def`.** Retrieval is CPU-bound and ingest
does a blocking outbound fetch, so declaring them sync lets Starlette run
them in its threadpool instead of stalling the event loop. `Retriever` is
built for exactly that concurrency - see [retrieval.md](retrieval.md).

**State is built in a lifespan handler, not at import time.** The retriever
used to be a module-level singleton, which meant merely importing `scout.api`
- as any test file must - loaded the embedding model and touched disk.
Building it in `lifespan` also gives test isolation a place to redirect where
it reads and writes.

## Batch semantics

Both batch endpoints exist to do something a client looping over the single
endpoint cannot.

**`/search/batch`** answers every query against one corpus snapshot and
embeds the whole batch in a single pass through the model, so results within
a batch stay mutually consistent even if an ingest lands mid-batch. That
guarantee is the endpoint's reason to exist; the lock-hold cost of it is
noted in [retrieval.md](retrieval.md).

**`/ingest/batch`** isolates per-item failure. The two exception types the
single-page endpoint turns into 502 and 422 become per-item `status` values
(`fetch_error`, `extract_error`), so an agent crawling ten links gets the
eight that worked instead of one error. Every successful page goes in through
one `add_chunks()` call, so the index write happens once for the batch. A URL
repeated inside one batch is indexed once, matching the store's own "one
version per source" rule, while every item still gets its own response entry.

A batch costs the rate limiter one unit *per item*. Charging it as one
request would make `/ingest/batch` a way to buy N outbound fetches for the
price of one.

## Limits live on the models

Request-shape bounds are declared on the Pydantic models rather than checked
inside endpoint bodies, so FastAPI publishes them in `/openapi.json`. An
agent can read the limits instead of discovering them through a 422 in
production. A test asserts the bounds are actually published.

| Bound | Setting | Default |
| --- | --- | --- |
| `top_k` | `max_top_k` | 50 |
| query length | `max_query_chars` | 2000 |
| HTML payload | `max_html_bytes` | 5 MB |
| queries per batch | `max_batch_queries` | 10 |
| pages per batch | `max_batch_ingest` | 5 |
| URL length | (constant) | 2048 |

Queries are stripped before validation, so a whitespace-only query collapses
to `""` and fails `min_length` in the schema rather than in a hand-rolled
check. `top_k` had no lower bound once: `top_k or self.top_k` turned an
explicit `0` into the server default, and a negative value sliced from the
end of the ranking. It is now bounded on the model and validated again in
`Retriever.search()` for direct library callers.

The caps bound what a single request costs Scout. They are not a substitute
for auth and rate limiting.

## schema_version

Every response envelope carries `schema_version` (`"1.0"`), bumped on any
breaking change to `Evidence` / `SearchResponse` / `IngestResponse` shape.

It is a check, not a contract. Nothing negotiates it, and there is no
documented list of what changes between versions. That gap starts to matter
once integrations ship, which is why it is ROADMAP Phase 7.

## Auth

A shared-secret `X-API-Key` header covering every `/api/v1` route. There is
no user model, no key issuance and no scopes: Scout serves agents on behalf
of one operator, so a key answers "is this caller allowed to spend my network
and CPU", which is the only question the service has.

Two properties matter more than the size of the mechanism:

- **Fails closed on a bad key, open only when explicitly unconfigured.** No
  keys configured means no auth at all, which is right for a laptop and wrong
  for anything reachable, so startup logs which of the two it is.
- **The key never reaches a log line or an error body.** Callers are
  identified downstream by a 12-character hash of the key.

Implementation details that are not incidental:

- Keys are compared **as bytes** in constant time against **every** configured
  key. Returning on the first match would leak key ordering to anyone timing
  it; comparing as `str` raises `TypeError` on non-ASCII input, which is
  caller-controlled and would have been a 500 rather than a 401.
- `APIKeyHeader(auto_error=False)`, so a missing header reaches Scout's own
  handler instead of FastAPI raising a 403: an unauthenticated deployment has
  to serve the request, and an authenticated one wants to answer 401 with its
  own message.
- The dependency is declared with `Security()` rather than `Depends()`, so
  the header appears as a security scheme in `/openapi.json`.
- Keys are stripped and emptied in `ApiKeyAuth` as well as in the config
  parser, because a list can also arrive from `config.json`, which does no
  stripping, and a whitespace-only "key" would be a live credential that a
  blank header matches.

Gaps, all ROADMAP Phase 5: no rotation without a restart, no per-key scopes,
and `X-Forwarded-For` is deliberately not trusted (anyone can send it), so a
deployment behind a proxy needs a configured trusted-proxy list rather than
blanket trust.

## Rate limiting

Ingest only, per caller identity - the API key when auth is on, the client IP
when it is not.

A **token bucket** rather than a fixed window, for two reasons: a window
boundary lets a caller send 2x the limit across it, and a continuously
refilling bucket can answer `Retry-After` exactly instead of "some time
before the window flips". A rejected request spends nothing, so a client
retrying in a loop does not push its own deadline out. `Retry-After` is
rounded up, since it is defined in whole seconds and a client obeying it
exactly should not come back a fraction too early.

A request asking for more than the whole bucket could never succeed however
long the caller waits, so the cost is clamped to the bucket size and the
answer is one window.

Idle buckets are evicted on the write path. The alternative is an unbounded
map of every caller identity ever seen - for an unauthenticated deployment,
every client IP - which is a slow memory leak driven by whoever is talking to
Scout.

In-process, so N uvicorn workers means N times the configured limit. A shared
limiter wants the same external store a vector database would, which is why
ADR-001 records that those two decisions should land together.

## The Python client

`scout/client.py`. The point of Scout is that an agent does not write the
glue between "I have a URL" and "I have structured evidence"; a framework
hand-rolling `httpx.post` would rediscover the auth header, the error shape
and the batch payloads every time.

- **Returns plain dicts**, not the server's Pydantic models. Parsing into
  them would mean a client one version behind rejects a response carrying a
  field it has not heard of - exactly the coupling `schema_version` exists to
  let callers manage. The version is checked and warns on a major mismatch;
  it is the caller who decides whether it can still work.
- **`ScoutAPIError` keeps `status_code`, `detail` and `retry_after`
  separate**, so a caller can branch: 401 means fix your key, 429 means back
  off, 502 means the page was unreachable and retrying may work.
- **An `httpx.Client` can be injected.** That is what makes it testable
  against the real ASGI app with no server running, and the hook a caller
  needs for a proxy, a retry transport or mTLS. A client passed in is not
  closed by this one - whoever opened it owns it.
- **A non-JSON error body is surfaced verbatim** (truncated), because it
  means something other than Scout answered: a proxy, a load balancer, an
  error page.
- The default timeout is 60s, well above the server's own 10s fetch timeout,
  because an ingest can spend that fetch plus extraction plus embedding
  before it answers.

## Logging

Scout's own loggers run at the configured level; everything else is pinned to
WARNING. Without that split, turning Scout up to INFO also turns on
sentence-transformers' model-loading chatter and httpx's per-request logging
- roughly two dozen extra lines on every startup, enough to bury Scout's own
output.

A host that has already configured logging (uvicorn, gunicorn, pytest) keeps
its handlers and formatting; only Scout's level is adjusted, so this never
fights the process that owns the root logger.
