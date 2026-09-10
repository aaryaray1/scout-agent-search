# The MCP server

Covers `scout/mcp.py`. This is the surface an agent finds on its own, rather
than one somebody wires up for it.

```bash
pip install "scout-agent-search[mcp]"
scout-mcp                                   # Scout in this process
scout-mcp --url https://scout.example.com   # a shared Scout over HTTP
```

```json
{"mcpServers": {"scout": {"command": "scout-mcp"}}}
```

## Two tools, and why only two

**`ingest(urls)`** fetches pages, extracts their real content, indexes them,
and returns a short summary per page.

**`search(query, top_k)`** returns the passages that answer a question.

That pair is the whole workflow: ingest what you would otherwise read, then
pull back only the parts that matter. Anything else an agent might want -
filtering by source, re-reading one page - is a variation on `search`, and
adding a tool for each one costs tokens in the tool listing that every single
request pays, whether or not the tool is used.

## ingest returns a summary, not the page

This is the design decision the whole server turns on. An `ingest` that
echoed the page body back would cost exactly the tokens the agent came here
to save; it would be `fetch` with extra steps.

So the response is bounded regardless of page size:

```json
{"status": "ok", "url": "...", "title": "Quernstone Bearings",
 "chunks_indexed": 3, "words": 812,
 "metadata": {"author": null, "date": "2026-02-01", "sitename": "..."},
 "preview": "Quernstone bearings fail at fault code Q-8812 when..."}
```

`preview` is capped at 40 words and exists for one reason: so the agent can
tell it got the page it meant without paying for the body. `words` and
`chunks_indexed` say how much is now searchable.

The body is not discarded, it is indexed. `search` is how it comes back, a
few hundred tokens at a time instead of a few thousand.

`tests/test_mcp.py` pins this with a page long enough that a bounded summary
cannot accidentally contain all of it - the first version of that test used a
22-word fixture that fit entirely inside the preview and passed for the wrong
reason.

## search returns less than the HTTP API does

`/api/v1/search` returns `vector_score`, `keyword_score` and a chunk id
alongside each passage. An agent acts on the text and the source and does
nothing with the rest, so the MCP tool returns `content`, `source` and
`confidence` only.

That trimming is not code, it is the `Passage` model: the tool declares
`-> SearchResult`, and pydantic validates the retriever's richer dicts down
to the declared shape. There is nothing to keep in sync.

## Both tools publish an output schema

Returning a pydantic model rather than a `dict` is what makes the SDK emit
structured content and an `outputSchema` in the tool listing. A bare `dict`
return is rejected for structured output, and would leave the agent parsing
JSON text out of a content block - which is precisely the thing Scout exists
to stop agents doing.

## Failure is per URL

`ingest` takes a list, and one bad page fails alone with its own status
(`fetch_error`, `extract_error`, `error`), matching
`/api/v1/ingest/batch`. An agent handing over ten links gets the eight that
worked. Anything else propagates: a bug should look like a bug.

## Two backends, one shape

`LocalScout` runs Scout in this process - a `Retriever` plus the ingest
pipeline. Nothing to deploy, which is the point for a single agent on a
laptop, and the index lives wherever `SCOUT_INDEX_DIR` points.

`RemoteScout` wraps `ScoutClient` against a running server, for several
agents sharing one corpus, or for keeping the embedding model out of the
agent's process.

Both expose the same two methods and both return the same `IngestedPage`, so
the tools never branch on which is behind them. `--url` chooses.

## Threading

Tool functions are plain `def`. The SDK runs sync tools through
`anyio.to_thread.run_sync`, so concurrent tool calls reach the `Retriever`
from several threads - which is the concurrency it is already built for (see
[retrieval.md](retrieval.md)). Writing them `async` would have put blocking
fetches and embedding on the event loop instead.

## stdout belongs to the transport

Logging goes to stderr. Anything written to stdout that is not a protocol
message breaks the session.

Worth knowing when driving this from a script: the embedding model writes
progress to stderr on startup, so a parent process that pipes stderr must
drain it. An undrained pipe fills and deadlocks the server before it
answers - which is a property of pipes, not of Scout, but it looks exactly
like a hang.
