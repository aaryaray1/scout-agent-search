# Retrieval

Covers `scout/search.py`, `scout/bm25.py` and `scout/embeddings.py`.

## The shape of a search

A query is scored two ways and the results are fused:

- **Dense.** The query is embedded with a sentence-transformer and compared
  against every chunk embedding by cosine similarity. One numpy matmul.
- **Sparse.** Okapi BM25 over an inverted index of the same chunks.

Raw BM25 scores are unbounded and can run well past 1 for a strong rare-term
match, so they are min-max normalized to [0, 1] before fusing. Without that,
`vector_weight` and `keyword_weight` would not mean what they say.

```
fused = vector_weight * cosine + keyword_weight * normalized_bm25
```

Both weights are guessed rather than tuned. Fixing that needs a labeled
relevance set and an eval harness, which is ROADMAP Phase 6.

Ranking happens on the score array, and result dicts are built only for the
chunks actually returned. An earlier version built a dict for every chunk in
the corpus on every query and discarded all but `top_k`.

## Two corpus layers

`Retriever` holds the corpus as two layers concatenated into one searchable
view:

- `_docs_*` - the markdown corpus under `docs_path`, loaded at startup and
  cached to disk by content hash.
- `_ingested_*` - pages added later through `add_chunks()` / `/api/v1/ingest`,
  persisted through a `ChunkStore` (see [storage.md](storage.md)).

`corpus`, `corpus_embeddings` and `_bm25_slots` are derived from both and
replaced as a group whenever either layer changes.

Two layers rather than one flat list is what makes de-duplication on
re-ingest tractable: dropping a stale source only ever touches the ingested
layer, never the docs corpus. Docs come first in the concatenation, so slot
order matches corpus order and a compaction can reset both mappings to plain
ranges.

An empty `docs_path` is a warning, not an error. An ingest-only deployment is
the whole point of `/api/v1/ingest`, and refusing to start without local
markdown would mean shipping a demo corpus alongside every deployment.

### Invariants

The test suite enforces these. Breaking them does not raise, it silently
returns the wrong evidence:

- `corpus_embeddings.shape[0] == len(corpus)`
- `len(_bm25_slots) == len(corpus)`, each entry mapping a corpus position to
  its slot in the keyword index
- `corpus` holds live chunk dicts only: no holes, no stale version of a
  re-ingested source
- `/health` `corpus_size` is the live chunk count

## The keyword index

`scout/bm25.py` is an inverted index, `term -> {slot: term frequency}`,
implemented directly rather than pulling in a search library. The formula is
compact, there is no dependency of uncertain modern-Python support, and every
step of it is under test against hand-computed reference values.

It started as one `Counter` per document with a full scan per query and no
way to add or remove anything, so `Retriever` rebuilt it from scratch on
every ingest. At 20k chunks that rebuild cost 3.4 seconds per ingest. The
inverted form makes three things cheap that were not:

- adding documents costs the terms in those documents, not the corpus
- removing them costs the same, since a slot can be popped out of each of its
  terms' posting lists directly
- scoring touches only documents containing a query term

Full measurements and the decision behind it:
[ADR-001](../architecture/adr-001-incremental-index.md).

### Slots

A document's position in the keyword index is called a **slot** and never
changes while the document is live, because posting lists hold slot numbers.
Removing a document leaves a tombstone: its slot stays, and `scores()` still
returns an entry for it (always 0.0) so the returned list stays aligned with
the caller's own mapping.

`Retriever._bm25_slots` is that mapping, corpus position to slot. Scoring
gathers through it, which is a numpy index rather than a Python loop.

Getting this wrong does not raise. It attributes every keyword score to the
wrong chunk. `tests/test_search.py` has three tests aimed squarely at it, and
the fuzz oracle in the same file checks the mapping against a rebuild across
random ingest sequences.

`compacted()` reclaims tombstoned slots by renumbering, and is the only
operation that invalidates slot numbers. It returns a new index rather than
mutating in place, because re-keying every posting list is proportional to
the whole index (about 600ms at 20k chunks half-tombstoned) and doing that
with readers excluded would stall every concurrent search. The retriever
builds it outside the lock and swaps it in under the lock.

Compaction preserves insertion order, and the corpus is laid out in that same
order, so afterwards both slot mappings become plain ranges again.

Thresholds (`_COMPACT_TOMBSTONE_RATIO`, `_COMPACT_MIN_TOMBSTONES`) exist
because tombstones cost memory but compaction costs a renumbering: compact
only once the dead weight is a meaningful share of the index and more than a
handful of documents, so a tiny corpus does not compact on every re-ingest.

### Deliberately not thread-safe

`BM25Index` is mutated in place, so a reader iterating a posting list while a
writer inserts raises `RuntimeError`. Callers serialize access. Pushing a
lock down into the index would duplicate the one `Retriever` already holds.

## Concurrency

FastAPI runs Scout's sync endpoints in a threadpool, so a search can land
mid-ingest. Two locks:

- **`_ingest_lock`** serializes `add_chunks()` against itself, covering the
  ingested layer and its durable store.
- **`_index_lock`** covers the searchable view - corpus, embeddings, slot
  mapping, and the BM25 index itself - on both the read and the write side.

`_index_lock` covering *reads* is a change from the original design, which
kept readers lock-free by building a whole new BM25 index per ingest and
swapping it in. That swap was the thing being made cheap, and it is exactly
what an incremental index cannot offer.

It is affordable for the same reason the change was worth making: mutation is
proportional to what changed rather than to the corpus, and BM25 scoring is
pure Python, which the GIL already serializes across threads. The work that
genuinely parallelizes - embedding a query, the numpy similarity scan, disk
writes - all happens outside the lock. Copy-on-write was the alternative and
was rejected: copying the postings is O(total postings), the exact cost the
change exists to remove.

### Known exception

`/api/v1/search/batch` scores up to `max_batch_queries` queries in one locked
section. The GIL argument still applies per query, but hold time multiplies,
and that endpoint is not rate limited, so concurrent batch callers can delay
an ingest more than a single search would.

Scoring per query would shorten the hold but would give up the guarantee that
every query in a batch sees one corpus snapshot, which is the endpoint's
reason to exist. The real fix is a reader-writer lock, tracked in ROADMAP
Phase 5 where concurrency actually gets exercised.

## Failure part way through an ingest

`add_chunks()` touches the keyword index, two parallel lists, an embedding
matrix and a file. Three orderings are load-bearing, and each one is a
regression test:

1. **Build before assigning.** `_without_sources()` is free of side effects
   and the merged layer is built entirely into locals, so an allocation
   failure part way through leaves the retriever exactly as it was. Without
   that, the chunk list outgrows the embedding matrix, every later search
   raises, and the mismatch gets persisted so the service will not restart.
2. **Durable before searchable.** The store write happens before the page
   goes live in memory. The reverse means a failed write returns an error
   while the page still answers queries until the next restart, then vanishes
   with nothing having reported a problem.
3. **Compaction outside the lock.** See slots, above.

## The embedding model

`scout/embeddings.py` caches the loaded `SentenceTransformer` per model name
at module level. `SentenceTransformer(name)` reads weights off disk (or
downloads them) on every call, and nothing about the model changes between
`Retriever` instances, so without the cache every new `Retriever()` - one per
test, in the suite - paid that load again.

The `dimension` property exists to size an empty embedding matrix when there
is nothing to embed yet, which is the normal state for an ingest-only
deployment.
