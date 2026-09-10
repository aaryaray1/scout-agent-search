# ADR-001: An incremental in-process index, not a vector database

## Status

Accepted, 2026-09-10. Supersedes the Phase 2 plan as originally written in
`ROADMAP.md` ("swap the flat numpy array and JSON metadata cache for a real
vector store").

## Context

Phase 2 exists because the Phase 0 storage design does not scale. That much
is not in dispute. The question is which part of it to fix first, and the
original roadmap answered it by naming a technology (`qdrant-client`, which
sat in `requirements.txt` early in the project's history) rather than by
measuring anything.

Measured before deciding, with `scripts/bench_index.py`. Synthetic corpus,
400-word Zipf-distributed chunks over an 8k vocabulary, best of three runs
with the collector disabled inside each timed region:

| chunks | keyword index rebuild | keyword query | vector query | full store rewrite |
| --- | --- | --- | --- | --- |
| 1,000 | 128ms | 1.1ms | 0.5ms | 16ms |
| 5,000 | 794ms | 8.9ms | 3.5ms | 57ms |
| 20,000 | 3,441ms | 39.4ms | 11.3ms | 238ms |

The rebuild and the store rewrite are both paid on **every ingest**, because
`Retriever.add_chunks()` calls `_rebuild_combined_corpus()`, which constructs
`BM25Index(corpus)` from scratch and re-tokenizes the entire corpus, and
`save_ingested()`, which rewrites the whole store.

So at 20k chunks an ingest costs about 3.7 seconds, of which more than 90% is
re-tokenizing text that did not change, and it grows linearly: 100k chunks is
roughly 17 seconds per ingest.

A vector database addresses the 11.3ms column. It does not address the
3,441ms column, which is 300 times larger and is the one that grows into a
timeout.

A note on how these were measured, since two earlier passes produced numbers
that did not hold up. The first drew words uniformly from the vocabulary,
which yields chunks with about 390 distinct terms where real prose at that
length has closer to 237; uniform draws inflate indexing cost and deflate
query cost at once, because query terms land in short posting lists no real
query would hit. The second timed several 20k-chunk indexes while they were
all alive at once, and allocation pressure moved the result by more than the
effect being measured. Neither reversed the decision - the gap between the
columns is far too wide for that - but both were wrong enough to be worth not
publishing, and the benchmark script now controls for both.

## Decision

Make the in-process index incremental, and defer the vector database.

1. Replace `BM25Index`'s full-rebuild model with an inverted index
   (`term -> {slot: tf}`) supporting `add_documents`, `remove_slots` and
   `compact`. Ingest cost becomes proportional to what changed.
2. Change the retriever's concurrency model to suit an index that is mutated
   in place rather than rebuilt and swapped (see "Consequences").
3. Replace the full-store rewrite with an append-only segmented store plus
   tombstones and threshold compaction.
4. Introduce a `ChunkStore` seam (`scout/store/base.py`) that the segmented
   store implements, so that adding Qdrant or pgvector later is a new class
   behind an existing interface plus a config value, not a rewrite of
   `Retriever`.

## Rationale

1. **It targets the measured bottleneck.** Items 1 and 3 remove the 3,441ms
   and the 238ms respectively. Item 4 leaves the door open to remove the
   11.3ms when that becomes the largest number, which it currently is not.
2. **It preserves the deployment story.** Scout's ergonomics claim is that an
   agent operator can `pip install` it and point an agent at it. Requiring a
   Qdrant container to run at all contradicts the product, and would have to
   be justified by a performance problem that the measurements say does not
   exist yet.
3. **It costs no new dependencies.** An inverted index and a segment log are
   compact enough to own directly, which is the same argument that produced
   `scout/bm25.py` instead of a `whoosh` dependency, and it keeps the whole
   thing under test.
4. **A vector database does not remove the BM25 problem anyway.** Scout is a
   hybrid retriever. Moving the dense half out of process leaves the sparse
   half exactly where it is, so item 1 would still have to be built.

## Options considered

| Option | Fixes the 3,441ms rebuild | Fixes the 238ms rewrite | Fixes the 11.3ms scan | New deps | Deploy |
| --- | --- | --- | --- | --- | --- |
| Incremental in-process index (chosen) | yes | yes | no | none | `pip install` |
| Integrate Qdrant now | no | yes | yes | qdrant-client + server | Docker required |
| Both | yes | yes | yes | qdrant-client + server | Docker for the scale path |
| Do nothing | no | no | no | none | unchanged |

"Both" is the correct end state and is not ruled out; item 4 is what makes it
reachable incrementally. It was rejected as a single step only because it
roughly doubles the scope of Phase 2 to buy a column that is not yet the
constraint.

## Trade-offs accepted

- **Retrieval stays a linear scan.** Accepted: 11.3ms at 20k chunks, and the
  matmul releases the GIL, so it does not serialize across threads. This is
  the number to watch.
- **The index stays in one process.** Multiple uvicorn workers still cannot
  share one index, which is the same constraint that already applies to the
  rate limiter. Accepted for a single-operator service.
- **Memory holds the whole corpus.** At 20k chunks and 384 dimensions that is
  about 30MB of vectors plus the postings. Fine at the scale this decision
  covers, and the revisit trigger below catches the point where it stops
  being fine.
- **Readers now take a lock during keyword scoring.** See below.

## Consequences

**Positive.** Ingest goes from O(corpus) to O(changed) on the CPU side.
Queries get faster too, because an inverted index only scores documents that
contain a query term instead of looping over every document in the corpus.
No new operational surface.

**Negative, and the one that needs care.** The old design got its thread
safety from immutability: `_rebuild_combined_corpus()` built a whole new
`BM25Index` off to the side and swapped it in under `_swap_lock`, so a
concurrent search either saw the old index or the new one. An index that is
mutated in place cannot offer that. A search iterating a posting list while
an ingest inserts into it raises `RuntimeError: dictionary changed size
during iteration`, and a search that reads a partially-added document scores
against a corpus that never existed.

**Mitigation.** Promote `_swap_lock` to a single index lock that covers both
mutation and keyword scoring, rather than only the handover. This is
affordable precisely because of the change being made: mutation is now
O(changed) instead of O(corpus), and BM25 scoring is pure Python, which the
GIL already serializes across threads, so putting a lock around it costs
close to nothing that was not already being paid. Query embedding and all
numpy work stay outside the lock, which is where the parallelism actually
was.

Copy-on-write was considered as an alternative mitigation and rejected:
copying the postings is O(total postings), which reintroduces the cost the
whole change exists to remove.

## Outcome

Items 1 and 2 landed on 2026-09-10. Same benchmark, same machine:

| chunks | keyword update per ingest | keyword query |
| --- | --- | --- |
| 1,000 | 1.25ms (was 128ms) | 0.6ms (was 1.1ms) |
| 5,000 | 1.13ms (was 794ms) | 1.8ms (was 8.9ms) |
| 20,000 | 1.97ms (was 3,441ms) | 3.3ms (was 39.4ms) |

The keyword half of an ingest is now flat in the size of the corpus, which
was the point. What remains per ingest is the 238ms store rewrite, which is
item 3.

Absolute figures depend on machine state more than is comfortable: the same
script on a busy desktop reports every column, including ones no Scout code
touches, at roughly 3x these numbers. Compare columns within one run rather
than numbers between runs. The ratio is what the decision rests on, and it
holds at roughly three orders of magnitude either way.

The store-rewrite column was later found to include temp-directory creation
and teardown, and the add column to be measured against an index the previous
repeat had grown. Both were corrected in the script; the add figure did not
move materially, and the store-rewrite figure should be treated as an upper
bound until it is re-baselined on an idle machine.

Two things did not go the way the plan assumed, and are worth recording:

- **Building the index cold got no cheaper, and by some measurements got
  more expensive.** An inverted index writes one entry per (document, term)
  pair into thousands of separate posting lists, where the old structure
  incremented one flat counter. That cost is now paid once at startup rather
  than on every ingest, which is the trade being made deliberately; it is not
  a free win. Startup for a 20k-chunk ingested corpus spends about 3.4s
  building the keyword index.
- **Full-rebuild timings past about 20k chunks were not reproducible** to
  better than a factor of three, because the measurement itself allocates
  heavily enough to provoke the collector. The per-ingest and per-query
  numbers above are stable and were reproduced across runs; the cold-build
  column should be read as an order of magnitude, not a figure.

## Revisit trigger

Reopen this decision when any of these becomes true:

- the corpus passes roughly 200k chunks, at which point the linear scan is
  the largest measured number rather than the smallest
- a deployment needs more than one worker process sharing one index
- the working set stops fitting comfortably in memory

At that point item 4's seam is what gets used: implement `ChunkStore` against
Qdrant or pgvector and select it with `SCOUT_VECTOR_BACKEND`. Note that the
seam deliberately stops short of a delegated nearest-neighbour query, so
adopting an external store is a change to retrieval and not only to storage;
`scout/store/base.py` says why guessing at that half of the interface now
would be worse than discovering it. The
shared rate limiter (`ROADMAP.md` Phase 3) should land with that same
decision, since both want the same piece of external infrastructure.
