# Benchmarking

`scripts/bench_index.py` is where the numbers in `ROADMAP.md`,
[storage.md](storage.md) and
[ADR-001](../architecture/adr-001-incremental-index.md) come from, so they can
be re-checked rather than trusted.

```
python scripts/bench_index.py
python scripts/bench_index.py --sizes 1000,5000,20000,100000
```

## What it measures, and what it leaves out

Five columns: building the keyword index cold, adding one page to it, one
keyword query, one vector query, and one durable store write per backend.

**It deliberately does not load the embedding model.** The costs here are
tokenization, posting-list maintenance, the similarity scan and the store
write. An embedding model would add a large constant to every row and make
the shape of the growth harder to see. Vector similarity is measured against
random vectors of the right width for the same reason.

## Three things that had to be controlled for

Each of these produced a number that did not hold up, and each is now
handled in the script.

**Word frequency follows a Zipf distribution, not a uniform one.** Drawing
400 words uniformly from an 8k vocabulary yields a chunk with about 390
distinct terms, which no real document has; Zipf yields about 237, which is
in range for English prose. The distribution sets both how much work indexing
a document is and how long the posting lists a query walks are, so uniform
draws flatter the query numbers while inflating the indexing ones. An earlier
uniform version of this benchmark pointed at a different conclusion.

**Timing takes the minimum of several trials, with the collector disabled
inside each.** These structures allocate heavily enough that a GC pause
landing in a timed region moves the result by more than the thing being
measured. Minimum rather than mean because the noise is all additive.

**Setup and teardown stay outside the timed region.** An early version of the
store column measured `mkdir` plus a recursive delete of a 30MB file
alongside the write, and an early version of the keyword-add column measured
each repeat against an index the previous repeat had grown - so it described
an index that never existed. `_time_one_add` now undoes its add by
tombstoning the slots, which restores the statistics an add depends on
without reallocating the corpus.

## Reading the output

**Absolute figures are machine-state dependent and are not comparable across
runs.** On a busy desktop every column here runs roughly 2-3x slower,
including the ones no Scout code touches. Compare columns within a single
run, never numbers between runs.

The ratios are what the decisions rest on, and they hold regardless of
machine state: cold build against incremental add is about three orders of
magnitude, and full store rewrite against segmented write grows with corpus
size.

**The cold-build column past about 20k chunks is an order of magnitude, not a
figure.** The measurement itself allocates heavily enough to provoke the
collector there. The per-ingest and per-query columns are stable and
reproduce across runs.
