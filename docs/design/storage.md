# Storage

Covers `scout/index.py` and `scout/store/`.

Everything durable lives under one directory (`index_dir`, default
`data/index/`), so a deployment mounts one volume.

## Two stores, two failure modes

```
data/index/
  embeddings.npy            docs corpus cache
  meta.json                 docs corpus cache
  corpus_hash.txt           docs corpus cache
  ingested/                 ingested pages (segmented store)
    manifest.json
    seg-000001.npy
    seg-000001.json
  ingested_embeddings.npy   ingested pages (json store, and the legacy format)
  ingested_meta.json
```

- **The docs cache** is a rebuildable cache of a fixed folder, keyed by a
  content hash. Losing it costs a re-embed, so corruption there degrades to a
  cache miss and a warning.
- **The ingested store** is the only copy of every page ever ingested.
  Corruption there raises `IndexCorruptError` rather than starting up empty,
  because starting up empty lets the next ingest overwrite data that was
  still recoverable.

A store whose embedding count and chunk count disagree is rejected the same
way. It would otherwise keep working while attributing every score to the
wrong chunk.

## Crash safety

Every write goes through `_atomic_write`: content lands in a temp file in the
same directory, is flushed and `fsync`ed, then `os.replace`d over the target.
A crash mid-write leaves the previous good file rather than a truncated one
that would poison the next startup.

Both stores were originally written with bare `open()` calls whose handles
were never closed and whose encoding defaulted to whatever the host happened
to use. `pytest` is configured to treat `ResourceWarning` as an error, so a
reintroduced leaked handle fails the suite.

## Cache validity is more than a hash

`compute_corpus_hash()` covers `source` as well as content: two documents
swapping filenames leaves the concatenated content identical, and hashing
content alone would serve a cached index attributing every chunk to the wrong
source.

Row count and vector width are checked alongside the hash. Changing
`SCOUT_VECTOR_MODEL` otherwise leaves a cache built by the previous model,
and stacking those vectors against freshly embedded ones either raises deep
inside numpy or silently scores against the wrong space.

## The ChunkStore seam

`scout/store/base.py` defines the interface; `chunk_store` selects an
implementation. `Retriever` talks only to the interface.

```python
load()                     -> (chunks, embeddings), or ([], None) when empty
upsert(chunks, embeddings) -> persist, replacing every source in `chunks`
delete_sources(sources)    -> remove every chunk under those sources
stats()                    -> StoreStats(live, sources, dead, segments)
```

Three rules the interface makes non-negotiable per backend:

- **Sources are the unit of replacement.** Re-ingesting a URL replaces that
  source's chunks rather than adding to them. Matching chunks up individually
  would leave orphans behind, because a page that gained or lost a section
  produces a different number of chunks than the version it replaces.
- **Implementations own crash safety.** A partially written store must never
  be readable as a complete one, and a store that cannot be read must raise
  rather than present itself as empty.
- **`load()` is never mandatory.** Every method reads what is on disk first
  if it has not already. An operation that destroys data when a caller skips
  an optional step is a bug in the store, not in the caller. This was a real
  bug: `JsonChunkStore.upsert()` on an instance that had never called
  `load()` rewrote the store with only the new chunks, destroying every
  previously ingested page.

Implementations are **not** thread-safe. `Retriever` serializes writes behind
its ingest lock, which is also where the embedding and the in-memory update
happen, so pushing locking down would only duplicate it.

### Deliberately not a vector-search interface

There is no `search()` here. Scout fuses dense and sparse scores over one
in-memory view of the corpus, so retrieval reads the whole embedding matrix
rather than delegating a nearest-neighbour query. An external vector database
would want the opposite, and the honest thing to record is that adopting one
is a change to *retrieval*, not only to storage. Inventing that half of the
interface now, with no implementation to check it against, would be guessing
at a shape rather than discovering it.

[ADR-001](../architecture/adr-001-incremental-index.md) has the measurements
behind deferring that and the conditions that reopen it.

## JsonChunkStore

One embeddings file and one metadata file, both rewritten in full on every
write. This is what Scout did before segments existed.

It is kept as the reference implementation: `tests/test_store.py` runs the
whole contract against every backend, so a case that passes here and fails
elsewhere is a regression in the new code rather than a new expectation. It
is also the rollback path (`chunk_store = "json"`).

Its weakness is why the segmented store exists. `upsert` costs the size of
everything ever ingested, not the size of what changed.

It holds the live set in memory between calls because rewriting in full
requires it: there is no way to replace one source's chunks in this format
without rewriting every other source's too.

## SegmentedChunkStore

The default. Each `upsert` writes one new segment - an embeddings file and a
metadata file, written once and never modified - and appends an entry to a
manifest recording which sources it supersedes.

```json
{"format": 1, "next_id": 3, "entries": [
  {"segment": "seg-000001", "sources": {"https://a": 4, "https://b": 2}},
  {"segment": "seg-000002", "sources": {"https://a": 6}},
  {"segment": null, "removes": ["https://b"]}
]}
```

Reading replays the manifest in order to decide which entry holds the live
version of each source, then reads only the segments that still own
something. An upsert entry supersedes exactly the sources it carries, so it
stores no separate `removes` list; only a deletion needs one.

**Nothing is held in memory between calls except the manifest.** `upsert`
does not need to know what is already stored, because superseding a source is
a line in the manifest rather than a rewrite. The retriever holds the live
corpus; this store holds source names and counts.

### Crash safety comes from the ordering

Segment files are written first and are inert until something references
them. The manifest is written last, through the same atomic path everything
else here uses. So a crash leaves either the old manifest - and an
unreferenced segment - or the new one. There is no window in which a
partially written store reads as a complete one.

An unreferenced segment is the debris of a crash between those two writes,
along with any temp file left the same way. The manifest is the record of
what exists, so anything it does not name was never committed and holds
nothing a reader could ever have seen. Those files are deleted on the next
load. Failing to delete one costs disk rather than correctness, so a
read-only mount logs a warning instead of refusing to open the store.

### Compaction

Superseded chunks stay on disk until a merge rewrites the live set into a
single segment. That merge costs the size of the store, so it sits behind two
thresholds:

- **dead weight** - at least 64 dead chunks *and* at least 30% of what is
  written, so a small corpus does not merge on every re-ingest
- **segment count** - more than 32 live segments, because a load opens two
  files per segment regardless of how much is dead

Merging at the segment bound spreads one full rewrite across at least that
many ingests. The merge follows the same ordering: new segment, then the
manifest naming it, then the files it replaced.

### Upgrading from the pre-segment format

If there is no manifest but the old `ingested_*.json` / `.npy` files are
there, they are imported as the first segment. Without that, upgrading Scout
would start the ingested store empty and let the next ingest supersede the
only copy of every page already in it.

The old files are left in place rather than deleted, so the upgrade stays
reversible, and are ignored from then on.

### What it costs

`scripts/bench_index.py`, same run, 400-word Zipf chunks over an 8k
vocabulary, best of three. Compare columns within a run, never numbers
between runs - machine state moves every column here, including ones no Scout
code touches.

| chunks | full rewrite, per ingest | segmented, per ingest |
| --- | --- | --- |
| 1,000 | 22ms | 3.9ms |
| 5,000 | 62ms | 3.9ms |
| 20,000 | 725ms | 21ms |

The full rewrite grows with total chunks. The segmented write grows with
*distinct live sources*, not chunks, because the manifest is rewritten each
time: the 20k row above holds 2,500 sources. That is a far shallower slope
and the file is small and compact-encoded, but it is not flat, and it is the
number to watch if the revisit trigger in ADR-001 is ever approached.
