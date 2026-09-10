"""Benchmark the index paths that Phase 2 is about.

The numbers in ROADMAP.md and docs/architecture/adr-001-incremental-index.md
came from this script, so they can be re-checked rather than trusted.

Deliberately does not load the embedding model. The costs being measured
here are tokenization, posting-list maintenance, the similarity scan and the
store write; an embedding model would add a large constant to every row and
make the shape of the growth harder to see. Vector similarity is measured
against random vectors of the right width for the same reason.

Word frequencies follow a Zipf distribution rather than a uniform one.
Drawing 400 words uniformly from an 8k vocabulary yields a chunk with about
390 distinct terms, which no real document has; Zipf yields about 237, which
is in the right range for English prose. The distribution matters here
because it sets both how much work indexing a document is and how long the
posting lists a query walks are, and uniform draws flatter the query numbers
while inflating the indexing ones.

Timing takes the minimum of several trials with the collector disabled
during each. These structures allocate heavily enough that a GC pause landing
inside a timed region moves the result by more than the thing being measured.

Absolute figures are machine-state dependent and are not comparable across
runs: measured on a busy desktop, every column here (including the ones no
Scout code touches) runs roughly 3x slower than on an idle one. Compare
columns within a single run, not numbers between runs. The ratio between the
cold-build column and the add column is the load-bearing result, and it holds
at roughly three orders of magnitude regardless of machine state.

    python scripts/bench_index.py
    python scripts/bench_index.py --sizes 1000,5000,20000,100000
"""
import argparse
import gc
import json
import os
import random
import string
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scout.bm25 import BM25Index  # noqa: E402

CHUNK_WORDS = 400  # matches the default chunk_size in scout/config.py
VOCAB_SIZE = 8000
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2
QUERY_TERMS = 6
PAGE_CHUNKS = 8  # a plausible number of chunks for one ingested web page
TRIALS = 3
REPEATS = 10


def _vocabulary(rng):
    return [
        "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 9)))
        for _ in range(VOCAB_SIZE)
    ]


def _zipf_weights(size):
    """Weight of rank r proportional to 1/r, which is roughly how word
    frequency distributes in natural language."""
    return [1.0 / (rank + 1) for rank in range(size)]


def _corpus(rng, vocab, weights, n_chunks):
    return [
        {
            "content": " ".join(rng.choices(vocab, weights=weights, k=CHUNK_WORDS)),
            "source": f"https://example.com/page-{i // PAGE_CHUNKS}",
        }
        for i in range(n_chunks)
    ]


def _best(fn, repeats=1):
    """Fastest of TRIALS runs, in milliseconds, with GC held off inside each
    timed region. Minimum rather than mean: the noise here is all additive
    (a GC pause, the scheduler), so the fastest run is the closest to the
    real cost.
    """
    timings = []
    for _ in range(TRIALS):
        gc.collect()
        gc.disable()
        try:
            start = time.perf_counter()
            for _ in range(repeats):
                fn()
            timings.append((time.perf_counter() - start) / repeats * 1000)
        finally:
            gc.enable()
    return min(timings)


def _time_one_add(index, page):
    """Time one add, then undo it so the next trial measures the same index.

    Repeating an add against an index without undoing it grows the index,
    so each repeat is measured against a bigger corpus than the last and
    the result describes an index that never existed. Rebuilding a fresh
    index per trial would fix that but allocates several more copies of the
    corpus, and at 20k chunks that costs more in collector pressure than
    the measurement is worth.

    Tombstoning the slots that were just added is enough to undo it: it
    restores the live document count and the corpus statistics an add
    depends on, and leaves behind only a handful of dead slots, which an
    add never touches.
    """
    gc.collect()
    gc.disable()
    try:
        start = time.perf_counter()
        slots = index.add_documents(page)
        elapsed = (time.perf_counter() - start) * 1000
    finally:
        gc.enable()
    index.remove_slots(slots)
    return elapsed


def _bench_keyword(corpus, query):
    """Cold build, incremental add, and one query.

    Cold build is paid once at startup. Incremental add is what an ingest
    pays, and is the number this whole phase exists to move: before the
    inverted index, an ingest paid the cold build column instead.
    """
    cold_build = _best(lambda: BM25Index(corpus))

    # Measured on an index holding exactly `corpus`, before any add has had
    # a chance to grow it.
    index = BM25Index(corpus)
    query_time = _best(lambda: index.scores(query), repeats=REPEATS)

    page = corpus[:PAGE_CHUNKS]
    add_time = min(_time_one_add(index, page) for _ in range(TRIALS))
    return cold_build, add_time, query_time


def _bench_vector(n_chunks):
    embeddings = np.random.rand(n_chunks, EMBEDDING_DIM).astype(np.float32)
    vector = np.random.rand(EMBEDDING_DIM).astype(np.float32)

    def scan():
        denom = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(vector)
        denom[denom == 0] = 1e-8
        return (embeddings @ vector) / denom

    return _best(scan, repeats=REPEATS), embeddings


def _bench_store_write(corpus, embeddings):
    """What save_ingested() costs today: the whole store, every ingest.

    This is the remaining O(everything) step, and the next Phase 2 item.
    """
    with tempfile.TemporaryDirectory() as directory:
        # The directory is created and torn down outside the timed region.
        # Leaving it inside measured mkdir plus a recursive delete of a
        # 30MB file alongside the write, which is not what save_ingested
        # costs. Rewriting the same paths each trial also matches what the
        # real store does.
        embeddings_path = os.path.join(directory, "ingested_embeddings.npy")
        meta_path = os.path.join(directory, "ingested_meta.json")

        def write():
            np.save(embeddings_path, embeddings)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(corpus, f, indent=2)

        return _best(write)


def _row(n_chunks, rng, vocab, weights):
    corpus = _corpus(rng, vocab, weights, n_chunks)
    query = " ".join(rng.choices(vocab, weights=weights, k=QUERY_TERMS))

    cold_build, add, query_time = _bench_keyword(corpus, query)
    vector_time, embeddings = _bench_vector(n_chunks)
    write_time = _bench_store_write(corpus, embeddings)

    return (
        f"| {n_chunks:,} | {cold_build:,.0f}ms | {add:.2f}ms | {query_time:.1f}ms "
        f"| {vector_time:.1f}ms | {write_time:,.0f}ms |"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", default="1000,5000,20000", help="comma-separated corpus sizes, in chunks"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    vocab = _vocabulary(rng)
    weights = _zipf_weights(len(vocab))

    print(
        f"{CHUNK_WORDS}-word Zipf-distributed chunks, {VOCAB_SIZE}-term vocabulary, "
        f"{EMBEDDING_DIM}-dim vectors, best of {TRIALS}\n"
    )
    print("| chunks | keyword cold build | keyword add (per ingest) | keyword query "
          "| vector query | full store rewrite (per ingest) |")
    print("| --- | --- | --- | --- | --- | --- |")
    for size in (int(s) for s in args.sizes.split(",")):
        print(_row(size, rng, vocab, weights))


if __name__ == "__main__":
    main()
