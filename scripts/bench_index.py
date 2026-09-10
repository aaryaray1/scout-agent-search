"""Benchmark the keyword index, the vector scan and the durable store.

Compare columns within a run, never numbers between runs: machine state
moves every column, including ones no Scout code touches. Methodology and
caveats: docs/design/benchmarks.md.

    python scripts/bench_index.py
    python scripts/bench_index.py --sizes 1000,5000,20000,100000
"""
import argparse
import gc
import itertools
import random
import string
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scout.bm25 import BM25Index  # noqa: E402
from scout.index import save_ingested  # noqa: E402
from scout.store import JsonChunkStore, SegmentedChunkStore  # noqa: E402

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
    """Fastest of TRIALS runs in ms, GC held off inside each timed region.
    Minimum, not mean: the noise here is all additive."""
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
    """Time one add, then tombstone it so the next trial measures the same
    index rather than one the previous repeat grew.

    Tombstoning restores the statistics an add depends on without
    reallocating the corpus, which a fresh index per trial would.
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


def _bench_store_write(store_class, corpus, embeddings):
    """One ingest against a store already holding `corpus`, through the same
    ChunkStore call for both backends so the columns are comparable.

    Each timed call uses a fresh source, as a real ingest does; repeating one
    would pile up dead chunks and eventually time a merge instead.
    """
    page = corpus[:PAGE_CHUNKS]
    page_embeddings = embeddings[:PAGE_CHUNKS]
    counter = itertools.count()

    with tempfile.TemporaryDirectory() as directory:
        # Seeded, and the directory created and torn down, outside the timed
        # region: leaving either inside would measure mkdir and a recursive
        # delete of a 30MB store alongside the write.
        save_ingested(embeddings, corpus, index_dir=directory)
        store = store_class(index_dir=directory)
        store.load()

        def write():
            source = f"https://example.com/new-{next(counter)}"
            store.upsert(
                [dict(chunk, source=source) for chunk in page], page_embeddings
            )

        return _best(write)


def _row(n_chunks, rng, vocab, weights):
    corpus = _corpus(rng, vocab, weights, n_chunks)
    query = " ".join(rng.choices(vocab, weights=weights, k=QUERY_TERMS))

    cold_build, add, query_time = _bench_keyword(corpus, query)
    vector_time, embeddings = _bench_vector(n_chunks)
    full_write = _bench_store_write(JsonChunkStore, corpus, embeddings)
    segment_write = _bench_store_write(SegmentedChunkStore, corpus, embeddings)

    return (
        f"| {n_chunks:,} | {cold_build:,.0f}ms | {add:.2f}ms | {query_time:.1f}ms "
        f"| {vector_time:.1f}ms | {full_write:,.0f}ms | {segment_write:.2f}ms |"
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
          "| vector query | store write, full rewrite | store write, segmented |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for size in (int(s) for s in args.sizes.split(",")):
        print(_row(size, rng, vocab, weights))


if __name__ == "__main__":
    main()
