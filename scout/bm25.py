"""BM25 keyword relevance scoring over an incremental inverted index.

Replaces the token-overlap scoring scout/keyword.py used to provide (raw
set intersection: no notion of how rare a term is across the corpus, or
how many times it appears in a given document) with real Okapi BM25:
term-frequency weighted, and normalized against corpus-wide term rarity
and document length.

Implemented directly rather than pulling in a search library like whoosh
(the dependency ROADMAP.md originally floated for this): the formula
itself is compact, this way there's no dependency of uncertain modern
Python support, and every step of it is under test below.

**Why an inverted index rather than the per-document scan this started
as.** The first version stored one Counter per document and, on every
query, looped over every document in the corpus. Worse, it had no way to
add or remove a document, so Retriever rebuilt the whole index -
re-tokenizing every chunk - on every single ingest. Measured at 20k
chunks that rebuild cost 3.4 seconds per ingest, against 11ms for the
vector scan the roadmap had proposed replacing instead. See
docs/architecture/adr-001-incremental-index.md.

The index is therefore keyed the other way round, `term -> {slot: tf}`,
which makes three things cheap that were not:

- adding documents costs the terms in those documents, not the corpus
- removing them costs the same, since a slot can be popped out of each of
  its terms' posting lists directly
- scoring touches only documents that contain a query term, instead of
  every document

**Slots.** A document's position in this index is called a slot and never
changes while it is live, because postings hold slot numbers. Removing a
document leaves its slot behind as a tombstone: `scores()` still returns
an entry for it (always 0.0) so the returned list stays aligned with the
caller's own view. compact() reclaims tombstoned slots by renumbering,
which is the only operation that invalidates slot numbers, and it does so
without re-tokenizing anything.

**Thread safety: none, by design.** This index is mutated in place, so a
reader iterating a posting list while a writer inserts into it would
raise RuntimeError. Callers serialize access; see Retriever's class
docstring for why a lock is affordable here and copy-on-write is not.
"""
import math
import re
from collections import Counter

import numpy as np

K1 = 1.5  # term-frequency saturation: higher values let repeated terms
B = 0.75  # keep adding weight for longer before diminishing returns
          # document-length normalization: 0 = ignore length, 1 = full

_WORD_RE = re.compile(r"\w+")


def term_counts(text):
    """Tokenize `text` into a Counter of lowercase word -> frequency."""
    return Counter(_WORD_RE.findall(text.lower()))


class BM25Index:
    """BM25 relevance scoring over a list of corpus chunk dicts.

    Documents are addressed by slot (see the module docstring). Slots are
    handed out in insertion order, so an index that has never had anything
    removed has slot == position, which is what lets a caller that only
    ever appends ignore slots entirely.
    """

    def __init__(self, corpus=None):
        self._reset()
        if corpus:
            self.build(corpus)

    def _reset(self):
        self._postings = {}      # term -> {slot: term frequency}
        self._doc_terms = []     # slot -> Counter, or None once removed
        self._doc_lengths = []   # slot -> token count, 0 once removed
        self._live_docs = 0
        self._total_length = 0

    # -- shape ---------------------------------------------------------------

    @property
    def size(self):
        """Number of slots, live and tombstoned. This is the length of the
        list scores() returns."""
        return len(self._doc_terms)

    @property
    def live_docs(self):
        """Number of documents actually in the index."""
        return self._live_docs

    @property
    def tombstones(self):
        return self.size - self._live_docs

    @property
    def _avg_doc_length(self):
        """Derived rather than stored: it moves on every add and remove, and
        a stale copy would silently skew every length normalization."""
        return self._total_length / self._live_docs if self._live_docs else 0.0

    # -- mutation ------------------------------------------------------------

    def build(self, corpus):
        """Replace the index contents with `corpus`, resetting slots."""
        self._reset()
        self.add_documents(corpus)

    def add_documents(self, docs):
        """Index `docs`, returning the slots assigned to them in order.

        Costs the terms in `docs`, not the size of the corpus, which is the
        whole reason this method exists.
        """
        slots = []
        postings = self._postings
        for doc in docs:
            slot = len(self._doc_terms)
            counts = term_counts(doc["content"])
            length = sum(counts.values())

            self._doc_terms.append(counts)
            self._doc_lengths.append(length)
            for term, frequency in counts.items():
                # get-then-create rather than setdefault: setdefault
                # evaluates its default eagerly, so it allocates a throwaway
                # dict for every (document, term) pair even though almost
                # all of them find an existing posting list.
                bucket = postings.get(term)
                if bucket is None:
                    bucket = postings[term] = {}
                bucket[slot] = frequency

            self._live_docs += 1
            self._total_length += length
            slots.append(slot)
        return slots

    def remove_slots(self, slots):
        """Tombstone `slots`, dropping their postings.

        The slot numbers themselves are not reused, so every other
        document's slot stays valid and the caller's mapping survives.
        Removing an already-removed slot is a no-op rather than an error,
        since a caller replacing two sources that share a chunk shouldn't
        have to de-duplicate first.
        """
        for slot in slots:
            counts = self._doc_terms[slot]
            if counts is None:
                continue
            for term in counts:
                postings = self._postings.get(term)
                if postings is None:
                    continue
                postings.pop(slot, None)
                # A term nothing contains any more would otherwise sit in
                # the index forever inflating nothing but memory, and would
                # make len(postings) == 0 a case _idf has to handle.
                if not postings:
                    del self._postings[term]

            self._total_length -= self._doc_lengths[slot]
            self._doc_lengths[slot] = 0
            self._doc_terms[slot] = None
            self._live_docs -= 1

    def compacted(self):
        """Return a new index holding only the live documents, renumbered.

        Invalidates every slot number the caller holds: in the result, live
        documents occupy slots 0..live_docs-1 in the order they were added,
        so a caller whose own view is in that same order can simply reset
        its mapping to a range.

        A new object rather than a mutation in place, because re-keying
        every posting list is proportional to the whole index rather than
        to what changed. Building it off to the side lets a caller do that
        work without excluding readers, and swap the result in afterwards.

        Cheap relative to a rebuild because the per-document term counts
        are reused as they are; nothing is tokenized again. Those counters
        end up shared with this index, which is safe because nothing ever
        mutates one after it is built -- documents are replaced wholesale,
        never edited.
        """
        survivors = self._live_slots()
        renumbered = {old: new for new, old in enumerate(survivors)}

        compacted = BM25Index()
        compacted._postings = self._renumbered_postings(renumbered)
        compacted._doc_terms = [self._doc_terms[slot] for slot in survivors]
        compacted._doc_lengths = [self._doc_lengths[slot] for slot in survivors]
        compacted._live_docs = len(survivors)
        # Removals already decremented this, so it is the live total.
        compacted._total_length = self._total_length
        return compacted

    def _live_slots(self):
        """Slots still holding a document, in order."""
        return [
            slot for slot, counts in enumerate(self._doc_terms) if counts is not None
        ]

    def _renumbered_postings(self, renumbered):
        """Re-key every posting list through `renumbered`.

        Postings only ever hold live slots, because remove_slots pops them
        as it goes, so every key here is guaranteed to be present in the
        mapping.
        """
        return {
            term: {renumbered[slot]: frequency for slot, frequency in postings.items()}
            for term, postings in self._postings.items()
        }

    # -- querying ------------------------------------------------------------

    def _idf(self, term):
        # Document frequency is the length of the posting list, so there is
        # no separate counter to keep in sync with it.
        n_t = len(self._postings.get(term, ()))
        # +1 smoothed variant: keeps IDF non-negative even for a term that
        # appears in every document, instead of going negative like the
        # classic Robertson-Sparck Jones formula can.
        return math.log(1 + (self._live_docs - n_t + 0.5) / (n_t + 0.5))

    def _accumulate(self, query_terms, avg_doc_length):
        """Sum each query term's BM25 contribution into a per-slot list.

        Walks posting lists, so the cost is the number of (term, document)
        pairs the query actually matches rather than the size of the
        corpus. Constants are hoisted out of the inner loop because that
        loop is the hot path for every search Scout serves.
        """
        scores = [0.0] * len(self._doc_terms)
        lengths = self._doc_lengths
        length_floor = K1 * (1 - B)
        length_scale = K1 * B / avg_doc_length
        numerator_scale = K1 + 1

        for term in query_terms:
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf(term)
            for slot, frequency in postings.items():
                length_norm = length_floor + length_scale * lengths[slot]
                scores[slot] += idf * frequency * numerator_scale / (frequency + length_norm)
        return scores

    def scores(self, query):
        """Return one BM25 score per slot, in slot order.

        0.0 for documents sharing no term with the query, and 0.0 for
        tombstoned slots, which are kept in the output so the list stays
        aligned with a caller's own slot mapping.
        """
        if not self._doc_terms:
            return []
        avg_doc_length = self._avg_doc_length
        query_terms = set(term_counts(query))
        if not query_terms or avg_doc_length == 0:
            return [0.0] * len(self._doc_terms)
        return self._accumulate(query_terms, avg_doc_length)

    def scores_array(self, query):
        """scores() as a numpy array, for callers that fuse it with vector
        similarities."""
        return np.asarray(self.scores(query), dtype=float)


def normalize(scores):
    """Min-max scale `scores` to [0, 1] so they're comparable to the
    cosine similarity half of the hybrid score. Raw BM25 scores are
    unbounded (can run well past 1 for strong, rare-term matches), so
    fusing them in directly would swamp the vector_weight/keyword_weight
    balance in scout/config.json.
    """
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    if hi == lo:
        # No discriminating signal (including the common case where every
        # score is 0, i.e. nothing matched any query term).
        return [0.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def normalize_array(scores):
    """normalize() for numpy input, without the round trip through a list.

    Same scaling and the same all-equal special case; kept alongside
    normalize() rather than replacing it so a library caller holding plain
    lists isn't forced into numpy.
    """
    if scores.size == 0:
        return scores
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)
