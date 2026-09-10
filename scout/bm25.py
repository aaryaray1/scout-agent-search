"""Okapi BM25 over an incremental inverted index (term -> {slot: tf}).

Documents are addressed by slot, which survives removal. Not thread-safe by
design: it is mutated in place and callers serialize. See
docs/design/retrieval.md.
"""
import math
import re
from collections import Counter

import numpy as np

K1 = 1.5  # term-frequency saturation
B = 0.75  # document-length normalization: 0 = ignore length, 1 = full

_WORD_RE = re.compile(r"\w+")


def term_counts(text):
    """Tokenize `text` into a Counter of lowercase word -> frequency."""
    return Counter(_WORD_RE.findall(text.lower()))


class BM25Index:
    """BM25 scoring over a list of corpus chunk dicts.

    Slots are handed out in insertion order, so an index that has never had
    anything removed has slot == position.
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
        """Slots, live and tombstoned. The length of what scores() returns."""
        return len(self._doc_terms)

    @property
    def live_docs(self):
        return self._live_docs

    @property
    def tombstones(self):
        return self.size - self._live_docs

    @property
    def _avg_doc_length(self):
        # Derived, not stored: it moves on every add and remove, and a stale
        # copy would skew every length normalization.
        return self._total_length / self._live_docs if self._live_docs else 0.0

    # -- mutation ------------------------------------------------------------

    def build(self, corpus):
        """Replace the index contents with `corpus`, resetting slots."""
        self._reset()
        self.add_documents(corpus)

    def add_documents(self, docs):
        """Index `docs`, returning their assigned slots in order.

        Costs the terms in `docs`, not the size of the corpus.
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
                # get-then-create, not setdefault, which would allocate a
                # throwaway dict for every (document, term) pair.
                bucket = postings.get(term)
                if bucket is None:
                    bucket = postings[term] = {}
                bucket[slot] = frequency

            self._live_docs += 1
            self._total_length += length
            slots.append(slot)
        return slots

    def remove_slots(self, slots):
        """Tombstone `slots` and drop their postings, leaving every other
        slot number valid. Removing an already-removed slot is a no-op."""
        for slot in slots:
            counts = self._doc_terms[slot]
            if counts is None:
                continue
            for term in counts:
                postings = self._postings.get(term)
                if postings is None:
                    continue
                postings.pop(slot, None)
                # An empty posting list would sit there forever and make
                # len(postings) == 0 a case _idf has to handle.
                if not postings:
                    del self._postings[term]

            self._total_length -= self._doc_lengths[slot]
            self._doc_lengths[slot] = 0
            self._doc_terms[slot] = None
            self._live_docs -= 1

    def compacted(self):
        """A new index holding only live documents, renumbered in insertion
        order, so a caller in that same order resets its mapping to a range.

        A new object because re-keying is O(index) and the caller wants that
        outside its read lock. Term counters are reused, not re-tokenized.
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
        return [
            slot for slot, counts in enumerate(self._doc_terms) if counts is not None
        ]

    def _renumbered_postings(self, renumbered):
        # Postings only ever hold live slots, since remove_slots pops them.
        return {
            term: {renumbered[slot]: frequency for slot, frequency in postings.items()}
            for term, postings in self._postings.items()
        }

    # -- querying ------------------------------------------------------------

    def _idf(self, term):
        # Document frequency is the posting list length, so there is no
        # separate counter to keep in sync.
        n_t = len(self._postings.get(term, ()))
        # +1 smoothed: stays non-negative for a term in every document.
        return math.log(1 + (self._live_docs - n_t + 0.5) / (n_t + 0.5))

    def _accumulate(self, query_terms, avg_doc_length):
        """Sum each query term's BM25 contribution into a per-slot list.

        Walks posting lists, so cost is the (term, document) pairs the query
        matches. Constants are hoisted: this is the hot path for every search.
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
        """One BM25 score per slot, in slot order.

        Tombstoned slots score 0.0 and stay in the output, so the list stays
        aligned with a caller's slot mapping.
        """
        if not self._doc_terms:
            return []
        avg_doc_length = self._avg_doc_length
        query_terms = set(term_counts(query))
        if not query_terms or avg_doc_length == 0:
            return [0.0] * len(self._doc_terms)
        return self._accumulate(query_terms, avg_doc_length)

    def scores_array(self, query):
        """scores() as a numpy array, for fusing with vector similarities."""
        return np.asarray(self.scores(query), dtype=float)


def normalize(scores):
    """Min-max scale to [0, 1] so raw, unbounded BM25 scores are comparable
    to cosine similarity before fusing."""
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    if hi == lo:
        # No discriminating signal, including the common all-zero case.
        return [0.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def normalize_array(scores):
    """normalize() for numpy input, kept alongside it so a library caller
    holding plain lists isn't forced into numpy."""
    if scores.size == 0:
        return scores
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)
