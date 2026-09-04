"""BM25 keyword relevance scoring.

Replaces the token-overlap scoring scout/keyword.py used to provide (raw
set intersection: no notion of how rare a term is across the corpus, or
how many times it appears in a given document) with real Okapi BM25:
term-frequency weighted, and normalized against corpus-wide term rarity
and document length.

Implemented directly rather than pulling in a search library like whoosh
(the dependency ROADMAP.md originally floated for this): the formula
itself is compact, this way there's no dependency of uncertain modern
Python support, and every step of it is under test below.
"""
import math
import re
from collections import Counter

K1 = 1.5  # term-frequency saturation: higher values let repeated terms
B = 0.75  # keep adding weight for longer before diminishing returns
          # document-length normalization: 0 = ignore length, 1 = full

_WORD_RE = re.compile(r"\w+")


def term_counts(text):
    """Tokenize `text` into a Counter of lowercase word -> frequency."""
    return Counter(_WORD_RE.findall(text.lower()))


class BM25Index:
    """BM25 relevance scoring over a list of corpus chunk dicts.

    Rebuilt from scratch on every build() call -- consistent with how
    Retriever already rebuilds self.corpus/corpus_embeddings by
    concatenating its docs/ingested layers on every corpus change (see
    ROADMAP.md's incremental-indexing note); true incremental index
    updates are future work, not a regression introduced here.
    """

    def __init__(self, corpus=None):
        self._doc_term_counts = []
        self._doc_lengths = []
        self._doc_freq = Counter()  # term -> number of docs containing it
        self._n_docs = 0
        self._avg_doc_length = 0.0
        if corpus:
            self.build(corpus)

    def build(self, corpus):
        self._doc_term_counts = [term_counts(d["content"]) for d in corpus]
        self._doc_lengths = [sum(counts.values()) for counts in self._doc_term_counts]
        self._n_docs = len(corpus)
        self._avg_doc_length = (
            sum(self._doc_lengths) / self._n_docs if self._n_docs else 0.0
        )
        self._doc_freq = Counter()
        for counts in self._doc_term_counts:
            self._doc_freq.update(counts.keys())  # each doc counts once per term

    def _idf(self, term):
        n_t = self._doc_freq.get(term, 0)
        # +1 smoothed variant: keeps IDF non-negative even for a term that
        # appears in every document, instead of going negative like the
        # classic Robertson-Sparck Jones formula can.
        return math.log(1 + (self._n_docs - n_t + 0.5) / (n_t + 0.5))

    def _length_norm(self, length):
        """BM25's document-length penalty, which depends only on the
        document -- not on the query term -- so it's computed once per
        document rather than once per term."""
        return K1 * (1 - B + B * length / self._avg_doc_length)

    def _score_doc(self, counts, length, idfs):
        """Sum the BM25 contribution of every query term present in one
        document. `idfs` is pre-computed per query, not per document."""
        length_norm = self._length_norm(length)
        score = 0.0
        for term, idf in idfs.items():
            f = counts.get(term, 0)
            if f:
                score += idf * (f * (K1 + 1)) / (f + length_norm)
        return score

    def scores(self, query):
        """Return one BM25 score per corpus position, in build()'s order.
        0.0 for documents sharing no term with the query."""
        if not self._n_docs:
            return []
        query_terms = set(term_counts(query).keys())
        if not query_terms or self._avg_doc_length == 0:
            return [0.0] * self._n_docs

        idfs = {term: self._idf(term) for term in query_terms}
        return [
            self._score_doc(counts, length, idfs)
            for counts, length in zip(self._doc_term_counts, self._doc_lengths)
        ]


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
