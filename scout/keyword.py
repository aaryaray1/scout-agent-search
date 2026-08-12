import re


def tokenize(text):
    return set(re.findall(r"\w+", text.lower()))


def score_tokens(q_tokens, d_tokens):
    """Overlap score between two already-tokenized sets.

    Callers that score one query against many documents (Retriever.search)
    should tokenize each side once and call this directly, rather than
    re-tokenizing the same text on every comparison.
    """
    if not q_tokens or not d_tokens:
        return 0.0
    overlap = q_tokens & d_tokens
    return len(overlap) / len(q_tokens)


def keyword_score(query, document):
    """Convenience one-off wrapper: tokenizes both sides and scores them."""
    return score_tokens(tokenize(query), tokenize(document))
