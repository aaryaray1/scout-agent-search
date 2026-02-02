import re

def tokenize(text):
    return set(re.findall(r"\w+", text.lower()))

def keyword_score(query, document):
    q_tokens = tokenize(query)
    d_tokens = tokenize(document)

    if not q_tokens or not d_tokens:
        return 0.0

    overlap = q_tokens.intersection(d_tokens)
    return len(overlap) / len(q_tokens)
