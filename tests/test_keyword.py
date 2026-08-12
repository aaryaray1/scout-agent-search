from scout.keyword import tokenize, score_tokens, keyword_score


def test_tokenize_lowercases_and_extracts_words():
    assert tokenize("Error 1008: Insufficient Balance!") == {
        "error", "1008", "insufficient", "balance",
    }


def test_tokenize_empty_string():
    assert tokenize("") == set()


def test_score_tokens_full_overlap():
    assert score_tokens({"a", "b"}, {"a", "b", "c"}) == 1.0


def test_score_tokens_partial_overlap():
    assert score_tokens({"a", "b"}, {"b", "c"}) == 0.5


def test_score_tokens_no_overlap():
    assert score_tokens({"a"}, {"b"}) == 0.0


def test_score_tokens_empty_query_or_doc():
    assert score_tokens(set(), {"a"}) == 0.0
    assert score_tokens({"a"}, set()) == 0.0


def test_keyword_score_matches_score_tokens():
    # keyword_score() is the one-off convenience wrapper; it should agree
    # with tokenizing both sides once and calling score_tokens directly,
    # which is what Retriever.search() does for repeated queries.
    query, document = "rate limit", "Rate limit exceeded"
    assert keyword_score(query, document) == score_tokens(
        tokenize(query), tokenize(document)
    )
