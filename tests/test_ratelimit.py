"""Per-caller ingest rate limiting.

The unit tests drive the bucket with an injected clock rather than
sleeping, so they assert the refill arithmetic exactly instead of racing
wall time. The HTTP tests only need to show that the limit is wired to
the right endpoints and charged the right amount.
"""
import pytest
from fastapi.testclient import TestClient

from scout.api import app
from scout.config import reset_config_cache
from scout.ratelimit import TokenBucketLimiter

HTML = (
    "<html><head><title>Rate Limit Page</title></head><body><article>"
    "<p>A page with enough words in it to survive boilerplate removal "
    "and produce at least one usable chunk of content.</p>"
    "</article></body></html>"
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def limited_client(monkeypatch):
    """A running app that allows two ingests per minute."""
    monkeypatch.setenv("SCOUT_INGEST_RATE_LIMIT", "2")
    monkeypatch.setenv("SCOUT_INGEST_RATE_WINDOW", "60")
    reset_config_cache()
    with TestClient(app) as client:
        yield client
    reset_config_cache()


def _ingest(client, slug):
    return client.post(
        "/api/v1/ingest", json={"html": HTML, "source_url": f"https://example.com/{slug}"}
    )


# -- unit ---------------------------------------------------------------------

def test_allows_up_to_the_limit_then_refuses():
    limiter = TokenBucketLimiter(3, 60, clock=FakeClock())
    assert [limiter.check("caller") for _ in range(3)] == [None, None, None]
    assert limiter.check("caller") is not None


def test_callers_have_independent_buckets():
    limiter = TokenBucketLimiter(1, 60, clock=FakeClock())
    assert limiter.check("a") is None
    assert limiter.check("b") is None
    assert limiter.check("a") is not None


def test_retry_after_is_the_actual_wait():
    """3 per 60s refills one token every 20s, so an exhausted bucket is one
    request away from usable in exactly 20s."""
    clock = FakeClock()
    limiter = TokenBucketLimiter(3, 60, clock=clock)
    for _ in range(3):
        limiter.check("caller")

    assert limiter.check("caller") == pytest.approx(20.0)
    clock.advance(20.0)
    assert limiter.check("caller") is None


def test_a_refused_request_costs_nothing():
    """An impatient client retrying in a loop must not push its own
    deadline further out with every attempt: the wait it's told to serve
    stays the same, and serving it works."""
    clock = FakeClock()
    limiter = TokenBucketLimiter(1, 60, clock=clock)
    limiter.check("caller")

    wait = limiter.check("caller")
    assert [limiter.check("caller") for _ in range(5)] == [wait] * 5

    clock.advance(wait)
    assert limiter.check("caller") is None


def test_burst_is_capped_at_the_limit():
    """Idling for an hour shouldn't bank an hour's worth of requests."""
    clock = FakeClock()
    limiter = TokenBucketLimiter(2, 60, clock=clock)
    clock.advance(3600)
    assert [limiter.check("c") for _ in range(3)] == [None, None, pytest.approx(30.0)]


def test_a_cost_above_the_limit_is_clamped_not_rejected_forever():
    """A batch larger than the whole bucket could otherwise never succeed,
    no matter how long the caller waited."""
    limiter = TokenBucketLimiter(2, 60, clock=FakeClock())
    assert limiter.check("caller", cost=5) is None
    assert limiter.check("caller") is not None


def test_limit_of_zero_disables_the_limiter():
    limiter = TokenBucketLimiter(0, 60, clock=FakeClock())
    assert all(limiter.check("caller", cost=99) is None for _ in range(50))


def test_idle_buckets_are_evicted():
    """The bucket dict is keyed by caller identity, which for an
    unauthenticated deployment means every client IP ever seen. Without
    eviction that's an unbounded map driven by whoever is talking to
    Scout."""
    clock = FakeClock()
    limiter = TokenBucketLimiter(5, 60, clock=clock)
    limiter.check("old-caller")
    clock.advance(60 * 5)
    limiter.check("new-caller")

    assert "old-caller" not in limiter._buckets
    assert "new-caller" in limiter._buckets


# -- HTTP ---------------------------------------------------------------------

def test_ingest_is_limited_and_says_when_to_retry(limited_client):
    assert _ingest(limited_client, "one").status_code == 200
    assert _ingest(limited_client, "two").status_code == 200

    refused = _ingest(limited_client, "three")
    assert refused.status_code == 429
    assert int(refused.headers["retry-after"]) > 0


def test_batch_ingest_is_charged_per_item(limited_client):
    """Otherwise /ingest/batch is a way to make N outbound fetches for the
    price of one request."""
    resp = limited_client.post("/api/v1/ingest/batch", json={
        "items": [
            {"html": HTML, "source_url": "https://example.com/a"},
            {"html": HTML, "source_url": "https://example.com/b"},
        ]
    })
    assert resp.status_code == 200
    assert _ingest(limited_client, "after-batch").status_code == 429


def test_search_is_not_rate_limited(limited_client):
    """Search is local CPU work already bounded by the request-shape caps;
    only ingest spends Scout's network on the caller's behalf."""
    for _ in range(5):
        assert limited_client.post(
            "/api/v1/search", json={"query": "rate limit"}
        ).status_code == 200


def test_the_limit_is_off_by_default():
    """The default config allows 30/minute, so the handful of ingests the
    rest of the suite does must not trip it."""
    reset_config_cache()
    with TestClient(app) as client:
        for i in range(5):
            assert _ingest(client, f"default-{i}").status_code == 200
