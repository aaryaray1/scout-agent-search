"""Per-caller rate limiting for the ingest endpoints.

Search is local CPU work already bounded by the request-shape caps in
scout/models.py. Ingest is different in kind: it spends Scout's own
network, on hosts the caller chooses, at whatever rate the caller asks
for. That's the endpoint ROADMAP.md Phase 3 singled out, and it's the one
limited here.

A token bucket rather than a fixed window, for two reasons: a window
boundary lets a caller send 2x the limit across it (all of one window's
budget, then all of the next), and a bucket refilling continuously gives
an exact answer to "how long until this succeeds" for the Retry-After
header, instead of "some time before the window flips".

In-process and therefore per-worker: running four uvicorn workers means
four buckets and four times the configured limit. Enough for a
single-process deployment, which is what Scout is today; a shared store
(Redis) is what a multi-worker deployment would need, and belongs with
the Phase 2 storage work rather than hidden in here.
"""
import logging
import threading
import time

logger = logging.getLogger(__name__)

# A bucket that has been full and untouched for this multiple of its
# refill window carries no information -- it's indistinguishable from a
# caller that has never been seen. Dropping those keeps the dict from
# growing once per distinct caller forever.
_IDLE_EVICTION_WINDOWS = 2


class TokenBucketLimiter:
    """Refilling token bucket per caller identity.

    `limit` tokens per `window` seconds, refilled continuously. A caller
    may burst up to `limit` and then proceeds at the sustained rate.
    limit=0 disables the limiter entirely.
    """

    def __init__(self, limit, window, clock=time.monotonic):
        self.limit = limit
        self.window = window
        self._clock = clock
        self._rate = limit / window if window else 0.0
        self._buckets = {}  # identity -> (tokens, last_refill_time)
        self._lock = threading.Lock()

    @property
    def enabled(self):
        return self.limit > 0

    def _refilled(self, identity, now):
        """Current token count for `identity`, brought up to date."""
        tokens, last = self._buckets.get(identity, (float(self.limit), now))
        return min(float(self.limit), tokens + (now - last) * self._rate)

    def _evict_idle(self, now):
        """Drop buckets that have refilled to full and gone quiet.

        Called on the write path rather than from a timer: without it the
        bucket dict is an unbounded map of every caller identity ever
        seen, which for an unauthenticated deployment means every client
        IP -- a slow memory leak driven by whoever is talking to Scout.
        """
        cutoff = self.window * _IDLE_EVICTION_WINDOWS
        stale = [i for i, (_, last) in self._buckets.items() if now - last > cutoff]
        for identity in stale:
            del self._buckets[identity]

    def check(self, identity, cost=1):
        """Spend `cost` tokens for `identity`.

        Returns None if the request is allowed, or the number of seconds
        to wait before it would be (the Retry-After value) if it isn't.
        Nothing is spent on a rejected request, so a caller that keeps
        retrying early doesn't push its own deadline further out.
        """
        if not self.enabled:
            return None
        # A single request asking for more than the whole bucket could
        # never succeed, no matter how long the caller waits; charging the
        # full bucket and reporting one window is the honest answer.
        cost = min(cost, self.limit)

        with self._lock:
            now = self._clock()
            self._evict_idle(now)
            tokens = self._refilled(identity, now)
            if tokens < cost:
                return round((cost - tokens) / self._rate, 3)
            self._buckets[identity] = (tokens - cost, now)
            return None
