"""Per-caller rate limiting for the ingest endpoints.

Ingest is the endpoint that spends Scout's network on caller-chosen hosts.
See docs/design/api.md for why a token bucket and not a fixed window.
"""
import logging
import threading
import time

logger = logging.getLogger(__name__)

# A bucket full and untouched this many windows carries no information, so
# dropping it keeps the map from growing once per caller ever seen.
_IDLE_EVICTION_WINDOWS = 2


class TokenBucketLimiter:
    """`limit` tokens per `window` seconds, refilled continuously.

    A caller may burst up to `limit`, then proceeds at the sustained rate.
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
        """Drop buckets that refilled to full and went quiet, on the write
        path, since the alternative is an unbounded map of every caller."""
        cutoff = self.window * _IDLE_EVICTION_WINDOWS
        stale = [i for i, (_, last) in self._buckets.items() if now - last > cutoff]
        for identity in stale:
            del self._buckets[identity]

    def check(self, identity, cost=1):
        """Spend `cost` tokens, returning None if allowed or the seconds to
        wait if not. A rejected request spends nothing."""
        if not self.enabled:
            return None
        # More than a whole bucket could never succeed however long the
        # caller waits, so clamp and answer one window.
        cost = min(cost, self.limit)

        with self._lock:
            now = self._clock()
            self._evict_idle(now)
            tokens = self._refilled(identity, now)
            if tokens < cost:
                return round((cost - tokens) / self._rate, 3)
            self._buckets[identity] = (tokens - cost, now)
            return None
