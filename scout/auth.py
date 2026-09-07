"""API-key authentication for the /api/v1 surface.

ROADMAP.md Phase 3 called auth "the main blocker to public hosting", and
that's what this closes. The shape is deliberately small: a shared-secret
header, checked in constant time, configured through the environment.
There is no user model, no key issuance and no scopes -- Scout serves
agents on behalf of one operator, so a key answers "is this caller
allowed to spend my network and CPU", which is the only question the
service actually has.

Two properties matter more than the size of the mechanism:

- **Fails closed on a bad key, open only when explicitly unconfigured.**
  No keys configured means no auth, which is right for a laptop and wrong
  for anything reachable, so startup logs which of the two it is rather
  than leaving an operator to infer it.
- **Never logs or returns the key itself.** The rate limiter needs a
  stable per-caller identity, so callers are identified by a short hash
  of their key. A log line or an error body therefore can't leak the
  credential it's describing.
"""
import hashlib
import logging
import secrets

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

API_KEY_HEADER = "X-API-Key"

# auto_error=False so a missing header reaches identify() as None instead
# of FastAPI raising a 403 of its own: an unauthenticated deployment has
# to be allowed to serve the request, and an authenticated one wants to
# answer 401 with its own message.
api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


def fingerprint(key):
    """Short, stable, non-reversible handle for an API key.

    Used as the rate-limiter's bucket key and in log lines, so neither
    ever has to hold the secret itself.
    """
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


class ApiKeyAuth:
    """Checks the API-key header and turns a request into a caller identity.

    The identity is what the rate limiter buckets on: `key:<fingerprint>`
    when authenticated, `ip:<address>` when auth is disabled. Both are
    strings, so nothing downstream needs to know which mode is active.
    """

    def __init__(self, api_keys):
        # Stripped and emptied out here as well as in the config parser: a
        # key list can also arrive from config.json, which does no
        # stripping, and a whitespace-only "key" would otherwise be a live
        # credential that a blank header matches.
        self._keys = [k.strip().encode("utf-8") for k in (api_keys or []) if k.strip()]

    @property
    def enabled(self):
        return bool(self._keys)

    def log_status(self):
        if self.enabled:
            logger.info(
                "API-key auth enabled: %d key(s), header %s", len(self._keys), API_KEY_HEADER
            )
        else:
            logger.warning(
                "API-key auth is DISABLED (no api_keys configured); every caller can "
                "reach /api/v1. Set SCOUT_API_KEYS before exposing this beyond localhost."
            )

    def _matches(self, presented):
        """Constant-time comparison against every configured key.

        Every key is compared even after a match is found: returning as
        soon as one hits would make the response time depend on which key
        was presented, which leaks the key ordering to anyone timing it.
        """
        # Compared as bytes: compare_digest raises TypeError on str inputs
        # holding non-ASCII characters, and a header is caller-controlled.
        presented_bytes = presented.encode("utf-8")
        matched = False
        for key in self._keys:
            if secrets.compare_digest(presented_bytes, key):
                matched = True
        return matched

    def identify(self, presented, client_host):
        """Return the caller identity, or raise 401 if the key is bad.

        `presented` is whatever arrived in the header (possibly None).
        """
        if not self.enabled:
            return f"ip:{client_host}"
        presented = (presented or "").strip()
        if not presented:
            raise HTTPException(
                status_code=401,
                detail=f"missing {API_KEY_HEADER} header",
                headers={"WWW-Authenticate": API_KEY_HEADER},
            )
        if not self._matches(presented):
            logger.warning("rejected an invalid API key from %s", client_host)
            raise HTTPException(status_code=401, detail="invalid API key")
        return f"key:{fingerprint(presented)}"


def client_host(request):
    """Best-effort caller address, for identifying unauthenticated callers.

    Behind a proxy every caller looks like the proxy, which is exactly why
    this is only the fallback bucket: an authenticated deployment buckets
    on the key instead and doesn't depend on it.
    """
    return request.client.host if request.client else "unknown"


def require_api_key(request: Request, presented: str = Security(api_key_header)) -> str:
    """FastAPI dependency: authenticate the request, return its identity.

    Declared with Security() rather than Depends() so the header shows up
    as a security scheme in /openapi.json -- an agent reading the schema
    can see that it needs a key, the same way it can already read the
    request-shape limits off the models.
    """
    auth = request.app.state.auth
    return auth.identify(presented, client_host(request))
