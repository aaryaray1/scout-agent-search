"""API-key authentication for /api/v1: a shared secret checked in constant
time, off by default. See docs/design/api.md for what it does not cover.
"""
import hashlib
import logging
import secrets

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

API_KEY_HEADER = "X-API-Key"

# auto_error=False so a missing header reaches identify() rather than
# FastAPI raising a 403 an unauthenticated deployment must not send.
api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


def fingerprint(key):
    """Short non-reversible handle for a key, so logs and the rate limiter
    never hold the secret itself."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


class ApiKeyAuth:
    """Turns a request into a caller identity: `key:<fingerprint>` when
    authenticated, `ip:<address>` when auth is off."""

    def __init__(self, api_keys):
        # Stripped here too: a list can arrive from config.json, which does
        # none, and a whitespace-only key would match a blank header.
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
        """Constant-time compare against every key.

        Every key is checked even after a match, since returning early
        leaks key ordering to anyone timing it. Compared as bytes because
        compare_digest raises on non-ASCII str, which a header can be.
        """
        presented_bytes = presented.encode("utf-8")
        matched = False
        for key in self._keys:
            if secrets.compare_digest(presented_bytes, key):
                matched = True
        return matched

    def identify(self, presented, client_host):
        """Return the caller identity, or raise 401 if the key is bad."""
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
    """Best-effort caller address, used only as the unauthenticated fallback
    bucket -- behind a proxy every caller looks like the proxy."""
    return request.client.host if request.client else "unknown"


def require_api_key(request: Request, presented: str = Security(api_key_header)) -> str:
    """FastAPI dependency: authenticate, return the caller identity.

    Security() rather than Depends() so the header publishes as a security
    scheme in /openapi.json.
    """
    auth = request.app.state.auth
    return auth.identify(presented, client_host(request))
