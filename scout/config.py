"""Runtime configuration.

Layered so the same code runs from a checkout and from a container:
scout/config.json holds the committed defaults, and any key can be
overridden at deploy time with a SCOUT_-prefixed environment variable
(SCOUT_TOP_K, SCOUT_INDEX_DIR, ...). Hosting a service means changing its
settings without editing files inside the image, so the env layer wins.
"""
import json
import logging
import os
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "config.json"

ENV_PREFIX = "SCOUT_"

_DEFAULTS = {
    "vector_model": "all-MiniLM-L6-v2",
    "docs_path": "data/docs",
    "index_dir": "data/index",
    "top_k": 3,
    "vector_weight": 0.65,
    "keyword_weight": 0.35,
    "chunk_size": 400,
    "chunk_overlap": 50,
    # Request-shape caps. These bound what a single caller can ask Scout to
    # do; they are not a substitute for the auth and rate limiting tracked
    # in ROADMAP.md Phase 3.
    "max_top_k": 50,
    "max_query_chars": 2000,
    "max_html_bytes": 5 * 1024 * 1024,
    # Batch ceilings. A batch request costs Scout roughly its length, so
    # these bound fan-out the same way max_top_k bounds a single search.
    "max_batch_queries": 10,
    "max_batch_ingest": 5,
    # Auth. Empty means no authentication at all, which is the right
    # default for `uvicorn scout.api:app` on a laptop and the wrong one
    # for anything reachable. Startup says so out loud either way.
    "api_keys": [],
    # Per-caller rate limit on ingest, which is the endpoint that spends
    # Scout's network on the caller's behalf. 0 disables it.
    "ingest_rate_limit": 30,
    "ingest_rate_window": 60,
    "log_level": "INFO",
}


def _parse_key_list(raw):
    """Read a comma-separated env value into a list of non-empty items.

    Secrets arrive through the environment far more often than through a
    committed config.json, so SCOUT_API_KEYS has to be expressible as one
    string. An empty or whitespace-only value yields [], which is a
    deliberate "no auth", not a parse failure.
    """
    if isinstance(raw, list):
        return [str(k).strip() for k in raw if str(k).strip()]
    return [k.strip() for k in str(raw).split(",") if k.strip()]


# Environment variables arrive as strings; every non-string setting needs to
# say how to read itself back.
_PARSERS = {
    "top_k": int,
    "vector_weight": float,
    "keyword_weight": float,
    "chunk_size": int,
    "chunk_overlap": int,
    "max_top_k": int,
    "max_query_chars": int,
    "max_html_bytes": int,
    "max_batch_queries": int,
    "max_batch_ingest": int,
    "api_keys": _parse_key_list,
    "ingest_rate_limit": int,
    "ingest_rate_window": int,
}


def _from_file():
    if not _CONFIG_PATH.exists():
        return {}
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def _parse(key, raw):
    parser = _PARSERS.get(key, str)
    try:
        return parser(raw)
    except (TypeError, ValueError):
        logger.warning(
            "ignoring %s%s=%r: not a valid %s",
            ENV_PREFIX, key.upper(), raw, parser.__name__,
        )
        return None


def _from_env(keys):
    overrides = {}
    for key in keys:
        raw = os.environ.get(ENV_PREFIX + key.upper())
        if raw is None:
            continue
        value = _parse(key, raw)
        if value is not None:
            overrides[key] = value
    return overrides


def _validate_chunking(config):
    """chunk_overlap is the setting that genuinely matters: chunk_text()
    advances by (chunk_size - chunk_overlap), so an overlap at or above
    chunk_size never advances and loops forever, filling memory. Catching
    it here turns a hang into a startup error naming the setting.
    """
    if config["chunk_size"] < 1:
        raise ValueError(f"chunk_size must be >= 1, got {config['chunk_size']}")
    if not 0 <= config["chunk_overlap"] < config["chunk_size"]:
        raise ValueError(
            f"chunk_overlap must be >= 0 and < chunk_size "
            f"({config['chunk_size']}), got {config['chunk_overlap']}"
        )


def _validate_limits(config):
    """Bounds that would otherwise fail as a confusing 422, an always-empty
    result, or (for the rate window) a division by zero at request time."""
    if config["top_k"] < 1:
        raise ValueError(f"top_k must be >= 1, got {config['top_k']}")
    for key in ("max_batch_queries", "max_batch_ingest"):
        if config[key] < 1:
            raise ValueError(f"{key} must be >= 1, got {config[key]}")
    if config["ingest_rate_limit"] < 0:
        raise ValueError(
            f"ingest_rate_limit must be >= 0 (0 disables it), "
            f"got {config['ingest_rate_limit']}"
        )
    if config["ingest_rate_window"] < 1:
        raise ValueError(
            f"ingest_rate_window must be >= 1 second, got {config['ingest_rate_window']}"
        )


def _validate(config):
    """Reject settings that would break the pipeline rather than letting
    them fail somewhere less obvious."""
    _validate_chunking(config)
    _validate_limits(config)
    return config


@lru_cache(maxsize=1)
def _resolved_config():
    config = dict(_DEFAULTS)
    config.update(_from_file())
    config.update(_from_env(config))
    return _validate(config)


def load_config():
    """Return the merged config: defaults < config.json < SCOUT_* env vars.

    Resolved once per process and handed out as a copy, so callers on a hot
    path (chunking every document, say) don't re-read config.json each
    time, and a caller mutating what it gets back can't corrupt anyone
    else's view of it. List settings (api_keys) are copied too, so that
    promise holds a level deeper than a plain dict() copy would make it.
    """
    return {
        key: list(value) if isinstance(value, list) else value
        for key, value in _resolved_config().items()
    }


def reset_config_cache():
    """Drop the memoized config so a changed environment takes effect.

    Only meaningful for tests and for a process that edits its own
    environment after import; a deployed server resolves config once at
    startup and leaves it alone.
    """
    _resolved_config.cache_clear()
