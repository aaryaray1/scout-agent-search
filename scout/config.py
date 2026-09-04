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
    "log_level": "INFO",
}

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


def _validate(config):
    """Reject settings that would break the pipeline rather than letting
    them fail somewhere less obvious.

    chunk_overlap is the one that genuinely matters: chunk_text() advances
    by (chunk_size - chunk_overlap), so an overlap at or above chunk_size
    never advances and loops forever, filling memory. Catching it here
    turns a hang into a startup error naming the offending setting.
    """
    if config["chunk_size"] < 1:
        raise ValueError(f"chunk_size must be >= 1, got {config['chunk_size']}")
    if not 0 <= config["chunk_overlap"] < config["chunk_size"]:
        raise ValueError(
            f"chunk_overlap must be >= 0 and < chunk_size "
            f"({config['chunk_size']}), got {config['chunk_overlap']}"
        )
    if config["top_k"] < 1:
        raise ValueError(f"top_k must be >= 1, got {config['top_k']}")
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
    else's view of it.
    """
    return dict(_resolved_config())


def reset_config_cache():
    """Drop the memoized config so a changed environment takes effect.

    Only meaningful for tests and for a process that edits its own
    environment after import; a deployed server resolves config once at
    startup and leaves it alone.
    """
    _resolved_config.cache_clear()
