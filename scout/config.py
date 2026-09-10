"""Runtime configuration: defaults < config.json < SCOUT_* env vars.

The env layer wins so a deployment can be retargeted without editing files
inside the image. See docs/design/configuration.md for every setting.
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
    # "segmented" appends what changed; "json" rewrites the whole store.
    "chunk_store": "segmented",
    # Request-shape caps, bounding what one caller can ask Scout to do.
    "max_top_k": 50,
    "max_query_chars": 2000,
    "max_html_bytes": 5 * 1024 * 1024,
    "max_batch_queries": 10,
    "max_batch_ingest": 5,
    # Empty means no authentication at all. Startup says so out loud.
    "api_keys": [],
    # Ingest only, since it spends Scout's network. 0 disables it.
    "ingest_rate_limit": 30,
    "ingest_rate_window": 60,
    "log_level": "INFO",
}


def _parse_key_list(raw):
    """Read a comma-separated env value into a list of non-empty items.

    An empty value yields [], a deliberate "no auth" rather than a failure.
    """
    if isinstance(raw, list):
        return [str(k).strip() for k in raw if str(k).strip()]
    return [k.strip() for k in str(raw).split(",") if k.strip()]


# Env values arrive as strings, so non-string settings say how to read back.
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

# A literal rather than an import from scout.store, which reads this module.
_CHUNK_STORES = ("segmented", "json")


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
    """chunk_text() advances by (chunk_size - chunk_overlap), so an overlap
    at or above chunk_size loops forever. Turn that hang into an error."""
    if config["chunk_size"] < 1:
        raise ValueError(f"chunk_size must be >= 1, got {config['chunk_size']}")
    if not 0 <= config["chunk_overlap"] < config["chunk_size"]:
        raise ValueError(
            f"chunk_overlap must be >= 0 and < chunk_size "
            f"({config['chunk_size']}), got {config['chunk_overlap']}"
        )


def _validate_limits(config):
    """Bounds that would otherwise surface as a confusing 422, an
    always-empty result, or a division by zero at request time."""
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
    """Reject settings that would break the pipeline, naming the setting."""
    _validate_chunking(config)
    _validate_limits(config)
    if config["chunk_store"] not in _CHUNK_STORES:
        # At startup, not on the first ingest with the store half-written.
        raise ValueError(
            f"chunk_store must be one of {list(_CHUNK_STORES)}, "
            f"got {config['chunk_store']!r}"
        )
    return config


@lru_cache(maxsize=1)
def _resolved_config():
    config = dict(_DEFAULTS)
    config.update(_from_file())
    config.update(_from_env(config))
    return _validate(config)


def load_config():
    """Return the merged config, resolved once per process and copied per
    call so a mutating caller can't corrupt anyone else's view."""
    return {
        key: list(value) if isinstance(value, list) else value
        for key, value in _resolved_config().items()
    }


def reset_config_cache():
    """Drop the memoized config so a changed environment takes effect.

    For tests; a deployed server resolves config once at startup.
    """
    _resolved_config.cache_clear()
