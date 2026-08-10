import json
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "config.json"

_DEFAULTS = {
    "vector_model": "all-MiniLM-L6-v2",
    "docs_path": "data/docs",
    "top_k": 3,
    "vector_weight": 0.65,
    "keyword_weight": 0.35,
}


def load_config():
    """Load scout/config.json, falling back to defaults for any missing key."""
    config = dict(_DEFAULTS)
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            config.update(json.load(f))
    return config
