"""Tests for config layering and the settings that can break the pipeline."""
import pytest

from scout.config import ENV_PREFIX, load_config, reset_config_cache


@pytest.fixture(autouse=True)
def clear_config_cache():
    """load_config() memoizes, so a test that changes the environment has to
    invalidate it on the way in and on the way out."""
    reset_config_cache()
    yield
    reset_config_cache()


def test_defaults_are_available_without_any_env():
    config = load_config()
    assert config["vector_model"]
    assert config["top_k"] >= 1
    assert config["chunk_overlap"] < config["chunk_size"]


def test_env_var_overrides_the_config_file(monkeypatch):
    monkeypatch.setenv(ENV_PREFIX + "TOP_K", "11")
    monkeypatch.setenv(ENV_PREFIX + "INDEX_DIR", "/mnt/scout-index")
    reset_config_cache()

    config = load_config()
    assert config["top_k"] == 11
    assert config["index_dir"] == "/mnt/scout-index"


def test_env_var_is_coerced_to_the_right_type(monkeypatch):
    monkeypatch.setenv(ENV_PREFIX + "VECTOR_WEIGHT", "0.8")
    reset_config_cache()
    assert load_config()["vector_weight"] == pytest.approx(0.8)


def test_unparseable_env_var_falls_back_instead_of_crashing(monkeypatch):
    """A typo in a deployment env var shouldn't take the service down; it
    should be logged and ignored in favour of the configured value."""
    monkeypatch.setenv(ENV_PREFIX + "TOP_K", "not-a-number")
    reset_config_cache()
    assert load_config()["top_k"] == 3


def test_config_returns_a_copy_so_callers_cannot_corrupt_it():
    first = load_config()
    first["top_k"] = 999
    assert load_config()["top_k"] != 999


@pytest.mark.parametrize("key,value", [
    ("CHUNK_OVERLAP", "400"),   # equal to chunk_size: chunking would never advance
    ("CHUNK_OVERLAP", "500"),   # larger still
    ("CHUNK_SIZE", "0"),
    ("TOP_K", "0"),
])
def test_settings_that_would_break_the_pipeline_are_rejected_at_load(monkeypatch, key, value):
    monkeypatch.setenv(ENV_PREFIX + key, value)
    reset_config_cache()
    with pytest.raises(ValueError):
        load_config()
