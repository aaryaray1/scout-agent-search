import pytest
import scout.index as index_module


@pytest.fixture(autouse=True)
def isolate_index_dir(tmp_path, monkeypatch):
    """Point scout.index at a throwaway directory for every test.

    Without it, anything that builds a Retriever reads and writes the real
    data/index/, dirtying it with fixtures a later run would pick back up.
    """
    monkeypatch.setattr(index_module, "INDEX_DIR", str(tmp_path / "index"))
