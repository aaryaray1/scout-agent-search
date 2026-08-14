import pytest
import scout.index as index_module


@pytest.fixture(autouse=True)
def isolate_index_dir(tmp_path, monkeypatch):
    """Point scout.index at a throwaway directory for every test.

    Without this, any test that builds a Retriever or calls add_chunks()
    reads and writes the real data/index/ in the project checkout: cache
    files for the docs corpus, and (since ingest persistence was added) a
    running record of every page ever ingested. A test run would leave
    that directory dirtied with test fixtures, and a later test run (or
    the real API server) would pick that leftover content back up as if it
    were genuine. Redirecting INDEX_DIR keeps every test's on-disk state
    scoped to its own tmp_path.
    """
    monkeypatch.setattr(index_module, "INDEX_DIR", str(tmp_path / "index"))
