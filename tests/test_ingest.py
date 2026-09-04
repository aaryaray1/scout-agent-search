"""Tests for document loading and chunking."""
import pytest

from scout.ingest import chunk_docs, chunk_text, load_markdown_docs


def test_chunk_text_splits_with_overlap():
    words = " ".join(str(i) for i in range(1000))
    chunks = chunk_text(words, chunk_size=400, overlap=50)
    assert len(chunks) == 3
    # Each window starts 350 words after the previous one.
    assert chunks[0].split()[0] == "0"
    assert chunks[1].split()[0] == "350"


def test_chunk_text_on_empty_input():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_rejects_an_overlap_that_never_advances():
    """overlap >= chunk_size leaves a stride of zero, which would re-emit
    the same window forever and exhaust memory. It has to fail loudly."""
    with pytest.raises(ValueError, match="smaller than chunk_size"):
        chunk_text("a b c d", chunk_size=10, overlap=10)
    with pytest.raises(ValueError):
        chunk_text("a b c d", chunk_size=10, overlap=25)


def test_chunk_docs_does_not_mutate_the_documents_it_is_given():
    """The chunker used to stamp a generated id onto its input docs. Callers
    hand in dicts they still own, so chunking has to be read-only."""
    doc = {"source": "a.md", "content": "hello world", "type": "documentation"}
    original = dict(doc)

    chunks = chunk_docs([doc])

    assert doc == original
    assert "id" not in doc
    assert chunks[0]["id"]  # the chunk still gets one


def test_chunk_docs_reuses_an_id_the_caller_supplied():
    doc = {"id": "fixed-id", "source": "a.md", "content": "hello world"}
    chunks = chunk_docs([doc])
    assert all(c["id"] == "fixed-id" for c in chunks)


def test_chunk_docs_carries_extra_doc_fields_onto_every_chunk():
    doc = {
        "source": "https://example.com/page",
        "content": " ".join(["word"] * 900),
        "type": "web",
        "title": "A Page",
        "page_metadata": {"author": "Jane Doe"},
    }
    chunks = chunk_docs([doc])
    assert len(chunks) > 1
    assert all(c["title"] == "A Page" for c in chunks)
    assert all(c["page_metadata"]["author"] == "Jane Doe" for c in chunks)
    assert [c["order"] for c in chunks] == list(range(len(chunks)))


def test_load_markdown_docs_is_deterministically_ordered(tmp_path):
    """Corpus order feeds the content hash and therefore cache validity, so
    it must not depend on filesystem enumeration order."""
    for name in ["c.md", "a.md", "b.md"]:
        (tmp_path / name).write_text("content of " + name, encoding="utf-8")

    sources = [d["source"] for d in load_markdown_docs(str(tmp_path))]
    assert sources == ["a.md", "b.md", "c.md"]


def test_load_markdown_docs_on_an_empty_folder(tmp_path):
    assert load_markdown_docs(str(tmp_path)) == []
