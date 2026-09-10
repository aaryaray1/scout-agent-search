"""The MCP server, driven through MCPServer.call_tool.

Calling through the server rather than the plain functions is the point:
what an agent actually gets is the registered tool, its schema and its
description, and those are as much the product here as the return value.
"""
import asyncio

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("mcp", reason="the MCP server is an optional extra")

from scout.api import app  # noqa: E402
from scout.fetch import FetchError  # noqa: E402
from scout.mcp import (  # noqa: E402
    LocalScout,
    RemoteScout,
    build_backend,
    build_server,
)

# Long enough that a bounded summary cannot accidentally contain all of it,
# which is the property these tests are actually about.
OPENING = (
    "Quernstone bearings fail at fault code Q-8812 when lubricant pressure "
    "drops below the rated minimum for more than thirty seconds."
)
FILLER = " ".join(f"Maintenance note {i} covers routine inspection." for i in range(60))
TAIL = "Flimflammery tolerances are recorded in appendix seven."
BODY = f"{OPENING} {FILLER} {TAIL}"
HTML = (
    f"<html><head><title>Quernstone Bearings</title></head><body><article>"
    f"<h1>Quernstone Bearings</h1><p>{BODY}</p></article></body></html>"
)
URL = "https://example.com/quernstone"


@pytest.fixture
def served(monkeypatch):
    """Serve HTML without the network, leaving extraction and chunking real."""
    import scout.web as web_module

    monkeypatch.setattr(web_module, "fetch_html", lambda url: HTML)


def call(server, name, arguments):
    return asyncio.run(server.call_tool(name, arguments)).structured_content


def tools(server):
    return {t.name: t for t in asyncio.run(server.list_tools())}


# -- what an agent discovers -------------------------------------------------


def test_both_tools_are_registered_with_descriptions_and_schemas():
    listed = tools(build_server(LocalScout()))
    assert set(listed) == {"search", "ingest"}
    for tool in listed.values():
        assert tool.description and len(tool.description) > 80
        assert tool.input_schema["type"] == "object"
        # Structured output: the response shape is published, not guessed.
        assert tool.output_schema


def test_the_ingest_tool_tells_the_agent_it_takes_several_urls():
    schema = tools(build_server(LocalScout()))["ingest"].input_schema
    assert schema["properties"]["urls"]["type"] == "array"
    assert schema["required"] == ["urls"]


# -- the token-saving contract -----------------------------------------------


def test_ingest_returns_a_summary_and_not_the_page_body(served):
    """The whole point. An ingest that echoed the page back would cost the
    tokens the agent came here to save."""
    server = build_server(LocalScout())

    page = call(server, "ingest", {"urls": [URL]})["results"][0]

    assert page["status"] == "ok"
    assert page["title"] == "Quernstone Bearings"
    assert page["chunks_indexed"] >= 1
    assert page["words"] > 100
    # Indexed, not returned: the summary stays bounded however long the page
    # is, which is the whole reason to ingest instead of fetching and reading.
    assert TAIL not in str(page)
    assert len(str(page)) < len(BODY)


def test_the_preview_is_short_enough_to_be_worth_paying_for(served):
    server = build_server(LocalScout())
    page = call(server, "ingest", {"urls": [URL]})["results"][0]
    assert 0 < len(page["preview"].split()) <= 40


def test_search_returns_passages_without_the_scoring_breakdown(served):
    """An agent acts on the text and the source. vector_score and
    keyword_score are tokens it pays for and never uses."""
    server = build_server(LocalScout())
    call(server, "ingest", {"urls": [URL]})

    hits = call(server, "search", {"query": "quernstone bearing fault", "top_k": 2})

    assert hits["query"] == "quernstone bearing fault"
    assert hits["results"], "the page was ingested, so it must be findable"
    assert set(hits["results"][0]) == {"content", "source", "confidence"}
    assert hits["results"][0]["source"] == URL


def test_search_respects_top_k(served):
    server = build_server(LocalScout())
    call(server, "ingest", {"urls": [URL]})
    assert len(call(server, "search", {"query": "quernstone", "top_k": 1})["results"]) == 1


# -- failure isolation -------------------------------------------------------


def test_one_unreachable_url_does_not_lose_the_others(monkeypatch):
    """An agent handing over ten links should get back the ones that worked,
    the same way /api/v1/ingest/batch behaves."""
    import scout.web as web_module

    def fetch(url):
        if "broken" in url:
            raise FetchError("connection refused")
        return HTML

    monkeypatch.setattr(web_module, "fetch_html", fetch)
    server = build_server(LocalScout())

    results = call(
        server, "ingest", {"urls": [URL, "https://example.com/broken"]}
    )["results"]

    assert [r["status"] for r in results] == ["ok", "fetch_error"]
    assert "connection refused" in results[1]["error"]
    assert results[1]["url"] == "https://example.com/broken"


def test_a_page_with_nothing_extractable_is_reported_not_raised(monkeypatch):
    import scout.web as web_module

    monkeypatch.setattr(web_module, "fetch_html", lambda url: "<html></html>")
    server = build_server(LocalScout())

    result = call(server, "ingest", {"urls": [URL]})["results"][0]
    assert result["status"] == "extract_error"
    assert result["error"]


# -- backends ----------------------------------------------------------------


def test_build_backend_runs_in_process_unless_given_a_url():
    assert isinstance(build_backend(), LocalScout)
    assert isinstance(build_backend("http://scout.internal"), RemoteScout)


def test_ingested_pages_stay_searchable_after_the_ingest_call(served):
    """Ingest indexes; it does not just extract. Two separate tool calls have
    to see the same corpus."""
    server = build_server(LocalScout())
    call(server, "ingest", {"urls": [URL]})

    hits = call(server, "search", {"query": "lubricant pressure", "top_k": 3})
    assert any(h["source"] == URL for h in hits["results"])


def test_the_remote_backend_speaks_to_a_real_scout_server(monkeypatch):
    """Same two tools, same response shape, against the actual HTTP stack."""
    import scout.web as web_module

    monkeypatch.setattr(web_module, "fetch_html", lambda url: HTML)

    with TestClient(app) as transport:
        backend = RemoteScout("http://testserver")
        backend.client._client = transport
        server = build_server(backend)

        page = call(server, "ingest", {"urls": [URL]})["results"][0]
        assert page["status"] == "ok"
        assert TAIL not in str(page)

        hits = call(server, "search", {"query": "quernstone", "top_k": 2})
        assert any(h["source"] == URL for h in hits["results"])
