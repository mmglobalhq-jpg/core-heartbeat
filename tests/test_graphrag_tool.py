"""Unit tests for the knowledge-base tool (tools/graphrag.py) — no live service."""
import json

import httpx
import tools.graphrag as g


def test_format_context_empty():
    assert "No relevant information" in g.format_context({"chunks": []})
    assert "No relevant information" in g.format_context({})


def test_format_context_cites_title_and_truncates():
    out = g.format_context({"chunks": [
        {"document_id": "d1", "title": "Roasted Chicken", "content": "hello   world"},  # ws collapsed
        {"document_id": "d2", "title": "Long Doc", "content": "x" * 2000},               # truncated
        {"document_id": "d3", "content": "no title here"},                              # falls back
    ]})
    assert "[Roasted Chicken] hello world" in out
    assert "[Long Doc]" in out
    assert "[Untitled document] no title here" in out
    assert "…" in out


def test_source_titles_distinct_non_null():
    titles = g.source_titles({"sources": [
        {"id": "d1", "title": "Roasted Chicken"},
        {"id": "d2", "title": "Roasted Chicken"},   # dupe collapsed
        {"id": "d3", "title": None},                 # dropped
        {"id": "d4", "title": "Braising 101"},
    ]})
    assert titles == ["Roasted Chicken", "Braising 101"]


def test_run_graphrag_tool_success(monkeypatch):
    monkeypatch.setenv("GRAPHRAG_SERVICE_URL", "http://kb")
    monkeypatch.setenv("GRAPHRAG_API_KEY", "secret")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-user-id"] == "user-1"
        assert request.headers["authorization"] == "Bearer secret"
        body = json.loads(request.content)
        assert body["options"]["retrieve_only"] is True
        return httpx.Response(200, json={
            "chunks": [{"document_id": "d1", "title": "Roasted Chicken", "content": "the answer is 42"}],
            "sources": [{"id": "d1", "title": "Roasted Chicken"}],
        })

    g._transport = httpx.MockTransport(handler)
    try:
        context, titles = g.run_graphrag_tool("query_knowledge_base", "user-1", {"query": "q"})
    finally:
        g._transport = None
    assert "[Roasted Chicken] the answer is 42" in context
    assert titles == ["Roasted Chicken"]


def test_run_graphrag_tool_unknown_name():
    context, titles = g.run_graphrag_tool("nope", "user-1", {})
    assert "unknown tool" in context
    assert titles == []


def test_run_graphrag_tool_empty_query(monkeypatch):
    monkeypatch.setenv("GRAPHRAG_SERVICE_URL", "http://kb")
    monkeypatch.setenv("GRAPHRAG_API_KEY", "secret")
    assert g.run_graphrag_tool("query_knowledge_base", "user-1", {"query": ""}) == ("error: empty query", [])


def test_run_graphrag_tool_missing_env_never_raises(monkeypatch):
    monkeypatch.delenv("GRAPHRAG_SERVICE_URL", raising=False)
    monkeypatch.delenv("GRAPHRAG_API_KEY", raising=False)
    context, titles = g.run_graphrag_tool("query_knowledge_base", "user-1", {"query": "q"})
    assert context.startswith("error:")
    assert titles == []
