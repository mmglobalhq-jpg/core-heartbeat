"""Unit tests for the knowledge-base tool (tools/graphrag.py) — no live service."""
import json

import httpx
import tools.graphrag as g


def test_format_context_empty():
    assert "No relevant information" in g.format_context({"chunks": []})
    assert "No relevant information" in g.format_context({})


def test_format_context_cites_and_truncates():
    out = g.format_context({"chunks": [
        {"document_id": "d1", "content": "hello   world"},   # whitespace collapsed
        {"document_id": "d2", "content": "x" * 2000},          # truncated
    ]})
    assert "[doc:d1] hello world" in out
    assert "[doc:d2]" in out
    assert "…" in out


def test_run_graphrag_tool_success(monkeypatch):
    monkeypatch.setenv("GRAPHRAG_SERVICE_URL", "http://kb")
    monkeypatch.setenv("GRAPHRAG_API_KEY", "secret")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-user-id"] == "user-1"
        assert request.headers["authorization"] == "Bearer secret"
        body = json.loads(request.content)
        assert body["options"]["retrieve_only"] is True
        return httpx.Response(200, json={
            "chunks": [{"document_id": "d1", "content": "the answer is 42"}],
            "sources": ["d1"],
        })

    g._transport = httpx.MockTransport(handler)
    try:
        out = g.run_graphrag_tool("query_knowledge_base", "user-1", {"query": "q"})
    finally:
        g._transport = None
    assert "[doc:d1] the answer is 42" in out


def test_run_graphrag_tool_unknown_name():
    assert "unknown tool" in g.run_graphrag_tool("nope", "user-1", {})


def test_run_graphrag_tool_empty_query(monkeypatch):
    monkeypatch.setenv("GRAPHRAG_SERVICE_URL", "http://kb")
    monkeypatch.setenv("GRAPHRAG_API_KEY", "secret")
    assert g.run_graphrag_tool("query_knowledge_base", "user-1", {"query": ""}) == "error: empty query"


def test_run_graphrag_tool_missing_env_never_raises(monkeypatch):
    monkeypatch.delenv("GRAPHRAG_SERVICE_URL", raising=False)
    monkeypatch.delenv("GRAPHRAG_API_KEY", raising=False)
    out = g.run_graphrag_tool("query_knowledge_base", "user-1", {"query": "q"})
    assert out.startswith("error:")
