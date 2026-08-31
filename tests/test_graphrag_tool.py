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


# --- summarize_document -----------------------------------------------------
#
# The failure these cover: asking the assistant to summarize one document produced a
# summary built mostly from a DIFFERENT document. query_knowledge_base is a
# corpus-wide top-k search, so a title-shaped query about one issue of a weekly
# publication matched every issue's near-identical boilerplate equally.


def _kb_env(monkeypatch):
    monkeypatch.setenv("GRAPHRAG_SERVICE_URL", "http://kb")
    monkeypatch.setenv("GRAPHRAG_API_KEY", "secret")


DIGEST = {
    "document": {
        "id": "d-aug28",
        "title": "North America Securitized Products Weekly Research - August 28, 2026",
        "summary": "Stable MBS spreads; caution on 6.5 pools with temp buydowns.",
        "created_at": "2026-08-31T12:37:24Z",
    },
    "chunks": [
        {"chunk_index": 0, "content": "MBS   market   commentary opening."},
        {"chunk_index": 31, "content": "Specified pool payup decomposition."},
        {"chunk_index": 63, "content": "CMBS multifamily bottoming out."},
    ],
    "alternatives": [],
}


def test_summarize_document_is_document_scoped(monkeypatch):
    """The request must name the document; it must NOT be a corpus-wide search."""
    _kb_env(monkeypatch)
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        assert request.headers["x-user-id"] == "user-1"
        return httpx.Response(200, json=DIGEST)

    g._transport = httpx.MockTransport(handler)
    try:
        context, titles = g.run_graphrag_tool(
            "summarize_document", "user-1", {"document": "securitized products august 28"}
        )
    finally:
        g._transport = None

    assert seen["path"] == "/api/summarize"
    assert seen["body"]["document_title"] == "securitized products august 28"
    # The whole point: one document, cited as itself.
    assert titles == [DIGEST["document"]["title"]]
    assert "August 28, 2026" in context


def test_summarize_document_leads_with_the_ingest_time_abstract(monkeypatch):
    """documents.summary was written at ingest and never read. It leads the context."""
    _kb_env(monkeypatch)
    g._transport = httpx.MockTransport(lambda r: httpx.Response(200, json=DIGEST))
    try:
        context, _ = g.run_graphrag_tool("summarize_document", "user-1", {"document": "aug 28"})
    finally:
        g._transport = None
    assert "Stable MBS spreads" in context
    assert context.index("Stable MBS spreads") < context.index("Specified pool payup")


def test_summarize_document_covers_the_whole_document(monkeypatch):
    """Excerpts span the document rather than clustering at its head."""
    _kb_env(monkeypatch)
    g._transport = httpx.MockTransport(lambda r: httpx.Response(200, json=DIGEST))
    try:
        context, _ = g.run_graphrag_tool("summarize_document", "user-1", {"document": "aug 28"})
    finally:
        g._transport = None
    assert "MBS market commentary opening." in context  # whitespace collapsed
    assert "CMBS multifamily bottoming out." in context


def test_summarize_document_passes_a_focus_through(monkeypatch):
    _kb_env(monkeypatch)
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=DIGEST)

    g._transport = httpx.MockTransport(handler)
    try:
        g.run_graphrag_tool(
            "summarize_document", "user-1",
            {"document": "aug 28", "focus": "specified pools"},
        )
    finally:
        g._transport = None
    assert seen["focus"] == "specified pools"


def test_summarize_document_never_falls_back_to_another_document(monkeypatch):
    """An unresolvable reference reports failure and lists what EXISTS.

    Silently widening to a corpus-wide search is what produced a confident summary of
    the wrong report. A wrong answer that looks right is worse than "I couldn't find it".
    """
    _kb_env(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/summarize":
            return httpx.Response(404, json={"error": "No document matched"})
        return httpx.Response(200, json={"documents": [
            {"title": "Agency MBS Weekly: The Basis Is Mid", "created_at": "2026-08-24"},
        ]})

    g._transport = httpx.MockTransport(handler)
    try:
        context, titles = g.run_graphrag_tool(
            "summarize_document", "user-1", {"document": "a report I never saved"}
        )
    finally:
        g._transport = None

    assert context.startswith("error:")
    assert titles == []                       # nothing gets cited
    assert "Agency MBS Weekly" in context     # but the real titles are offered


def test_summarize_document_requires_a_document_name():
    assert g.run_graphrag_tool("summarize_document", "user-1", {"document": ""}) == (
        "error: no document named", [],
    )


# --- list_knowledge_base_documents ------------------------------------------


def test_list_knowledge_base_documents(monkeypatch):
    _kb_env(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/documents"
        assert request.method == "GET"
        return httpx.Response(200, json={"documents": [
            {"title": "Doc One", "created_at": "2026-08-28T00:00:00Z"},
            {"title": "Doc Two", "created_at": "2026-07-31T00:00:00Z"},
        ]})

    g._transport = httpx.MockTransport(handler)
    try:
        context, titles = g.run_graphrag_tool("list_knowledge_base_documents", "user-1", {})
    finally:
        g._transport = None
    assert "Doc One" in context and "Doc Two" in context
    assert "2026-08-28" in context   # the date is how issues are told apart
    assert titles == []              # a listing is not a cited source


def test_list_knowledge_base_documents_empty(monkeypatch):
    _kb_env(monkeypatch)
    g._transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"documents": []}))
    try:
        context, _ = g.run_graphrag_tool("list_knowledge_base_documents", "user-1", {})
    finally:
        g._transport = None
    assert "no documents" in context.lower()


# --- guard scoping ----------------------------------------------------------


def test_retrieve_once_guard_covers_only_the_corpus_wide_search():
    """Listing then summarizing is a legitimate TWO-call sequence in one turn.

    The supervisor's consult-once guard must not cover the document tools, or
    "what do I have on X?" followed by "summarize the August 28 one" gets cut off
    after the listing.
    """
    assert g.KB_RETRIEVE_ONCE_TOOLS == {"query_knowledge_base"}
    assert g.KB_RETRIEVE_ONCE_TOOLS < set(g.GRAPHRAG_TOOL_REGISTRY)
    assert "summarize_document" in g.GRAPHRAG_TOOL_REGISTRY
