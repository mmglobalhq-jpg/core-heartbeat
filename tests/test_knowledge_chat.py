"""Knowledge chat agent: tool loop, KB-only grounding, citations, session memory.

A fake Gemini client replays scripted stream rounds and records every request, and the
KB service is monkeypatched, so the real loop runs end to end with no network.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from google.genai import types

import knowledge_chat as kc
import services.kb as kb
from auth import SANDBOX_USER_ID, resolve_user_id
from models import KnowledgeChatRequest

USER = "11111111-1111-1111-1111-111111111111"


def text_chunk(text: str) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part.from_text(text=text)]))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=100, candidates_token_count=20, total_token_count=120),
    )


def call_chunk(*calls: tuple[str, dict]) -> types.GenerateContentResponse:
    parts = [types.Part(function_call=types.FunctionCall(name=n, args=a)) for n, a in calls]
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=100, candidates_token_count=5, total_token_count=105),
    )


class FakeModels:
    def __init__(self, rounds: list[list[types.GenerateContentResponse]] | Exception):
        self.rounds = rounds
        self.requests: list[dict] = []

    async def generate_content_stream(self, *, model, contents, config):
        self.requests.append({"model": model, "contents": list(contents), "config": config})
        if isinstance(self.rounds, Exception):
            raise self.rounds
        chunks = self.rounds.pop(0)

        async def gen():
            for c in chunks:
                yield c

        return gen()


class FakeClient:
    def __init__(self, rounds):
        self.aio = type("Aio", (), {})()
        self.aio.models = FakeModels(rounds)


@pytest.fixture
def kbmock(monkeypatch):
    calls: list[tuple] = []

    async def _list(owner):
        calls.append(("list", owner))
        return {"documents": [
            {"id": "d-sep", "title": "Securitized Products Research - September 11, 2026", "created_at": "2026-09-16T00:00:00Z", "summary": "Sep issue"},
            {"id": "d-aug", "title": "Securitized Products Research - August 28, 2026", "created_at": "2026-09-16T00:00:00Z", "summary": "Aug issue"},
        ]}

    async def _search(owner, query, *, top_k=8, document_titles=None, include_parent_context=True):
        calls.append(("search", owner, query, document_titles))
        return {"chunks": [
            {"id": "c1", "document_id": "d-sep", "title": "Securitized Products Research - September 11, 2026",
             "document_created_at": "2026-09-16", "chunk_index": 4, "content": "short child",
             "parent_content": "Subprime auto delinquencies rose to 6.9% in August.", "score": 5.2},
            {"id": "c2", "document_id": "d-aug", "title": "Securitized Products Research - August 28, 2026",
             "chunk_index": 9, "content": "Unrelated disclaimer text.", "score": -10.4},
        ]}

    async def _read(owner, document, *, focus=None, max_chunks=12):
        calls.append(("read", owner, document, focus))
        if "nonexistent" in document:
            raise kb.KbNotFound({"error": "No document matched", "alternatives": [{"id": "d-aug", "title": "Securitized Products Research - August 28, 2026"}]})
        return {"document": {"id": "d-aug", "title": "Securitized Products Research - August 28, 2026", "summary": "Aug summary", "created_at": "2026-09-16"},
                "chunks": [{"chunk_index": 0, "content": "MBS spreads were stable."}, {"chunk_index": 30, "content": "Multifamily CMBS bottomed in Q2."}],
                "alternatives": []}

    monkeypatch.setattr(kb, "list_documents", _list)
    monkeypatch.setattr(kb, "search", _search)
    monkeypatch.setattr(kb, "read_document", _read)
    return calls


def run_turn(monkeypatch, rounds, **req) -> tuple[list[dict], FakeClient]:
    client = FakeClient(rounds)
    monkeypatch.setattr(kc, "_client", lambda: client)
    payload = KnowledgeChatRequest(text=req.pop("text", "question"), **req)

    async def collect():
        return [e async for e in kc.run(payload, USER)]

    return asyncio.run(collect()), client


def ledger_rows() -> list[dict]:
    from services import llm_ledger
    rows = []
    for f in Path(llm_ledger.SPOOL_DIR).rglob("*.jsonl"):
        rows += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return rows


def test_search_then_cited_answer(monkeypatch, kbmock):
    events, client = run_turn(monkeypatch, [
        [call_chunk(("search_knowledge", {"query": "subprime auto delinquencies"}))],
        [text_chunk("Subprime auto delinquencies rose to 6.9% [1].")],
    ])
    assert {"tool_call": {"name": "search_knowledge", "args": {"query": "subprime auto delinquencies"}}} in events
    tokens = "".join(e["token"] for e in events if "token" in e)
    assert tokens == "Subprime auto delinquencies rose to 6.9% [1]."
    sources = next(e["sources"] for e in events if "sources" in e)
    assert [s["n"] for s in sources] == [1]
    assert sources[0]["document_id"] == "d-sep" and "6.9%" in sources[0]["excerpt"]
    assert events[-1] == {"status": "completed"}

    # the function response the model saw: parent context used, and the -10 passage filtered out
    response_part = client.aio.models.requests[1]["contents"][-1].parts[0]
    result = response_part.function_response.response["result"]
    assert "[1] Securitized Products Research - September 11, 2026" in result
    assert "Subprime auto delinquencies rose" in result
    assert "Unrelated disclaimer" not in result

    # one ledger row per model round, grouped as one logical call
    rows = [r for r in ledger_rows() if r["operation"] == "heartbeat.knowledge_chat"]
    assert len(rows) == 2 and len({r["request_group"] for r in rows}) == 1
    assert all(r["status"] == "ok" and r["usage_source"] == "provider" for r in rows)


def test_system_prompt_is_kb_only_and_lists_the_library(monkeypatch, kbmock):
    _, client = run_turn(monkeypatch, [[text_chunk("Hi — ask me about your documents.")]])
    system = client.aio.models.requests[0]["config"].system_instruction
    assert "ONLY the user's own knowledge base" in system
    assert "Never use general knowledge" in system
    assert "Securitized Products Research - September 11, 2026" in system


def test_nothing_relevant_yields_no_sources(monkeypatch, kbmock):
    async def _empty(owner, query, **kw):
        return {"chunks": [{"id": "x", "document_id": "d", "title": "T", "content": "noise", "score": -9.0}]}
    monkeypatch.setattr(kb, "search", _empty)
    events, client = run_turn(monkeypatch, [
        [call_chunk(("search_knowledge", {"query": "who won the world series"}))],
        [text_chunk("Your knowledge base doesn't cover that.")],
    ])
    result = client.aio.models.requests[1]["contents"][-1].parts[0].function_response.response["result"]
    assert result == "No passages in the knowledge base are relevant to this query."
    assert {"sources": []} in events


def test_parallel_reads_number_passages_across_calls(monkeypatch, kbmock):
    events, client = run_turn(monkeypatch, [
        [call_chunk(("read_document", {"document": "August 28"}), ("search_knowledge", {"query": "auto", "documents": ["September 11"]}))],
        [text_chunk("Aug: spreads stable [1]; Sep: delinquencies up [3].")],
    ])
    assert ("search", USER, "auto", ["September 11"]) in kbmock
    sources = next(e["sources"] for e in events if "sources" in e)
    assert [s["n"] for s in sources] == [1, 3]
    assert [s["document_id"] for s in sources] == ["d-aug", "d-sep"]


def test_unknown_document_tells_the_model_the_closest_titles(monkeypatch, kbmock):
    _, client = run_turn(monkeypatch, [
        [call_chunk(("read_document", {"document": "nonexistent report"}))],
        [text_chunk("I couldn't find that document.")],
    ])
    result = client.aio.models.requests[1]["contents"][-1].parts[0].function_response.response["result"]
    assert "No document in the knowledge base matches" in result
    assert "August 28, 2026" in result


def test_follow_up_carries_the_documents_earlier_answers_cited(monkeypatch, kbmock):
    _, client = run_turn(
        monkeypatch,
        [[text_chunk("ok")]],
        text="What else did that report say?",
        history=[
            {"role": "assistant", "content": "orphan leading reply"},
            {"role": "user", "content": "Summarize the August 28 issue"},
            {"role": "assistant", "content": "It covered MBS [1].", "sources": [{"document_id": "d-aug", "title": "Securitized Products Research - August 28, 2026"}]},
        ],
    )
    contents = client.aio.models.requests[0]["contents"]
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert "(Documents cited in this answer: Securitized Products Research - August 28, 2026)" in contents[1].parts[0].text
    assert contents[2].parts[0].text == "What else did that report say?"


def test_rounds_are_bounded_and_the_last_one_cannot_call_tools(monkeypatch, kbmock):
    monkeypatch.setattr(kc, "MAX_ROUNDS", 2)
    events, client = run_turn(monkeypatch, [
        [call_chunk(("search_knowledge", {"query": "a"}))],
        [call_chunk(("search_knowledge", {"query": "b"}))],
        [text_chunk("Answer [1].")],
    ])
    reqs = client.aio.models.requests
    assert len(reqs) == 3
    assert reqs[0]["config"].tool_config.function_calling_config.mode == "AUTO"
    assert reqs[2]["config"].tool_config.function_calling_config.mode == "NONE"
    assert events[-1] == {"status": "completed"}


def test_model_failure_before_any_answer_is_reported(monkeypatch, kbmock):
    events, _ = run_turn(monkeypatch, RuntimeError("quota"))
    assert "couldn't reach the model" in events[0]["token"]
    assert events[-1] == {"status": "error"}


def test_kb_outage_becomes_text_not_a_crash(monkeypatch, kbmock):
    async def _boom(owner, query, **kw):
        raise ConnectionError("down")
    monkeypatch.setattr(kb, "search", _boom)
    events, client = run_turn(monkeypatch, [
        [call_chunk(("search_knowledge", {"query": "x"}))],
        [text_chunk("The knowledge base is unavailable right now.")],
    ])
    result = client.aio.models.requests[1]["contents"][-1].parts[0].function_response.response["result"]
    assert result.startswith("error: the knowledge base could not be reached")
    assert events[-1] == {"status": "completed"}


def test_context_budget_caps_retrieved_text(monkeypatch, kbmock):
    monkeypatch.setattr(kc, "CONTEXT_CHARS", 60)
    ctx = kc.TurnContext()
    assert ctx.add("a", "d", "T", 0, "x" * 50) is not None
    assert ctx.add("b", "d", "T", 1, "y" * 50) is None
    assert ctx.add("a", "d", "T", 0, "x" * 50).n == 1  # a repeat keeps its number


def test_route_streams_sse_and_rejects_sandbox(monkeypatch):
    from main import create_app

    async def fake_run(payload, user_id):
        yield {"token": f"hi {payload.text}"}
        yield {"sources": []}
        yield {"status": "completed"}

    monkeypatch.setattr(kc, "run", fake_run)
    app = create_app()
    app.dependency_overrides[resolve_user_id] = lambda: USER
    r = TestClient(app).post("/kb/chat/stream", json={"text": "there"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    frames = [json.loads(line[5:]) for line in r.text.splitlines() if line.startswith("data:")]
    assert frames == [{"token": "hi there"}, {"sources": []}, {"status": "completed"}]

    app.dependency_overrides[resolve_user_id] = lambda: SANDBOX_USER_ID
    assert TestClient(app).post("/kb/chat/stream", json={"text": "x"}).status_code == 401


def test_replace_is_passed_through_to_the_kb_service(monkeypatch):
    from main import create_app
    import services.documents as docs

    seen = {}

    async def _ingest(owner, filename, content, replaces_document_id=None):
        seen["replaces"] = replaces_document_id
        return {"job_id": "j", "status": "pending"}

    monkeypatch.setattr(kb, "ingest", _ingest)
    monkeypatch.setattr(docs, "fetch_original", lambda uid, did: b"bytes")
    app = create_app()
    app.dependency_overrides[resolve_user_id] = lambda: USER
    r = TestClient(app).post("/kb/ingest", json={"doc_id": "u1", "filename": "f.pdf", "replaces_document_id": "old-doc"})
    assert r.status_code == 200 and seen["replaces"] == "old-doc"


def test_kb_client_retries_5xx_and_surfaces_404_body(monkeypatch):
    import httpx

    monkeypatch.setenv("GRAPHRAG_SERVICE_URL", "http://graph-rag:3000")
    monkeypatch.setenv("GRAPHRAG_API_KEY", "k")
    monkeypatch.setattr(kb, "RETRY_BACKOFF_S", 0)
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.headers["X-User-Id"] == USER
        if request.url.path == "/api/summarize":
            return httpx.Response(404, json={"error": "No document matched", "alternatives": [{"title": "A"}]})
        hits["n"] += 1
        if hits["n"] < 3:
            return httpx.Response(503)
        assert body["options"]["document_titles"] == ["Sep 11"]
        return httpx.Response(200, json={"chunks": []})

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))

    assert asyncio.run(kb.search(USER, "q", document_titles=["Sep 11"])) == {"chunks": []}
    assert hits["n"] == 3
    with pytest.raises(kb.KbNotFound) as exc:
        asyncio.run(kb.read_document(USER, "nope"))
    assert exc.value.payload["alternatives"] == [{"title": "A"}]
