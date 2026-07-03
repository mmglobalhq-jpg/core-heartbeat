"""Tests for the live local Ollama worker node (feature 005).

Everything uses httpx.MockTransport — NO real network, NO Ollama daemon, no spend.
Covers real-text routing (US1/SC-001), token-usage capture (US2/SC-002), and the
full failure-degradation matrix (US3/SC-003/SC-005).
"""

import asyncio
import json

import httpx
import pytest

import orchestrator
from models import IntentPayload, RoutingDecision, TokenUsage
from orchestrator import generate_local


# --- MockTransport seam (no network) ----------------------------------------

def _mock_client(handler) -> httpx.AsyncClient:
    """AsyncClient whose transport is a MockTransport wrapping `handler`."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def json_handler(status=200, body=None):
    """Handler returning a fixed JSON response."""
    def _h(request):
        return httpx.Response(status, json=body if body is not None else {})
    return _h


def raising_handler(exc):
    """Handler that raises a transport-level exception."""
    def _h(request):
        raise exc
    return _h


def ndjson_handler(chunks, prompt_eval_count=3, eval_count=4):
    """Handler emitting Ollama-style streaming NDJSON: one object per chunk, plus a
    final done object carrying the token counts."""
    def _h(request):
        lines = [json.dumps({"response": c, "done": False}) for c in chunks]
        lines.append(json.dumps(
            {"response": "", "done": True,
             "prompt_eval_count": prompt_eval_count, "eval_count": eval_count}
        ))
        return httpx.Response(200, text="\n".join(lines))
    return _h


def state(intent="greet", messages=None, step=1):
    return {
        "intent": IntentPayload(intent=intent, confidence=0.9, raw_input="x", source="cli"),
        "messages": messages or [],
        "usage": None,
        "visited": [],
        "step": step,
        "next": "",
        "status": "",
    }


def call(handler):
    """Run generate_local against a MockTransport handler, returning its 3-tuple."""
    async def _run():
        async with _mock_client(handler) as client:
            return await generate_local(state(), client)
    return asyncio.run(_run())


def install_ollama(monkeypatch, handler):
    """Monkeypatch build_ollama_client to a MockTransport-backed client."""
    monkeypatch.setattr(
        orchestrator, "build_ollama_client", lambda: _mock_client(handler)
    )


# --- US1: real generated text routes through the node -----------------------

def test_generate_local_returns_model_text():
    text, failure, _ = call(
        json_handler(200, {"response": "hello from qwen", "prompt_eval_count": 26, "eval_count": 290})
    )
    assert failure is None
    assert text == "hello from qwen"


def test_local_llm_node_records_real_text(monkeypatch):
    install_ollama(monkeypatch, json_handler(200, {"response": "NODE TEXT"}))
    update = asyncio.run(orchestrator.local_llm(state()))
    assert update["messages"][0].content == "NODE TEXT"
    assert update["messages"][0].source == "local_llm"
    assert "local_llm" in update["visited"]


# --- US2: token usage stays observable --------------------------------------

def test_usage_extracted_from_counts():
    _, _, usage = call(
        json_handler(200, {"response": "x", "prompt_eval_count": 26, "eval_count": 290})
    )
    assert usage == TokenUsage(input_tokens=26, output_tokens=290, total_tokens=316)


def test_usage_absent_is_zero_no_error():
    text, failure, usage = call(json_handler(200, {"response": "x"}))
    assert failure is None and text == "x"
    assert usage == TokenUsage()  # zeros


def test_usage_partial_counts_missing_as_zero():
    _, _, usage = call(json_handler(200, {"response": "x", "prompt_eval_count": 12}))
    assert usage == TokenUsage(input_tokens=12, output_tokens=0, total_tokens=12)


def test_node_carries_usage_on_channel(monkeypatch):
    install_ollama(
        monkeypatch,
        json_handler(200, {"response": "y", "prompt_eval_count": 5, "eval_count": 7}),
    )
    update = asyncio.run(orchestrator.local_llm(state()))
    assert update["usage"] == TokenUsage(input_tokens=5, output_tokens=7, total_tokens=12)


# --- US3: failure degradation matrix ----------------------------------------

def test_connect_error_is_unreachable():
    text, failure, usage = call(raising_handler(httpx.ConnectError("refused")))
    assert text is None and failure.category == "unreachable"
    assert usage == TokenUsage()


def test_timeout_is_timeout():
    _, failure, _ = call(raising_handler(httpx.ReadTimeout("slow")))
    assert failure.category == "timeout"


def test_non_2xx_is_invalid_output():
    _, failure, _ = call(json_handler(404, {"error": "model not found"}))
    assert failure.category == "invalid_output"
    assert failure.detail == "HTTP 404"


def test_missing_response_field_is_invalid_output():
    _, failure, _ = call(json_handler(200, {"done": True}))  # no "response"
    assert failure.category == "invalid_output"


def test_non_json_body_is_invalid_output():
    def _h(request):
        return httpx.Response(200, text="not json at all")
    _, failure, _ = call(_h)
    assert failure.category == "invalid_output"


def test_generate_local_never_raises():
    # even an unexpected exception type from the transport is caught
    _, failure, _ = call(raising_handler(RuntimeError("boom")))
    assert failure is not None and failure.category == "unreachable"


def test_local_llm_node_degrades_on_failure(monkeypatch):
    install_ollama(monkeypatch, raising_handler(httpx.ConnectError("refused")))
    update = asyncio.run(orchestrator.local_llm(state()))
    assert update["messages"][0].content == "local inference failure: unreachable"
    assert update["usage"] == TokenUsage()
    assert "local_llm" in update["visited"]
    assert "next" not in update and "status" not in update  # routing stays with the Supervisor


# --- US3: full-run degradation still terminates -----------------------------

class _SupResp:
    def __init__(self, choice):
        self.parsed = RoutingDecision(next_node=choice)
        self.text = None
        self.usage_metadata = None


class _SupClient:
    """Scripted supervisor fake (feature 004 seam): routes local_llm then finish."""

    def __init__(self, choices):
        it = iter(choices)

        class _Models:
            def generate_content(self, model, contents, config):
                try:
                    return _SupResp(next(it))
                except StopIteration:
                    return _SupResp("finish")

        self.models = _Models()


def test_full_run_terminates_despite_local_failure(monkeypatch):
    monkeypatch.setattr(
        orchestrator, "get_client", lambda *a, **k: _SupClient(["local_llm", "finish"])
    )
    install_ollama(monkeypatch, raising_handler(httpx.ReadTimeout("slow")))
    outcome = asyncio.run(orchestrator.run(
        IntentPayload(intent="greet", confidence=0.9, raw_input="x", source="cli")
    ))
    assert "local_llm" in outcome.nodes_executed
    assert any("local inference failure: timeout" in m.content for m in outcome.messages)
    assert isinstance(outcome.steps, int)  # returned, no hang / no exception


# --- Feature: per-token streaming (SSE) -------------------------------------

def test_generate_local_streams_tokens_and_accumulates():
    tokens: list[str] = []

    async def on_tok(t):
        tokens.append(t)

    async def _run():
        async with _mock_client(ndjson_handler(["Hel", "lo", "!"])) as client:
            return await generate_local(state(), client, on_token=on_tok)

    text, failure, usage = asyncio.run(_run())
    assert failure is None
    assert tokens == ["Hel", "lo", "!"]          # each NDJSON chunk streamed live
    assert text == "Hello!"                        # ...and accumulated into the reply
    assert usage == TokenUsage(input_tokens=3, output_tokens=4, total_tokens=7)


def test_astream_run_streams_tokens_then_status(monkeypatch):
    # One memoized supervisor client so the scripted [local_llm, finish] iterator
    # advances across calls (a fresh client per call would loop to the step bound).
    sup = _SupClient(["local_llm", "finish"])
    monkeypatch.setattr(orchestrator, "get_client", lambda *a, **k: sup)
    install_ollama(monkeypatch, ndjson_handler(["Hi", " there"]))

    async def _run():
        payload = IntentPayload(intent="chat", confidence=0.95, raw_input="x", source="cli")
        return [ev async for ev in orchestrator.astream_run(payload)]

    events = asyncio.run(_run())
    tokens = [e["token"] for e in events if "token" in e]
    assert tokens == ["Hi", " there"]             # streamed via on_custom_event
    assert events[-1] == {"status": "completed"}  # terminal marker last


def test_astream_run_surfaces_local_message_when_no_tokens(monkeypatch):
    # A degraded local_llm streams NO tokens (it only records a failure notice).
    # astream_run must still surface that final message so a run stamped
    # "completed" (by finish or the anti-reloop guard) does not reach the client
    # with an empty body — which the UI renders as "No reply produced".
    sup = _SupClient(["local_llm", "finish"])
    monkeypatch.setattr(orchestrator, "get_client", lambda *a, **k: sup)
    install_ollama(monkeypatch, raising_handler(httpx.ConnectError("refused")))

    async def _run():
        payload = IntentPayload(intent="chat", confidence=0.95, raw_input="x", source="cli")
        return [ev async for ev in orchestrator.astream_run(payload)]

    events = asyncio.run(_run())
    tokens = [e["token"] for e in events if "token" in e]
    assert tokens == ["local inference failure: unreachable"]  # failure surfaced, not dropped
    assert events[-1] == {"status": "completed"}
