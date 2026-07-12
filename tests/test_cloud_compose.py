"""Tests for cloud answer composition (orchestrator.generate_cloud + local_llm path).

No network: a fake async-streaming Gemini client is injected via get_client. The
default (no COMPOSE_MODEL) keeps composition local, so the rest of the suite is
unaffected — these tests set COMPOSE_MODEL explicitly.
"""
import asyncio

import orchestrator
from models import IntentPayload, TokenUsage, WorkerFailure


def _state(raw="hi"):
    return {
        "intent": IntentPayload(intent="chat", confidence=0.9, raw_input=raw, source="t"),
        "messages": [],
        "prior_context": [],
        "documents": "",
        "user_id": "sandbox-user",
        "step": 1,
    }


# --- fake async-streaming Gemini client -------------------------------------

class _Usage:
    def __init__(self, p, c, t):
        self.prompt_token_count = p
        self.candidates_token_count = c
        self.total_token_count = t


class _Chunk:
    def __init__(self, text=None, usage=None):
        self.text = text
        self.usage_metadata = usage


class _AsyncStream:
    def __init__(self, chunks):
        self._it = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


class _Models:
    def __init__(self, chunks, exc=None):
        self._chunks = chunks
        self._exc = exc

    async def generate_content_stream(self, model, contents, config=None):
        if self._exc is not None:
            raise self._exc
        return _AsyncStream(self._chunks)


class _FakeGemini:
    def __init__(self, chunks=(), exc=None):
        self.aio = type("Aio", (), {"models": _Models(list(chunks), exc)})()


# --- provider selection -----------------------------------------------------

def test_compose_defaults_to_local(monkeypatch):
    monkeypatch.delenv("COMPOSE_MODEL", raising=False)
    assert orchestrator._is_local_compose() is True
    monkeypatch.setenv("COMPOSE_MODEL", "gemini-2.5-flash")
    assert orchestrator._is_local_compose() is False
    monkeypatch.setenv("COMPOSE_MODEL", "ollama")
    assert orchestrator._is_local_compose() is True


# --- generate_cloud ---------------------------------------------------------

def test_generate_cloud_streams_and_reports_usage(monkeypatch):
    monkeypatch.setenv("COMPOSE_MODEL", "gemini-2.5-flash")
    chunks = [_Chunk("Hello "), _Chunk("world", _Usage(5, 2, 7))]
    monkeypatch.setattr(orchestrator, "get_client", lambda *a, **k: _FakeGemini(chunks))
    emitted = []

    async def emit(t):
        emitted.append(t)

    text, failure, usage = asyncio.run(orchestrator.generate_cloud(_state(), on_token=emit))
    assert failure is None
    assert text == "Hello world"
    assert emitted == ["Hello ", "world"]
    assert usage.total_tokens == 7


def test_generate_cloud_no_client_returns_failure(monkeypatch):
    monkeypatch.setenv("COMPOSE_MODEL", "gemini-2.5-flash")
    monkeypatch.setattr(orchestrator, "get_client", lambda *a, **k: None)
    text, failure, _ = asyncio.run(orchestrator.generate_cloud(_state()))
    assert text is None and failure.category == "unreachable"


def test_generate_cloud_partial_stream_then_error_keeps_partial(monkeypatch):
    monkeypatch.setenv("COMPOSE_MODEL", "gemini-2.5-flash")
    # Emit one good chunk, then a chunk whose .text access raises mid-stream.
    class _Boom:
        @property
        def text(self):
            raise RuntimeError("stream dropped")
        usage_metadata = None
    monkeypatch.setattr(orchestrator, "get_client",
                        lambda *a, **k: _FakeGemini([_Chunk("partial "), _Boom()]))
    text, failure, _ = asyncio.run(orchestrator.generate_cloud(_state()))
    assert failure is None          # partial answer kept, not a hard failure
    assert text == "partial "


# --- local_llm fallback -----------------------------------------------------

def test_local_llm_falls_back_to_local_when_cloud_fails(monkeypatch):
    monkeypatch.setenv("COMPOSE_MODEL", "gemini-2.5-flash")

    async def cloud_fail(state, on_token=None):
        return None, WorkerFailure(category="unreachable", detail="down"), TokenUsage()

    async def local_ok(state, client, on_token=None):
        return "local answer", None, TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2)

    monkeypatch.setattr(orchestrator, "generate_cloud", cloud_fail)
    monkeypatch.setattr(orchestrator, "generate_local", local_ok)
    monkeypatch.setattr(orchestrator, "build_ollama_client", lambda: None)

    out = asyncio.run(orchestrator.local_llm(_state()))
    assert out["messages"][0].content == "local answer"
    assert out["visited"] == ["local_llm"]


def test_local_llm_uses_cloud_when_configured(monkeypatch):
    monkeypatch.setenv("COMPOSE_MODEL", "gemini-2.5-flash")
    monkeypatch.setattr(orchestrator, "get_client",
                        lambda *a, **k: _FakeGemini([_Chunk("cloud answer", _Usage(3, 2, 5))]))
    # If it tried to build the Ollama client we'd know it didn't use cloud.
    def _boom():
        raise AssertionError("must not touch local Ollama when cloud compose succeeds")
    monkeypatch.setattr(orchestrator, "build_ollama_client", _boom)

    out = asyncio.run(orchestrator.local_llm(_state()))
    assert out["messages"][0].content == "cloud answer"
    assert out["usage"].total_tokens == 5
