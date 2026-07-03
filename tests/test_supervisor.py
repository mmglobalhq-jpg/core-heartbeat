"""Tests for the model-driven Supervisor (feature 004).

Everything uses a FakeClient — NO real network calls, no API key required.
Covers valid decisions, invalid output, each failure category, and usage capture
(SC-001..SC-007, FR-002/004/005/008/009).
"""

import httpx
import pytest
from google.genai import errors

import orchestrator
from models import IntentPayload, RoutingDecision
from orchestrator import request_routing_decision, supervisor


# --- fake client (no network) -----------------------------------------------

class FakeUsage:
    def __init__(self, prompt=0, candidates=0, total=0):
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates
        self.total_token_count = total


class FakeResponse:
    def __init__(self, parsed=None, text=None, usage_metadata=None):
        self.parsed = parsed
        self.text = text
        self.usage_metadata = usage_metadata


class FakeModels:
    def __init__(self, responder):
        self._responder = responder

    def generate_content(self, model, contents, config):
        return self._responder()


class FakeClient:
    """Wraps a responder callable that returns a FakeResponse or raises."""

    def __init__(self, responder):
        self.models = FakeModels(responder)


def returns(parsed=None, text=None, usage=None):
    return FakeClient(lambda: FakeResponse(parsed=parsed, text=text, usage_metadata=usage))


def raises(exc):
    def _r():
        raise exc
    return FakeClient(_r)


def state(intent="greet", messages=None):
    return {
        "intent": IntentPayload(intent=intent, confidence=0.9, raw_input="x", source="cli"),
        "messages": messages or [],
        "usage": None,
        "visited": [],
        "step": 0,
        "next": "",
        "status": "",
    }


# --- US1: valid decisions + invalid output ----------------------------------

@pytest.mark.parametrize("choice", ["local_llm", "tool_execution", "finish"])
def test_valid_decision_parsed(choice):
    client = returns(parsed=RoutingDecision(next_node=choice))
    decision, failure, usage = request_routing_decision(state(), client)
    assert failure is None
    assert decision.next_node == choice


def test_valid_decision_from_json_text():
    client = returns(text='{"next_node": "tool_execution"}')
    decision, failure, _ = request_routing_decision(state(), client)
    assert failure is None and decision.next_node == "tool_execution"


def test_out_of_vocabulary_rejected():
    client = returns(text='{"next_node": "banana"}')
    decision, failure, _ = request_routing_decision(state(), client)
    assert decision is None
    assert failure.category == "invalid_output"


def test_unparseable_rejected():
    client = returns(text="not json at all")
    decision, failure, _ = request_routing_decision(state(), client)
    assert decision is None and failure.category == "invalid_output"


# --- US2: failure mapping ---------------------------------------------------

def test_auth_error_mapped():
    client = raises(errors.ClientError(401, {"error": {"message": "unauth"}}))
    decision, failure, _ = request_routing_decision(state(), client)
    assert decision is None and failure.category == "auth"


def test_non_auth_client_error_is_network():
    client = raises(errors.ClientError(400, {"error": {"message": "bad request"}}))
    _, failure, _ = request_routing_decision(state(), client)
    assert failure.category == "network"


def test_timeout_mapped():
    client = raises(httpx.TimeoutException("timed out"))
    _, failure, _ = request_routing_decision(state(), client)
    assert failure.category == "timeout"


def test_network_error_mapped():
    client = raises(httpx.ConnectError("connection refused"))
    _, failure, _ = request_routing_decision(state(), client)
    assert failure.category == "network"


def test_request_routing_decision_never_raises():
    # even an unexpected exception type is caught
    client = raises(RuntimeError("boom"))
    decision, failure, usage = request_routing_decision(state(), client)
    assert decision is None and failure is not None


# --- US2: supervisor-level degradation --------------------------------------

def test_supervisor_missing_credential(monkeypatch):
    monkeypatch.setattr(orchestrator, "get_client", lambda *a, **k: None)
    update = supervisor(state())
    assert update["next"] == "finish"
    assert update["status"] == "degraded"
    assert "missing_credential" in update["messages"][0].content


def test_supervisor_routes_on_valid_decision(monkeypatch):
    monkeypatch.setattr(orchestrator, "get_client",
                        lambda *a, **k: returns(parsed=RoutingDecision(next_node="local_llm")))
    update = supervisor(state())
    assert update["next"] == "local_llm"
    assert "status" not in update or update["status"] != "degraded"


def test_supervisor_finish_completes(monkeypatch):
    monkeypatch.setattr(orchestrator, "get_client",
                        lambda *a, **k: returns(parsed=RoutingDecision(next_node="finish")))
    update = supervisor(state())
    assert update["next"] == "finish"
    assert update["status"] == "completed"


def test_supervisor_degrades_on_failure(monkeypatch):
    monkeypatch.setattr(orchestrator, "get_client", lambda *a, **k: raises(httpx.TimeoutException("t")))
    update = supervisor(state())
    assert update["next"] == "finish" and update["status"] == "degraded"
    assert "timeout" in update["messages"][0].content


def test_supervisor_step_bound_no_model_call(monkeypatch):
    # at the bound, the supervisor must not call the model
    def _boom():
        raise AssertionError("model should not be called at the step bound")
    monkeypatch.setattr(orchestrator, "get_client", lambda *a, **k: FakeClient(_boom))
    s = state()
    s["step"] = orchestrator.MAX_STEPS
    update = supervisor(s)
    assert update["next"] == "finish" and update["status"] == "halted_step_bound"


def test_supervisor_no_reloop_guard_forces_finish(monkeypatch):
    # Deterministic guard: if the model re-dispatches a worker that already ran
    # (its name is in `visited`), override to a clean finish rather than looping
    # to the MAX_STEPS halt. Usage from the model call is still recorded.
    monkeypatch.setattr(
        orchestrator,
        "get_client",
        lambda *a, **k: returns(
            parsed=RoutingDecision(next_node="local_llm"),
            usage=FakeUsage(prompt=3, candidates=1, total=4),
        ),
    )
    s = state()
    s["visited"] = ["local_llm"]  # local_llm has already replied this run
    update = supervisor(s)
    assert update["next"] == "finish"
    assert update["status"] == "completed"
    assert update["usage"].total_tokens == 4
    assert "guard" in update["messages"][0].content


def test_supervisor_first_dispatch_not_guarded(monkeypatch):
    # A worker not yet in `visited` routes normally (guard must not over-fire).
    monkeypatch.setattr(
        orchestrator,
        "get_client",
        lambda *a, **k: returns(parsed=RoutingDecision(next_node="local_llm")),
    )
    update = supervisor(state())  # visited == []
    assert update["next"] == "local_llm"
    assert update.get("status") != "completed"


# --- US3: usage capture -----------------------------------------------------

def test_usage_captured_when_reported():
    client = returns(parsed=RoutingDecision(next_node="finish"),
                     usage=FakeUsage(prompt=7, candidates=3, total=10))
    _, _, usage = request_routing_decision(state(), client)
    assert usage.input_tokens == 7 and usage.output_tokens == 3 and usage.total_tokens == 10


def test_usage_absent_is_zero_no_error():
    client = returns(parsed=RoutingDecision(next_node="finish"))  # no usage_metadata
    _, _, usage = request_routing_decision(state(), client)
    assert usage.total_tokens == 0


# ---------------------------------------------------------------------------
# Feature 006: multi-model Supervisor support (OpenAI + Anthropic providers).
# All fakes are provider-shaped; NO real network, no SDK required.
# ---------------------------------------------------------------------------

# --- OpenAI-shaped fake (client.chat.completions.create) --------------------

class FakeOpenAIUsage:
    def __init__(self, prompt=0, completion=0, total=0):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = total


class FakeOpenAIResponse:
    def __init__(self, content, usage=None):
        message = type("Msg", (), {"content": content})()
        self.choices = [type("Choice", (), {"message": message})()]
        self.usage = usage


class FakeOpenAIClient:
    def __init__(self, content='{"next_node": "local_llm"}', usage=None, exc=None):
        outer = self
        self._content, self._usage, self._exc = content, usage, exc

        class _Completions:
            def create(self, model, messages, response_format):
                if outer._exc:
                    raise outer._exc
                return FakeOpenAIResponse(outer._content, outer._usage)

        self.chat = type("Chat", (), {"completions": _Completions()})()


# --- Anthropic-shaped fake (client.messages.create -> tool_use block) -------

class FakeAnthropicUsage:
    def __init__(self, input_tokens=0, output_tokens=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeToolUse:
    def __init__(self, inp):
        self.type = "tool_use"
        self.input = inp


class FakeAnthropicResponse:
    def __init__(self, decision_input, usage=None):
        self.content = [FakeToolUse(decision_input)]
        self.usage = usage


class FakeAnthropicClient:
    def __init__(self, decision_input=None, usage=None, exc=None):
        outer = self
        self._input = decision_input or {"next_node": "tool_execution"}
        self._usage, self._exc = usage, exc

        class _Messages:
            def create(self, model, max_tokens, messages, tools, tool_choice):
                if outer._exc:
                    raise outer._exc
                return FakeAnthropicResponse(outer._input, outer._usage)

        self.messages = _Messages()


# --- OpenAI provider: routing, usage, invalid output, failure ---------------

@pytest.mark.parametrize("choice", ["local_llm", "tool_execution", "finish"])
def test_openai_routes_to_each_node(choice):
    client = FakeOpenAIClient(content=f'{{"next_node": "{choice}"}}')
    decision, failure, _ = request_routing_decision(state(), client, "gpt-4o-mini")
    assert failure is None and decision.next_node == choice


def test_openai_usage_captured():
    client = FakeOpenAIClient(
        content='{"next_node": "finish"}', usage=FakeOpenAIUsage(prompt=7, completion=3, total=10)
    )
    _, _, usage = request_routing_decision(state(), client, "gpt-4o-mini")
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (7, 3, 10)


def test_openai_out_of_vocab_is_invalid_output():
    client = FakeOpenAIClient(content='{"next_node": "banana"}')
    decision, failure, _ = request_routing_decision(state(), client, "gpt-4o-mini")
    assert decision is None and failure.category == "invalid_output"


def test_openai_timeout_mapped():
    client = FakeOpenAIClient(exc=httpx.TimeoutException("t"))
    _, failure, _ = request_routing_decision(state(), client, "gpt-4o-mini")
    assert failure.category == "timeout"


def test_openai_auth_mapped_by_status():
    exc = RuntimeError("unauthorized")
    exc.status_code = 401
    client = FakeOpenAIClient(exc=exc)
    _, failure, _ = request_routing_decision(state(), client, "gpt-4o-mini")
    assert failure.category == "auth"


# --- Anthropic provider: routing, usage, invalid output, failure ------------

@pytest.mark.parametrize("choice", ["local_llm", "tool_execution", "finish"])
def test_anthropic_routes_to_each_node(choice):
    client = FakeAnthropicClient(decision_input={"next_node": choice})
    decision, failure, _ = request_routing_decision(state(), client, "claude-3.5-haiku")
    assert failure is None and decision.next_node == choice


def test_anthropic_usage_captured():
    client = FakeAnthropicClient(
        decision_input={"next_node": "finish"}, usage=FakeAnthropicUsage(input_tokens=11, output_tokens=4)
    )
    _, _, usage = request_routing_decision(state(), client, "claude-3.5-haiku")
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (11, 4, 15)


def test_anthropic_out_of_vocab_is_invalid_output():
    client = FakeAnthropicClient(decision_input={"next_node": "nope"})
    decision, failure, _ = request_routing_decision(state(), client, "claude-3.5-haiku")
    assert decision is None and failure.category == "invalid_output"


def test_anthropic_timeout_mapped():
    client = FakeAnthropicClient(exc=httpx.TimeoutException("t"))
    _, failure, _ = request_routing_decision(state(), client, "claude-3.5-haiku")
    assert failure.category == "timeout"


# --- Provider resolution + client dispatch (no SDK / no keys) ---------------

def test_resolve_model_maps_providers():
    assert orchestrator._resolve_model("gpt-4o-mini")[0] == "openai"
    assert orchestrator._resolve_model("claude-3.5-haiku")[0] == "anthropic"
    assert orchestrator._resolve_model("gemini-2.5-flash")[0] == "gemini"
    # unknown preference falls back to the default provider
    assert orchestrator._resolve_model("who-knows")[0] == "gemini"
    assert orchestrator._resolve_model(None)[0] == "gemini"


def test_get_client_none_without_keys(monkeypatch):
    monkeypatch.setattr(orchestrator, "_client_cache", {})
    for env in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    assert orchestrator.get_client("gpt-4o-mini") is None
    assert orchestrator.get_client("claude-3.5-haiku") is None
    assert orchestrator.get_client("gemini-2.5-flash") is None


# --- Supervisor honors the intent's model_preference ------------------------

def test_supervisor_selects_client_by_model_preference(monkeypatch):
    seen = {}

    def fake_get_client(model_preference=orchestrator.DEFAULT_MODEL_PREFERENCE):
        seen["model"] = model_preference
        return FakeOpenAIClient(content='{"next_node": "finish"}')

    monkeypatch.setattr(orchestrator, "get_client", fake_get_client)
    s = state()
    s["intent"] = IntentPayload(
        intent="greet", confidence=0.9, raw_input="x", source="cli",
        model_preference="gpt-4o-mini",
    )
    update = supervisor(s)
    assert seen["model"] == "gpt-4o-mini"
    assert update["next"] == "finish" and update["status"] == "completed"


def test_supervisor_default_model_preference_is_gemini(monkeypatch):
    seen = {}

    def fake_get_client(model_preference=orchestrator.DEFAULT_MODEL_PREFERENCE):
        seen["model"] = model_preference
        return returns(parsed=RoutingDecision(next_node="finish"))

    monkeypatch.setattr(orchestrator, "get_client", fake_get_client)
    update = supervisor(state())  # intent() default model_preference == gemini
    assert seen["model"] == "gemini-2.5-flash"
    assert update["status"] == "completed"
