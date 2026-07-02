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
    monkeypatch.setattr(orchestrator, "get_client", lambda: None)
    update = supervisor(state())
    assert update["next"] == "finish"
    assert update["status"] == "degraded"
    assert "missing_credential" in update["messages"][0].content


def test_supervisor_routes_on_valid_decision(monkeypatch):
    monkeypatch.setattr(orchestrator, "get_client",
                        lambda: returns(parsed=RoutingDecision(next_node="local_llm")))
    update = supervisor(state())
    assert update["next"] == "local_llm"
    assert "status" not in update or update["status"] != "degraded"


def test_supervisor_finish_completes(monkeypatch):
    monkeypatch.setattr(orchestrator, "get_client",
                        lambda: returns(parsed=RoutingDecision(next_node="finish")))
    update = supervisor(state())
    assert update["next"] == "finish"
    assert update["status"] == "completed"


def test_supervisor_degrades_on_failure(monkeypatch):
    monkeypatch.setattr(orchestrator, "get_client", lambda: raises(httpx.TimeoutException("t")))
    update = supervisor(state())
    assert update["next"] == "finish" and update["status"] == "degraded"
    assert "timeout" in update["messages"][0].content


def test_supervisor_step_bound_no_model_call(monkeypatch):
    # at the bound, the supervisor must not call the model
    def _boom():
        raise AssertionError("model should not be called at the step bound")
    monkeypatch.setattr(orchestrator, "get_client", lambda: FakeClient(_boom))
    s = state()
    s["step"] = orchestrator.MAX_STEPS
    update = supervisor(s)
    assert update["next"] == "finish" and update["status"] == "halted_step_bound"


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
