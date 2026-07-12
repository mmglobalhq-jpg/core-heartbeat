"""HTTP-level tests for the gateway endpoints via FastAPI's in-process client.

Covers POST /intent (accept / threshold-reject / validation-reject) and
GET /health, plus the shared-envelope invariants (Scenarios 1-6, 8; SC-001..SC-008).
"""

import httpx
import pytest
from starlette.testclient import TestClient

import orchestrator
from main import create_app
from models import RoutingDecision


def make_client(monkeypatch, threshold=None):
    """Build a client whose app was configured with the given threshold via env."""
    if threshold is None:
        monkeypatch.delenv("HEARTBEAT_CONFIDENCE_THRESHOLD", raising=False)
    else:
        monkeypatch.setenv("HEARTBEAT_CONFIDENCE_THRESHOLD", str(threshold))
    return TestClient(create_app())


# --- scripted fake supervisor client (feature 004; no network) --------------

class _Resp:
    def __init__(self, choice):
        self.parsed = RoutingDecision(next_node=choice)
        self.text = None
        self.usage_metadata = None


class _FakeClient:
    def __init__(self, choices):
        it = iter(choices)

        class _Models:
            def generate_content(self, model, contents, config):
                try:
                    return _Resp(next(it))
                except StopIteration:
                    return _Resp("finish")

        self.models = _Models()


GREET_PLAN = ["tool_execution", "local_llm", "finish"]


# Mocked Ollama counts preserve the historical stub totals (local 10/20/30 +
# tool 5/0/5 = 35) so accepted-run usage assertions stay stable (feature 005).
def _ollama_handler(request):
    return httpx.Response(
        200,
        json={"response": "[local] mocked inference", "prompt_eval_count": 10, "eval_count": 20},
    )


def install_ollama(monkeypatch, handler=_ollama_handler):
    """Inject a MockTransport-backed Ollama client so local_llm makes no real call."""
    monkeypatch.setattr(
        orchestrator,
        "build_ollama_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def install_supervisor(monkeypatch, choices=GREET_PLAN):
    """Inject scripted supervisor + MockTransport Ollama clients for accepted runs."""
    client = _FakeClient(choices)
    monkeypatch.setattr(orchestrator, "get_client", lambda *a, **k: client)
    install_ollama(monkeypatch)


def valid_payload(**overrides):
    body = {
        "intent": "greet",
        "confidence": 0.9,
        "entities": {"name": "Ada"},
        "raw_input": "hello",
        "source": "http",
    }
    body.update(overrides)
    return body


# --- US1: accept ------------------------------------------------------------

def test_accept_confident_intent(monkeypatch):
    install_supervisor(monkeypatch)  # feature 004: inject scripted fake model client
    client = make_client(monkeypatch)  # default threshold 0.5
    r = client.post("/intent", json=valid_payload(confidence=0.9))
    assert r.status_code == 200
    body = r.json()
    assert body["outcome"] == "accepted"
    assert body["accepted"] is True
    assert body["intent"] == "greet"
    # Feature 003: accepted intents trigger orchestration; usage is now populated
    # (was null in feature 002) and an orchestration outcome is present.
    assert isinstance(body["usage"], dict)
    assert body["orchestration"] is not None
    assert body["orchestration"]["nodes_executed"] == ["tool_execution", "local_llm"]


def test_accept_exactly_at_threshold(monkeypatch):
    client = make_client(monkeypatch, threshold=0.7)
    r = client.post("/intent", json=valid_payload(confidence=0.7))
    assert r.status_code == 200
    assert r.json()["outcome"] == "accepted"


# --- Feature 006: model_preference flows through the request body -----------

class _FakeOpenAIClient:
    """Minimal OpenAI-shaped fake returning a fixed decision (no network, no SDK)."""

    def __init__(self, next_node="finish"):
        content = f'{{"next_node": "{next_node}"}}'

        class _Completions:
            def create(self, model, messages, response_format):
                msg = type("Msg", (), {"content": content})()
                choice = type("Choice", (), {"message": msg})()
                return type("Resp", (), {"choices": [choice], "usage": None})()

        self.chat = type("Chat", (), {"completions": _Completions()})()


def test_model_preference_selects_provider_via_gateway(monkeypatch):
    # A gpt-4o-mini preference in the body drives the OpenAI provider path; inject
    # an OpenAI-shaped client so the accepted run routes with no real network.
    seen = {}

    def fake_get_client(model_preference=orchestrator.DEFAULT_MODEL_PREFERENCE):
        seen["model"] = model_preference
        return _FakeOpenAIClient(next_node="finish")

    monkeypatch.setattr(orchestrator, "get_client", fake_get_client)
    client = make_client(monkeypatch)  # default threshold 0.5
    r = client.post("/intent", json=valid_payload(confidence=0.9, model_preference="gpt-4o-mini"))
    assert r.status_code == 200
    body = r.json()
    assert body["outcome"] == "accepted"
    assert seen["model"] == "gpt-4o-mini"  # the dropdown value reached the Supervisor
    assert body["orchestration"]["status"] == "completed"


def test_model_preference_defaults_when_omitted(monkeypatch):
    # Omitting model_preference defaults to gemini (the field's default).
    seen = {}

    def fake_get_client(model_preference=orchestrator.DEFAULT_MODEL_PREFERENCE):
        seen["model"] = model_preference
        return _FakeClient(["finish"])  # gemini-shaped

    monkeypatch.setattr(orchestrator, "get_client", fake_get_client)
    client = make_client(monkeypatch)
    r = client.post("/intent", json=valid_payload(confidence=0.9))
    assert r.status_code == 200
    assert seen["model"] == "gemini-2.5-flash"


# --- US2: threshold rejection ----------------------------------------------

def test_reject_below_threshold(monkeypatch):
    client = make_client(monkeypatch, threshold=0.7)
    r = client.post("/intent", json=valid_payload(confidence=0.6))
    assert r.status_code == 422
    body = r.json()
    assert body["outcome"] == "threshold_rejected"
    assert body["confidence"] == 0.6
    assert body["threshold"] == 0.7
    assert "accepted" not in body  # never reported as accepted


def test_threshold_is_env_driven(monkeypatch):
    # Same mid-range intent, opposite decisions under two thresholds (SC-005).
    low = make_client(monkeypatch, threshold=0.5)
    assert low.post("/intent", json=valid_payload(confidence=0.6)).json()["outcome"] == "accepted"
    high = make_client(monkeypatch, threshold=0.9)
    assert high.post("/intent", json=valid_payload(confidence=0.6)).json()["outcome"] == "threshold_rejected"


# --- US3: validation rejection ---------------------------------------------

@pytest.mark.parametrize(
    "bad",
    [
        {"confidence": 0.9, "raw_input": "x", "source": "http"},          # missing intent
        valid_payload(confidence=1.5),                                     # out of range
        {**valid_payload(), "surprise": "boom"},                          # extra field (forbid)
        valid_payload(confidence="high"),                                 # wrong type
    ],
)
def test_validation_rejected(monkeypatch, bad):
    client = make_client(monkeypatch)
    r = client.post("/intent", json=bad)
    assert r.status_code == 422
    body = r.json()
    assert body["outcome"] == "validation_rejected"
    assert isinstance(body["errors"], list) and len(body["errors"]) >= 1


def test_out_of_range_confidence_is_validation_not_threshold(monkeypatch):
    # An out-of-range confidence must be a validation error, never a threshold one.
    client = make_client(monkeypatch, threshold=0.5)
    r = client.post("/intent", json=valid_payload(confidence=1.5))
    assert r.status_code == 422
    assert r.json()["outcome"] == "validation_rejected"


# --- US4: health ------------------------------------------------------------

def test_health(monkeypatch):
    client = make_client(monkeypatch)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "online"
    assert body["service"] == "core-heartbeat"


def test_health_accepts_head(monkeypatch):
    # Uptime probes issue HEAD (curl -I); the route must answer 200, not 405.
    client = make_client(monkeypatch)
    r = client.head("/health")
    assert r.status_code == 200


# --- Polish: shared-envelope consistency (SC-004, SC-007) -------------------

def test_all_outcomes_share_envelope(monkeypatch):
    install_supervisor(monkeypatch)
    client = make_client(monkeypatch, threshold=0.7)
    accepted = client.post("/intent", json=valid_payload(confidence=0.9)).json()
    threshold = client.post("/intent", json=valid_payload(confidence=0.6)).json()
    validation = client.post("/intent", json=valid_payload(confidence=1.5)).json()

    outcomes = {accepted["outcome"], threshold["outcome"], validation["outcome"]}
    assert outcomes == {"accepted", "threshold_rejected", "validation_rejected"}
    # Every response shares the envelope: the `usage` key is always present.
    for body in (accepted, threshold, validation):
        assert "usage" in body
    # Feature 003: accepted usage is populated (engine ran); rejections stay null
    # because the engine is not triggered for them (FR-013, SC-006).
    assert isinstance(accepted["usage"], dict)
    assert threshold["usage"] is None
    assert validation["usage"] is None


# --- Feature 003: orchestration integration (SC-005, SC-006) ----------------

def test_accepted_intent_triggers_orchestration_with_usage(monkeypatch):
    install_supervisor(monkeypatch)  # scripted greet plan
    client = make_client(monkeypatch)  # default threshold 0.5
    r = client.post("/intent", json=valid_payload(intent="greet", confidence=0.9))
    assert r.status_code == 200
    body = r.json()
    assert body["orchestration"]["nodes_executed"] == ["tool_execution", "local_llm"]
    assert body["orchestration"]["status"] == "completed"
    assert body["usage"]["total_tokens"] == 35  # fixed increments (10+20+30) + (5+0+5)


def test_accepted_no_key_degrades_gracefully(monkeypatch):
    # Feature 004: with no GEMINI_API_KEY and no injected client, the Supervisor
    # cannot route -> the run degrades safely but still returns HTTP 200 (SC-003).
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = make_client(monkeypatch)
    r = client.post("/intent", json=valid_payload(confidence=0.9))
    assert r.status_code == 200
    body = r.json()
    assert body["outcome"] == "accepted"
    assert body["orchestration"]["status"] == "degraded"
    assert body["orchestration"]["nodes_executed"] == []


def test_accepted_local_worker_failure_degrades_gracefully(monkeypatch):
    # Feature 005: supervisor routes to local_llm but the Ollama daemon is
    # unreachable. The run still returns HTTP 200, terminates, and records a
    # categorized local-worker failure in the orchestration messages (SC-003).
    client_ = _FakeClient(["local_llm", "finish"])
    monkeypatch.setattr(orchestrator, "get_client", lambda *a, **k: client_)

    def _refuse(request):
        raise httpx.ConnectError("refused")

    install_ollama(monkeypatch, _refuse)
    client = make_client(monkeypatch)
    r = client.post("/intent", json=valid_payload(confidence=0.9))
    assert r.status_code == 200
    body = r.json()
    assert body["outcome"] == "accepted"
    assert "local_llm" in body["orchestration"]["nodes_executed"]
    messages = body["orchestration"]["messages"]
    assert any("local inference failure: unreachable" in m["content"] for m in messages)


def test_rejections_do_not_trigger_orchestration(monkeypatch):
    client = make_client(monkeypatch, threshold=0.7)
    # below threshold -> engine not triggered, usage null, no orchestration data
    low = client.post("/intent", json=valid_payload(confidence=0.6)).json()
    assert low["outcome"] == "threshold_rejected"
    assert low["usage"] is None
    assert "orchestration" not in low
    # invalid -> engine not triggered
    bad = client.post("/intent", json=valid_payload(confidence=1.5)).json()
    assert bad["outcome"] == "validation_rejected"
    assert bad["usage"] is None
    assert "orchestration" not in bad
