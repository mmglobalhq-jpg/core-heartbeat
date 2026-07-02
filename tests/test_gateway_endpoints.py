"""HTTP-level tests for the gateway endpoints via FastAPI's in-process client.

Covers POST /intent (accept / threshold-reject / validation-reject) and
GET /health, plus the shared-envelope invariants (Scenarios 1-6, 8; SC-001..SC-008).
"""

import pytest
from starlette.testclient import TestClient

from main import create_app


def make_client(monkeypatch, threshold=None):
    """Build a client whose app was configured with the given threshold via env."""
    if threshold is None:
        monkeypatch.delenv("HEARTBEAT_CONFIDENCE_THRESHOLD", raising=False)
    else:
        monkeypatch.setenv("HEARTBEAT_CONFIDENCE_THRESHOLD", str(threshold))
    return TestClient(create_app())


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
    client = make_client(monkeypatch)  # default threshold 0.5
    r = client.post("/intent", json=valid_payload(confidence=0.9))
    assert r.status_code == 200
    body = r.json()
    assert body["outcome"] == "accepted"
    assert body["accepted"] is True
    assert body["intent"] == "greet"
    assert "usage" in body and body["usage"] is None


def test_accept_exactly_at_threshold(monkeypatch):
    client = make_client(monkeypatch, threshold=0.7)
    r = client.post("/intent", json=valid_payload(confidence=0.7))
    assert r.status_code == 200
    assert r.json()["outcome"] == "accepted"


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


# --- Polish: shared-envelope consistency (SC-004, SC-007) -------------------

def test_all_outcomes_share_envelope(monkeypatch):
    client = make_client(monkeypatch, threshold=0.7)
    accepted = client.post("/intent", json=valid_payload(confidence=0.9)).json()
    threshold = client.post("/intent", json=valid_payload(confidence=0.6)).json()
    validation = client.post("/intent", json=valid_payload(confidence=1.5)).json()

    outcomes = {accepted["outcome"], threshold["outcome"], validation["outcome"]}
    assert outcomes == {"accepted", "threshold_rejected", "validation_rejected"}
    for body in (accepted, threshold, validation):
        assert "usage" in body and body["usage"] is None
