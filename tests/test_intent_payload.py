"""Unit tests for the IntentPayload model.

Validates the field contract, construction-time validation, strictness,
immutability, and lossless serialization defined in:
    specs/001-intent-payload/quickstart.md
    specs/001-intent-payload/data-model.md
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from models import IntentPayload


def _valid_kwargs(**overrides):
    """A minimal set of valid constructor kwargs, with optional overrides."""
    kwargs = dict(
        intent="greet",
        confidence=0.9,
        entities={"name": "Ada"},
        raw_input="hello there",
        source="http",
    )
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# US1 - identity + entities (Scenarios 1, 2, 4)
# ---------------------------------------------------------------------------

def test_valid_construction_exposes_fields_intact():
    """Scenario 1 / SC-001: a valid payload exposes identity and params intact."""
    p = IntentPayload(**_valid_kwargs())
    assert p.intent == "greet"
    assert p.confidence == 0.9
    assert p.entities == {"name": "Ada"}
    assert p.raw_input == "hello there"
    assert p.source == "http"
    # timestamp auto-defaults to a tz-aware UTC value when not supplied.
    assert p.timestamp.tzinfo is not None
    assert p.timestamp.utcoffset() == timezone.utc.utcoffset(None)


def test_missing_intent_is_rejected():
    """Scenario 2 / FR-002 / VR-1: absent intent fails at construction."""
    kwargs = _valid_kwargs()
    del kwargs["intent"]
    with pytest.raises(ValidationError):
        IntentPayload(**kwargs)


def test_empty_intent_is_rejected():
    """FR-002 / VR-1: empty-string intent is not a valid identity."""
    with pytest.raises(ValidationError):
        IntentPayload(**_valid_kwargs(intent=""))


def test_entities_default_empty_when_omitted():
    """Scenario 4 / FR-005 / VR-3: omitted entities becomes {} (valid)."""
    kwargs = _valid_kwargs()
    del kwargs["entities"]
    p = IntentPayload(**kwargs)
    assert p.entities == {}


def test_entities_preserved_key_for_key():
    """Scenario 4 / FR-005: a populated map is preserved exactly."""
    payload_entities = {"city": "Paris", "count": 3, "flag": True, "tags": ["a", "b"]}
    p = IntentPayload(**_valid_kwargs(entities=payload_entities))
    assert p.entities == payload_entities


# ---------------------------------------------------------------------------
# US2 - confidence bounds (Scenario 3 / SC-003)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_confidence_in_range_accepted(value):
    """FR-003 / VR-2: inclusive bounds 0.0 and 1.0 (and mid-range) are accepted."""
    p = IntentPayload(**_valid_kwargs(confidence=value))
    assert p.confidence == value


@pytest.mark.parametrize("value", [-0.1, 1.1, -1.0, 2.0])
def test_confidence_out_of_range_rejected(value):
    """FR-004 / SC-003 / VR-2: out-of-range confidence fails at construction."""
    with pytest.raises(ValidationError):
        IntentPayload(**_valid_kwargs(confidence=value))


# ---------------------------------------------------------------------------
# US3 - traceability metadata (Scenario 6 / SC-005)
# ---------------------------------------------------------------------------

def test_raw_input_recovered_verbatim():
    """FR-006 / VR-4: raw input is stored and recovered unchanged."""
    original = "  MixEd Case\twith\nwhitespace  "
    p = IntentPayload(**_valid_kwargs(raw_input=original))
    assert p.raw_input == original


def test_timestamp_defaults_to_utc_aware():
    """FR-007: timestamp auto-defaults to a timezone-aware UTC value."""
    p = IntentPayload(**_valid_kwargs())
    assert p.timestamp.tzinfo is not None
    assert p.timestamp.utcoffset().total_seconds() == 0


def test_timestamp_override_respected():
    """FR-007: an explicitly supplied timestamp is used (needed for reconstruction)."""
    ts = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    p = IntentPayload(**_valid_kwargs(timestamp=ts))
    assert p.timestamp == ts


def test_source_required():
    """FR-008 / VR-5: source is required."""
    kwargs = _valid_kwargs()
    del kwargs["source"]
    with pytest.raises(ValidationError):
        IntentPayload(**kwargs)


def test_source_non_empty():
    """FR-008 / VR-5: empty source is rejected."""
    with pytest.raises(ValidationError):
        IntentPayload(**_valid_kwargs(source=""))


# ---------------------------------------------------------------------------
# Cross-cutting - strictness, immutability, round-trip (Phase 6)
# ---------------------------------------------------------------------------

def test_unknown_field_rejected():
    """VR-6 / extra='forbid': unknown fields fail at construction (fail closed)."""
    with pytest.raises(ValidationError):
        IntentPayload(**_valid_kwargs(unexpected="boom"))


def test_instance_is_frozen():
    """VR-7 / frozen=True: mutating any field after construction raises."""
    p = IntentPayload(**_valid_kwargs())
    with pytest.raises(ValidationError):
        p.intent = "changed"
    with pytest.raises(ValidationError):
        p.confidence = 0.1


def test_serialize_reconstruct_round_trip():
    """FR-010 / SC-004: model_validate(model_dump(mode='json')) == original."""
    p = IntentPayload(**_valid_kwargs(timestamp=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)))
    dumped = p.model_dump(mode="json")
    # timestamp serializes to an ISO-8601 string in JSON mode.
    assert isinstance(dumped["timestamp"], str)
    assert dumped["entities"] == {"name": "Ada"}
    reconstructed = IntentPayload.model_validate(dumped)
    assert reconstructed == p
