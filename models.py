"""Data models for core-heartbeat.

See specs/001-intent-payload/data-model.md for the field contract and
validation rules (VR-1..VR-7).
"""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> datetime:
    """Timezone-aware UTC creation time (default for IntentPayload.timestamp)."""
    return datetime.now(timezone.utc)


class IntentPayload(BaseModel):
    """Self-contained parsed intent carried from the API boundary to the router.

    Produced at the service boundary (main.py) and consumed by the router
    (router.py) to dispatch to a handler. Validated at construction, so any
    consumer may trust the contract without re-validating (FR-009).

    The payload is strict and immutable: unknown fields are rejected
    (``extra="forbid"``) and instances are frozen after construction
    (``frozen=True``). Intent->handler mapping and confidence-threshold policy
    are intentionally NOT modeled here; they belong to router.py.

    Fields:
        intent: Unique intent identity used to select a handler (FR-001/002).
        confidence: Normalized certainty in [0, 1] (FR-003/004).
        entities: Extracted parameters as a name->value map; may be empty (FR-005).
        raw_input: Verbatim original input that produced the intent (FR-006).
        timestamp: Timezone-aware UTC creation time, auto-defaulted (FR-007).
        source: Originating channel/component identifier (FR-008).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    intent: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    entities: dict[str, Any] = Field(default_factory=dict)
    raw_input: str
    source: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=_utc_now)
