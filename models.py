"""Data models for core-heartbeat.

See specs/001-intent-payload/data-model.md for the IntentPayload field contract
and validation rules (VR-1..VR-7), and specs/002-gateway-routing/data-model.md
for the gateway response envelope.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

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
        model_preference: Optional caller-selected Supervisor model (feature 006).
            Drives which provider the orchestrator's Supervisor uses; defaults to
            the Gemini flash model. Surfaced so a UI model dropdown can steer
            routing. Unknown values fall back to the default provider.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    intent: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    entities: dict[str, Any] = Field(default_factory=dict)
    raw_input: str
    source: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=_utc_now)
    model_preference: str | None = "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Orchestration models (feature 003).
# Carried through a LangGraph run and returned to the gateway. See
# specs/003-langgraph-orchestration/data-model.md.
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """An ordered entry appended to the run's message history as a node executes."""

    source: str  # which node produced it: "supervisor" / "local_llm" / "tool_execution"
    content: str
    step: int


class TokenUsage(BaseModel):
    """Additive token accumulator; model_dump()s into the response envelope usage field."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class OrchestrationOutcome(BaseModel):
    """The structured result of an orchestration run (FR-010)."""

    # "completed" / "halted_step_bound" / "error" / "degraded"
    # ("degraded" = terminated safely after a Supervisor model-routing failure, feature 004)
    status: str
    nodes_executed: list[str]
    messages: list[Message]
    usage: TokenUsage
    steps: int


class ToolArgs(BaseModel):
    """Arguments for a vault tool call — only the field the chosen tool needs is set.

    Kept as a fixed, typed object (rather than a free-form dict) so it maps cleanly
    onto every provider's structured-output schema. Mirrors the vault tools'
    parameters: ``read_user_note(filename)``, ``search_user_vault(query)``,
    ``write_user_note(filename, content)``.
    """

    filename: str | None = None
    query: str | None = None
    content: str | None = None


class RoutingDecision(BaseModel):
    """Strict schema for the Supervisor's model routing decision (features 004 + 007).

    Used both as the model's response_schema and to re-validate the response.
    ``next_node`` is constrained to the graph's routing vocabulary (FR-002).

    When ``next_node == "tool_execution"`` the model also names the vault tool to
    run (``tool_name``) and supplies its arguments (``tool_args``); both are absent
    for the ``local_llm``/``finish`` routes. This is the tool-calling loop: the
    supervisor threads the named call to the tool_execution node, whose isolated
    result is appended to the history and fed back to the model.
    """

    next_node: Literal["local_llm", "tool_execution", "finish"]
    tool_name: Literal["read_user_note", "search_user_vault", "write_user_note"] | None = None
    tool_args: ToolArgs = Field(default_factory=ToolArgs)


class RoutingFailure(BaseModel):
    """A categorized Supervisor routing failure, recorded for observability (FR-008)."""

    category: Literal[
        "missing_credential", "auth", "timeout", "network", "invalid_output"
    ]
    detail: str


class WorkerFailure(BaseModel):
    """A categorized local-worker inference failure, recorded for observability (feature 005).

    Parallel to RoutingFailure but for the local Ollama worker node. Ollama is a
    local, keyless service, so there is no credential/auth category; the detail is
    a short, secret-free diagnostic (exception type/message or ``"HTTP <code>"``).
    See specs/005-local-ollama-worker/data-model.md (FR-008).
    """

    category: Literal["unreachable", "timeout", "invalid_output"]
    detail: str


# ---------------------------------------------------------------------------
# Gateway response envelope (feature 002).
# Every gateway response shares GatewayResponse so callers get a consistent
# shape with an optional usage/metadata map (FR-013). The `outcome` enum is the
# authoritative discriminator between the three outcomes (SC-004).
# ---------------------------------------------------------------------------


class Outcome(str, Enum):
    """The three distinguishable outcomes of an intent submission."""

    ACCEPTED = "accepted"
    THRESHOLD_REJECTED = "threshold_rejected"
    VALIDATION_REJECTED = "validation_rejected"


class GatewayResponse(BaseModel):
    """Shared envelope carried by every gateway response.

    ``usage`` is present in the schema but unpopulated in this MVP; it is
    reserved for future token-count and cost pass-through (FR-013, SC-007).
    """

    outcome: Outcome
    usage: dict[str, Any] | None = None


class IntentAccepted(GatewayResponse):
    """Success: a validated intent whose confidence met the threshold (HTTP 200).

    For accepted intents the orchestration engine is triggered (feature 003):
    ``orchestration`` carries the run outcome and ``usage`` (inherited) is
    populated from it (FR-011, FR-012).
    """

    outcome: Outcome = Outcome.ACCEPTED
    intent: str
    accepted: bool = True
    detail: str = "Intent received and validated."
    orchestration: OrchestrationOutcome | None = None


class ThresholdRejected(GatewayResponse):
    """Policy rejection: valid intent below the acceptance threshold (HTTP 422)."""

    outcome: Outcome = Outcome.THRESHOLD_REJECTED
    intent: str
    confidence: float
    threshold: float
    detail: str = "Confidence below acceptance threshold."


class ValidationRejected(GatewayResponse):
    """Contract rejection: submission failed IntentPayload validation (HTTP 422)."""

    outcome: Outcome = Outcome.VALIDATION_REJECTED
    errors: list[dict[str, Any]]
    detail: str = "Submission failed intent contract validation."


class HealthStatus(BaseModel):
    """Liveness response for GET /health (HTTP 200)."""

    status: str = "online"
    service: str = "core-heartbeat"
