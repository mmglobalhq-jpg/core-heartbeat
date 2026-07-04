"""Gateway routing endpoints for core-heartbeat.

Exposes POST /intent (validate + threshold-check an IntentPayload) and
GET /health (liveness). See specs/002-gateway-routing/ for the contract.

The confidence-threshold policy lives here (per the IntentPayload spec's
assumptions). No handler dispatch happens in this MVP.
"""

import json
from collections.abc import Mapping
from os import environ

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from auth import resolve_user_id
from models import HealthStatus, IntentAccepted, IntentPayload, ThresholdRejected
from orchestrator import astream_run
from orchestrator import run as run_orchestration

THRESHOLD_ENV_VAR = "HEARTBEAT_CONFIDENCE_THRESHOLD"
DEFAULT_THRESHOLD = 0.5


def load_confidence_threshold(env: Mapping[str, str] | None = None) -> float:
    """Resolve the acceptance threshold from the environment at startup.

    Rules (FR-009, FR-012):
      - unset or blank             -> DEFAULT_THRESHOLD (0.5)
      - parseable float in [0, 1]  -> that value
      - out of range or unparseable -> ValueError (fail fast, named clearly)
    """
    source = environ if env is None else env
    raw = source.get(THRESHOLD_ENV_VAR)
    if raw is None or raw.strip() == "":
        return DEFAULT_THRESHOLD
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"{THRESHOLD_ENV_VAR}={raw!r} is not a valid number; "
            f"expected a float in [0.0, 1.0]."
        ) from None
    if not (0.0 <= value <= 1.0):
        raise ValueError(
            f"{THRESHOLD_ENV_VAR}={raw!r} is out of range; "
            f"expected a float in [0.0, 1.0]."
        )
    return value


def decide(confidence: float, threshold: float) -> bool:
    """Accept iff confidence meets the threshold (inclusive `>=`)."""
    return confidence >= threshold


def get_threshold(request: Request) -> float:
    """Dependency: the threshold resolved at startup and stored on app.state."""
    return request.app.state.confidence_threshold


router = APIRouter()


@router.post("/intent")
async def submit_intent(
    payload: IntentPayload,
    threshold: float = Depends(get_threshold),
    user_id: str = Depends(resolve_user_id),
) -> JSONResponse:
    """Validate an intent and evaluate its confidence against the threshold.

    Body validation is handled by FastAPI against IntentPayload; failures are
    reshaped into a ValidationRejected envelope by the app's exception handler.
    A valid payload is accepted (200) or threshold-rejected (422) here. On
    acceptance the orchestration engine is triggered and its outcome + usage are
    returned (feature 003). Async because orchestration performs an asynchronous
    local inference call (feature 005).
    """
    if decide(payload.confidence, threshold):
        outcome = await run_orchestration(payload, user_id)
        body = IntentAccepted(
            intent=payload.intent,
            orchestration=outcome,
            usage=outcome.usage.model_dump(),
        )
        return JSONResponse(status_code=200, content=body.model_dump(mode="json"))
    body = ThresholdRejected(
        intent=payload.intent,
        confidence=payload.confidence,
        threshold=threshold,
    )
    return JSONResponse(status_code=422, content=body.model_dump(mode="json"))


@router.post("/intent/stream", response_model=None)
async def submit_intent_stream(
    payload: IntentPayload,
    threshold: float = Depends(get_threshold),
    user_id: str = Depends(resolve_user_id),
) -> StreamingResponse | JSONResponse:
    """Accept an intent and stream the orchestration as Server-Sent Events.

    Same validation + threshold policy as POST /intent, but on acceptance the run
    is streamed: each worker-node reply is emitted as ``data: {"token": ...}`` and
    the run ends with ``data: {"status": ...}`` (see orchestrator.astream_run for
    the node-vs-token granularity note). A threshold rejection returns the normal
    422 JSON envelope (there is nothing to stream).
    """
    if not decide(payload.confidence, threshold):
        body = ThresholdRejected(
            intent=payload.intent,
            confidence=payload.confidence,
            threshold=threshold,
        )
        return JSONResponse(status_code=422, content=body.model_dump(mode="json"))

    async def event_stream():
        async for event in astream_run(payload, user_id):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.api_route("/health", methods=["GET", "HEAD"])
def health() -> HealthStatus:
    """Liveness check: reports the gateway is online. No body, no side effects.

    Accepts HEAD as well as GET so uptime probes that issue ``curl -I`` /
    HEAD-only monitors get a 200 instead of a 405 (Starlette does not auto-add
    HEAD for GET routes).
    """
    return HealthStatus()
