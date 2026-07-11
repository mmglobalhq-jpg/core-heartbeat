"""Gateway routing endpoints for core-heartbeat.

Exposes POST /intent (validate + threshold-check an IntentPayload) and
GET /health (liveness). See specs/002-gateway-routing/ for the contract.

The confidence-threshold policy lives here (per the IntentPayload spec's
assumptions). No handler dispatch happens in this MVP.
"""

import asyncio
import json
import logging
from collections.abc import Mapping
from os import environ

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from auth import SANDBOX_USER_ID, resolve_user_id
from models import (
    DocumentParseRequest,
    DocumentParseResult,
    HealthStatus,
    IntentAccepted,
    IntentPayload,
    KbIngestRequest,
    ThresholdRejected,
    TitleRequest,
    TitleResponse,
)
from orchestrator import astream_run, build_ollama_client, generate_title
from orchestrator import run as run_orchestration
from services import documents as docstore
from services import kb as kbstore
from services.document_parser import parse_document

logger = logging.getLogger(__name__)

MAX_DOC_BYTES = 20 * 1024 * 1024  # 20 MB upload cap

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
    is streamed. Each SSE frame is ``data: <json>\\n\\n`` carrying one of:
      - ``{"token": ...}``     — an assistant text chunk (local_llm).
      - ``{"tool_call": {"name", "args", "result"}}`` — a vault tool turn, so the
        UI can show a reading/searching-the-vault indicator (feature 007).
      - ``{"status": ...}``    — the terminal run status (always last).
    (See orchestrator.astream_run for the node-vs-token granularity note.) A
    threshold rejection returns the normal 422 JSON envelope (nothing to stream).
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


@router.post("/title")
async def make_title(payload: TitleRequest) -> TitleResponse:
    """Summarize a conversation into a short sidebar title via the local model.

    Best-effort and side-effect-free: runs one non-streaming Ollama call and
    returns ``{"title": <label>}`` — or ``{"title": null}`` when the model is
    unavailable, refuses, or emits junk, so the caller keeps its existing title.
    Empty ``messages`` fail IntentPayload-style validation (422) via the request
    model + the guard below. No confidence/threshold or vault involvement.
    """
    if not payload.messages:
        return JSONResponse(
            status_code=422, content={"detail": "messages must not be empty"}
        )
    # Shared, keep-alive client (not closed per-request — see build_ollama_client).
    title = await generate_title(payload.messages, build_ollama_client())
    return TitleResponse(title=title)


@router.post("/documents/parse", response_model=None)
async def parse_document_endpoint(
    payload: DocumentParseRequest,
    user_id: str = Depends(resolve_user_id),
) -> DocumentParseResult | JSONResponse:
    """Parse an already-uploaded document to plain text and store it alongside the
    original. Docs are per-user (Storage path is keyed by user_id), so a real
    identity is required. Never 500s: parse/storage failures return an ``error``
    result the frontend reflects on the doc chip.
    """
    if user_id == SANDBOX_USER_ID:
        return JSONResponse(
            status_code=401, content={"detail": "authentication required for documents"}
        )
    try:
        data = await asyncio.to_thread(docstore.fetch_original, user_id, payload.doc_id)
    except Exception:
        return DocumentParseResult(status="error", error="original file not found")
    if len(data) > MAX_DOC_BYTES:
        return DocumentParseResult(status="error", error="file exceeds the 20 MB limit")
    try:
        text = await parse_document(data, payload.filename, payload.content_type)
    except Exception as exc:  # docling failure / timeout — surface, don't crash
        logger.warning("document parse failed for doc_id=%s: %s", payload.doc_id, exc)
        return DocumentParseResult(status="error", error=f"{type(exc).__name__}: {exc}"[:200])
    await asyncio.to_thread(docstore.store_extracted, user_id, payload.doc_id, text)
    return DocumentParseResult(status="ready", char_count=len(text))


@router.post("/kb/ingest", response_model=None)
async def kb_ingest(
    payload: KbIngestRequest,
    user_id: str = Depends(resolve_user_id),
) -> JSONResponse:
    """Add an already-uploaded document to the knowledge base (async → returns job_id).

    Fetches the uploaded original from the caller's user-docs storage and forwards the
    bytes to the KB service. Private by default (owner = user_id); ``scope=global`` is
    admin-only (profiles.is_admin) — the KB service is told owner=global. This gateway
    is the trust boundary: the global scope is NEVER taken from an unverified client.
    """
    if user_id == SANDBOX_USER_ID:
        return JSONResponse(status_code=401, content={"detail": "authentication required for the knowledge base"})

    owner = user_id
    if payload.scope == "global":
        try:
            admin = await kbstore.is_admin(user_id)
        except Exception:
            admin = False
        if not admin:
            return JSONResponse(status_code=403, content={"detail": "admin required to add global documents"})
        owner = kbstore.GLOBAL_OWNER

    try:
        data = await asyncio.to_thread(docstore.fetch_original, user_id, payload.doc_id)
    except Exception:
        return JSONResponse(status_code=404, content={"detail": "original file not found"})
    if len(data) > MAX_DOC_BYTES:
        return JSONResponse(status_code=413, content={"detail": "file exceeds the 20 MB limit"})

    try:
        result = await kbstore.ingest(owner, payload.filename, data)
    except Exception as exc:  # KB unreachable / rejected — surface, don't crash
        logger.warning("kb ingest failed for doc_id=%s: %s", payload.doc_id, exc)
        return JSONResponse(status_code=502, content={"detail": "knowledge base ingest failed"})
    return JSONResponse(status_code=200, content=result)


@router.get("/kb/jobs/{job_id}", response_model=None)
async def kb_job(job_id: str, user_id: str = Depends(resolve_user_id)) -> JSONResponse:
    """Poll a KB ingest job's status (proxied from the KB service)."""
    try:
        return JSONResponse(status_code=200, content=await kbstore.get_job(job_id, user_id))
    except Exception as exc:
        logger.warning("kb job fetch failed for %s: %s", job_id, exc)
        return JSONResponse(status_code=502, content={"detail": "knowledge base unavailable"})


@router.get("/kb/documents", response_model=None)
async def kb_documents(user_id: str = Depends(resolve_user_id)) -> JSONResponse:
    """List the caller's knowledge-base documents (own + global)."""
    if user_id == SANDBOX_USER_ID:
        return JSONResponse(status_code=401, content={"detail": "authentication required for the knowledge base"})
    try:
        return JSONResponse(status_code=200, content=await kbstore.list_documents(user_id))
    except Exception as exc:
        logger.warning("kb documents fetch failed: %s", exc)
        return JSONResponse(status_code=502, content={"detail": "knowledge base unavailable"})


@router.api_route("/health", methods=["GET", "HEAD"])
def health() -> HealthStatus:
    """Liveness check: reports the gateway is online. No body, no side effects.

    Accepts HEAD as well as GET so uptime probes that issue ``curl -I`` /
    HEAD-only monitors get a 200 instead of a 405 (Starlette does not auto-add
    HEAD for GET routes).
    """
    return HealthStatus()
