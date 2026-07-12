"""core-heartbeat gateway application.

Builds the FastAPI app via create_app(): resolves the confidence threshold from
the environment at startup, mounts the router, and reshapes request-validation
failures into the shared ValidationRejected envelope. See
specs/002-gateway-routing/ for the contract.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from models import ValidationRejected
from orchestrator import warm_ollama_models
from router import load_confidence_threshold, router

# Surface app-level INFO logs (e.g. auth.resolve_user_id's resolved user_id).
# uvicorn configures only its own loggers and leaves the root logger unhandled,
# so without this our module loggers' INFO records are dropped. uvicorn's loggers
# do not propagate to root, so this does not duplicate its access log lines.
logging.basicConfig(level=logging.INFO)


def _validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Reshape FastAPI's default 422 into the shared ValidationRejected envelope.

    Triggered when the POST /intent body violates the IntentPayload contract
    (missing field, wrong type, out-of-range confidence, or unknown field via
    extra="forbid"). Keeps validation failures distinct from threshold
    rejections while sharing one envelope (FR-007, FR-013, SC-003).
    """
    errors = [
        {"type": e.get("type"), "loc": list(e.get("loc", [])), "msg": e.get("msg")}
        for e in exc.errors()
    ]
    body = ValidationRejected(errors=errors)
    return JSONResponse(status_code=422, content=body.model_dump(mode="json"))


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Fire the Ollama model warmup on startup WITHOUT blocking readiness.

    The warmup loads qwen2.5:7b + nomic-embed so the first user skips the ~15s
    cold-load. It's a detached background task (not awaited) so uvicorn starts
    serving immediately; it runs concurrently and is best-effort (never raises).
    Disabled with OLLAMA_WARMUP=0. Only runs under a real ASGI lifespan (uvicorn) —
    tests build TestClient without the lifespan context, so it never fires there.
    """
    task = None
    if os.environ.get("OLLAMA_WARMUP", "1") != "0":
        task = asyncio.create_task(warm_ollama_models())
    try:
        yield
    finally:
        if task is not None and not task.done():
            task.cancel()


def create_app() -> FastAPI:
    """Construct a configured gateway app.

    Reads the acceptance threshold once at startup (failing fast on
    misconfiguration) and stores it on app.state so the router can read it per
    request. Using a factory lets tests build apps with different thresholds.
    """
    app = FastAPI(title="core-heartbeat gateway", version="0.1.0", lifespan=_lifespan)
    app.state.confidence_threshold = load_confidence_threshold()
    app.include_router(router)
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)
    return app


app = create_app()
