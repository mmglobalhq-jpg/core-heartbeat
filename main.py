"""core-heartbeat gateway application.

Builds the FastAPI app via create_app(): resolves the confidence threshold from
the environment at startup, mounts the router, and reshapes request-validation
failures into the shared ValidationRejected envelope. See
specs/002-gateway-routing/ for the contract.
"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from models import ValidationRejected
from router import load_confidence_threshold, router


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


def create_app() -> FastAPI:
    """Construct a configured gateway app.

    Reads the acceptance threshold once at startup (failing fast on
    misconfiguration) and stores it on app.state so the router can read it per
    request. Using a factory lets tests build apps with different thresholds.
    """
    app = FastAPI(title="core-heartbeat gateway", version="0.1.0")
    app.state.confidence_threshold = load_confidence_threshold()
    app.include_router(router)
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)
    return app


app = create_app()
