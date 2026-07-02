# Phase 0 Research: Gateway Routing Interface

All Technical Context unknowns resolved. No open NEEDS CLARIFICATION.

## R1 — Shared response envelope shape

- **Decision**: One base model `GatewayResponse` with `outcome: Outcome` (enum) and `usage: dict[str, Any] | None = None`. Three success/rejection variants subclass it:
  - `IntentAccepted` — adds `intent: str`, `accepted: bool = True`, `detail: str`.
  - `ThresholdRejected` — adds `intent: str`, `confidence: float`, `threshold: float`, `detail: str`.
  - `ValidationRejected` — adds `errors: list[dict[str, Any]]`, `detail: str`.
  `Outcome` is a `str, Enum` with members `accepted`, `threshold_rejected`, `validation_rejected`.
- **Rationale**: The `outcome` enum is the authoritative discriminator required by SC-004 — a caller distinguishes all three cases from the body regardless of transport status. Sharing the `usage` field on the base satisfies FR-013 (consistent envelope across every response). Subclassing keeps each variant's extra detail explicit while guaranteeing the common envelope.
- **Alternatives considered**:
  - One flat model with all optional fields — loses per-variant clarity; easy to return an internally inconsistent object.
  - Distinguish only by HTTP status — insufficient for SC-004 ("from the response alone" includes body) and collides (see R3).

## R2 — Where response models live

- **Decision**: Add the response models to `models.py` alongside `IntentPayload`.
- **Rationale**: Small, flat project; a single `from models import ...` surface. Feature 001 already established `models.py` as the schema home.
- **Alternatives considered**: A new `schemas.py`/`responses.py` — premature file-splitting for ~5 small models; revisit if `models.py` grows large.

## R3 — HTTP status mapping for the three outcomes

- **Decision**:
  - `accepted` → **200 OK**
  - `validation_rejected` → **422 Unprocessable Entity**
  - `threshold_rejected` → **422 Unprocessable Entity**
  Both rejection kinds return 422; the `outcome` field (R1) is the authoritative discriminator between them.
- **Rationale**: 200 vs 422 cleanly separates accepted from rejected at the transport layer. Both rejections are "well-formed but not acceptable," which is exactly 422's semantics; 422 is also FastAPI's native validation status, so `validation_rejected` keeps the conventional code. Distinguishing threshold-vs-validation is delegated to the `outcome` enum, which SC-004 already requires callers to read. This avoids inventing an unconventional status for policy rejection.
- **Alternatives considered**:
  - Distinct codes (e.g. 400 validation / 422 threshold) — marginally more distinguishable by status, but fights FastAPI's 422 convention for body validation and adds no capability beyond the `outcome` field. Rejected for simplicity.
  - 200 for threshold_rejected with `accepted: false` — contradicts the spec, which frames below-threshold as a rejection/error, and weakens SC-002's "never reported as accepted."

## R4 — Reshaping validation errors into the envelope

- **Decision**: Register a FastAPI exception handler for `RequestValidationError` (raised when the `IntentPayload` body fails: missing field, wrong type, out-of-range confidence via `ge/le`, or unknown field via `extra="forbid"`). The handler returns a `ValidationRejected` envelope (HTTP 422) whose `errors` list is derived from the exception's `.errors()`.
- **Rationale**: FastAPI/Pydantic intercept body validation before the endpoint body runs and emit a default 422 with their own shape. Overriding the handler is the only way to make `validation_rejected` use the shared envelope (FR-013) and be distinguishable via `outcome` (SC-004, SC-003). This also cleanly separates validation failures from threshold rejections: an out-of-range confidence is caught here as a *validation* error and never reaches the threshold comparison (matches FR-002/FR-004 ordering).
- **Alternatives considered**:
  - Accept the body as a raw `dict` and manually construct `IntentPayload` inside the endpoint to catch `ValidationError` locally — loses FastAPI's automatic OpenAPI schema for the request body and duplicates framework behavior. Rejected.
  - Leave FastAPI's default 422 shape for validation — violates the consistent-envelope requirement (FR-013).

## R5 — Threshold configuration loader

- **Decision**: Read `HEARTBEAT_CONFIDENCE_THRESHOLD` from the environment at app startup inside `create_app()`. Parsing rules:
  - Unset or blank → default `0.5`.
  - Parseable float within `[0.0, 1.0]` → use it.
  - Parseable float outside `[0.0, 1.0]`, or unparseable → raise a clear configuration error at startup (fail fast), naming the variable and the offending value.
  Store the resolved threshold on `app.state.confidence_threshold`; the endpoint reads it via a small dependency.
- **Rationale**: Satisfies FR-009/FR-012 (env-driven, safe default) and the edge cases "missing/blank → default" and "out-of-range → surfaced clearly." Reading at startup means misconfiguration is caught immediately, not on first request. Storing on `app.state` + `create_app()` factory makes SC-005 testable: build two apps under two env values and observe opposite decisions on the same intent — no live restart needed.
- **Alternatives considered**:
  - Read env per-request — re-parses on every call and defers misconfig detection to request time. Rejected (spec says "at startup").
  - Pydantic `BaseSettings` — heavier than needed for one scalar; a small loader function is clearer and directly testable. Revisit if config grows.

## R6 — Endpoint testing approach & the httpx dependency

- **Decision**: Test at two levels:
  1. **Pure logic** (`tests/test_gateway_logic.py`) — unit-test the threshold-decision helper and the config loader directly, no HTTP. Covers SC-002/SC-005 boundary and config edge cases without any client dependency.
  2. **HTTP endpoint** (`tests/test_gateway_endpoints.py`) — drive `/intent` and `/health` through FastAPI's in-process test client. This environment's `starlette.testclient` requires an httpx-compatible package (probe showed httpx absent and starlette demanding it), so add the client as a **dev dependency** and pin it in `requirements-dev.txt`. The exact package/version is confirmed during the setup task by installing and importing successfully.
- **Rationale**: The two-level split means the core accept/reject/config correctness is provable even if the HTTP client is finicky, while endpoint tests still verify wiring, status codes (R3), and the envelope over the wire. Keeping the client dev-only honors "no new runtime deps."
- **Alternatives considered**:
  - HTTP tests only — would make the whole suite hostage to the test-client dependency and couple logic tests to the ASGI layer. Rejected.
  - Spin up a live uvicorn server and hit it over a socket — slower, flakier, unnecessary for in-process ASGI testing. Rejected.

## R7 — Endpoint names and app composition

- **Decision**: `router.py` defines an `APIRouter` with `POST /intent` and `GET /health`. `main.py` exposes `create_app()` (builds `FastAPI`, resolves threshold, registers the router and the R4 exception handler) and a module-level `app = create_app()` for uvicorn (`uvicorn main:app`).
- **Rationale**: `create_app()` factory enables per-test configuration (R5) while `app = create_app()` gives the conventional ASGI entrypoint. `POST /intent` (singular) reads as "submit one intent for evaluation"; the endpoint evaluates, it does not create a persisted resource.
- **Alternatives considered**:
  - Define routes directly on the app in `main.py` — the spec explicitly wants endpoints in `router.py`; a factory + APIRouter keeps that separation and testability.
  - `/intents` (plural collection) — implies resource creation/listing, which this MVP does not do. Rejected.

## Cross-cutting: what stays OUT

Per spec FR-010 and Assumptions — no handler dispatch, no intent→handler mapping, no persistence, no auth, no rate limiting. The `usage`/metadata field is present in every envelope but **unpopulated** in this MVP (FR-013).
