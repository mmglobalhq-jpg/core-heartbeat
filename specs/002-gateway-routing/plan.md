# Implementation Plan: Gateway Routing Interface

**Branch**: `002-gateway-routing` | **Date**: 2026-07-01 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/002-gateway-routing/spec.md`

## Summary

Build the core-heartbeat gateway: a FastAPI application (`main.py`) that includes a router (`router.py`) exposing two endpoints — `POST /intent` (submit an `IntentPayload` for evaluation) and `GET /health` (liveness). The submission flow is: validate against the strict `IntentPayload` contract → compare confidence against an environment-driven acceptance threshold (default `0.5`, inclusive `>=`) → return one of three outcomes through a **shared response envelope** (`accepted`, `threshold_rejected`, `validation_rejected`), each carrying an optional `usage`/metadata map reserved for future token/cost pass-through. Contract-validation failures are reshaped from FastAPI's default 422 into the same envelope so all outcomes are uniform and distinguishable. No handler dispatch.

## Technical Context

**Language/Version**: Python 3.14.4 (venv)

**Primary Dependencies**: FastAPI 0.139.0, Pydantic 2.13.4, Starlette 1.3.1, Uvicorn 0.49.0 (all installed). Reuses `IntentPayload` from `models.py` (feature 001).

**Storage**: N/A — intents are evaluated in-flight, not persisted.

**Testing**: pytest (installed, feature 001) + FastAPI in-process test client. The client needs an httpx-compatible package (see research R6 — currently absent in venv); resolved as a dev-dependency task. Core decision logic is also written as pure, HTTP-independent helpers so it is unit-testable without the client.

**Target Platform**: Linux server (WSL2 dev); ASGI app served by uvicorn.

**Project Type**: Single project — small web service (`main.py` app, `router.py` endpoints, `models.py` schemas).

**Performance Goals**: Not a stated objective for this MVP; correctness of the three outcomes is the goal. Per-request work is validation + one float comparison (sub-millisecond).

**Constraints**: All responses share one envelope (FR-013); the three outcomes must be distinguishable from the response alone (SC-004); threshold read from env at startup with a safe default and a clear failure on out-of-range/unparseable config (FR-009/FR-012); `/health` has no body and no side effects (FR-014).

**Scale/Scope**: Two endpoints, a handful of response models, one config loader, and an exception handler. ~2 production files touched (`main.py`, `router.py`) plus response models added to `models.py`.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Constitution (`.specify/memory/constitution.md`) is the unpopulated template — no ratified principles, no enforceable gates. **Gate status: PASS (vacuously).**

Applied defaults (consistent with feature 001): simplicity/YAGNI (no persistence, auth, or dispatch — all explicitly out of scope), test-first (acceptance scenarios map to endpoint + unit tests), no scope creep (exactly two endpoints, no handler mapping).

*Post-Phase 1 re-check*: Design adds only response models, a config loader, and an exception handler — no new architectural surface, no new runtime deps (httpx is dev-only). **PASS.**

## Project Structure

### Documentation (this feature)

```text
specs/002-gateway-routing/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   └── openapi.yaml      # Phase 1 — /intent + /health contract & response envelopes
├── checklists/
│   └── requirements.md  # from /speckit-specify
└── tasks.md             # Phase 2 (/speckit-tasks — NOT created here)
```

### Source Code (repository root)

```text
main.py          # create_app() factory: builds FastAPI, reads threshold at startup,
                 #   stores it on app.state, registers router + exception handler.
router.py        # APIRouter: POST /intent, GET /health; pure helpers for the
                 #   threshold decision and envelope construction.
models.py        # IntentPayload (001) + NEW response models: Outcome enum,
                 #   GatewayResponse base, IntentAccepted, ThresholdRejected,
                 #   ValidationRejected, HealthStatus.

tests/
├── test_intent_payload.py     # existing (001)
├── test_gateway_endpoints.py  # NEW — HTTP-level: /intent (3 outcomes), /health
└── test_gateway_logic.py      # NEW — pure helpers: threshold decision, config loader
```

**Structure Decision**: Keep the established flat layout. Response models go in `models.py` alongside `IntentPayload` (single import surface, small project). The app is built by a `create_app()` factory so tests can construct an app with a specific threshold via environment, satisfying SC-005 without a live restart. Decision logic (threshold comparison, envelope building, threshold parsing) lives in importable helpers in `router.py` so it can be unit-tested independently of the ASGI/HTTP layer.

## Complexity Tracking

> No constitution violations to justify — section intentionally empty.
