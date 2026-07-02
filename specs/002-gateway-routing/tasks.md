---
description: "Task list for Gateway Routing Interface implementation"
---

# Tasks: Gateway Routing Interface

**Input**: Design documents from `/specs/002-gateway-routing/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/openapi.yaml, quickstart.md

**Tests**: INCLUDED — the user requested pure-logic and HTTP endpoint tests; spec defines verifiable SC-001…SC-008.

**Organization**: Grouped by user story (US1 P1 → US2 P2 → US3 P3 → US4 P3). Response models and the router are shared plumbing built in Foundational so every story has a working endpoint to exercise. Field/route tasks that touch the same file (`models.py`, `router.py`, `main.py`) run sequentially; test files are separate and parallelizable.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependency on incomplete tasks)
- **[Story]**: US1 / US2 / US3 / US4 — maps to spec.md user stories
- Exact file paths included in each task

## Path Conventions

Flat single-project layout (per plan.md): `main.py`, `router.py`, `models.py` at repo root; tests under `tests/`.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Add the HTTP test-client dev dependency and confirm the app can be served.

- [X] T001 Install an httpx-compatible test client into the venv for FastAPI's in-process `TestClient` (starlette in this env requires it). Confirm by `./venv/bin/python -c "from starlette.testclient import TestClient"` succeeding; capture the exact package+version that made it import.
- [X] T002 Pin the confirmed test-client package(s) from T001 in `requirements-dev.txt` (append under the pytest block, with a comment that it powers the FastAPI TestClient). Do not add it to `requirements.txt` (dev-only).
- [X] T003 Verify baseline: `./venv/bin/python -m pytest tests/ -q` still passes (feature 001 suite green) and `./venv/bin/python -c "import fastapi, starlette; from starlette.testclient import TestClient; print('ok')"` succeeds.

**Checkpoint**: Test client available, existing suite green.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build the response envelope, the threshold loader, the app factory + validation-error handler, and the router skeleton — the shared plumbing every user story exercises.

**⚠️ CRITICAL**: No user story can be tested until the endpoint and envelope exist.

- [X] T004 In `models.py`, add the response models below `IntentPayload`: `Outcome(str, Enum)` with members `ACCEPTED="accepted"`, `THRESHOLD_REJECTED="threshold_rejected"`, `VALIDATION_REJECTED="validation_rejected"`; base `GatewayResponse(BaseModel)` with `outcome: Outcome` and `usage: dict[str, Any] | None = None`; subclasses `IntentAccepted` (intent, accepted=True, detail; outcome default ACCEPTED), `ThresholdRejected` (intent, confidence, threshold, detail; outcome default THRESHOLD_REJECTED), `ValidationRejected` (errors: list[dict[str, Any]], detail; outcome default VALIDATION_REJECTED); and `HealthStatus` (status="online", service="core-heartbeat"). Per data-model.md. (FR-013)
- [X] T005 In `router.py`, implement the threshold loader `load_confidence_threshold(env: Mapping | None = None) -> float`: reads `HEARTBEAT_CONFIDENCE_THRESHOLD`; unset/blank → `0.5`; parseable float in `[0.0, 1.0]` → that value; parseable but out of range OR unparseable → raise a clear `ValueError`/config error naming the variable and offending value. (FR-009, FR-012)
- [X] T006 In `router.py`, implement the pure helper `decide(confidence: float, threshold: float) -> bool` returning `confidence >= threshold` (inclusive), and create `router = APIRouter()`. (US1/US2 core; SC-005)
- [X] T007 In `main.py`, implement `create_app() -> FastAPI`: call `load_confidence_threshold()`, store the result on `app.state.confidence_threshold`, include `router`, and register a `RequestValidationError` exception handler that returns a `ValidationRejected` envelope (HTTP 422) built from `exc.errors()`. Also expose module-level `app = create_app()` for `uvicorn main:app`. (FR-007, FR-013, R4/R7)

**Checkpoint**: App builds, `/intent` and `/health` are reachable (routes added next), envelope + config + handler in place.

---

## Phase 3: User Story 1 - Accept a valid, confident intent (Priority: P1) 🎯 MVP

**Goal**: `POST /intent` accepts a valid `IntentPayload` whose confidence meets the threshold and returns a 200 `IntentAccepted` envelope echoing the intent and accepted status (with the `usage` field present).

**Independent Test**: POST a valid intent with confidence ≥ threshold; expect 200, `outcome="accepted"`, `accepted=true`, echoed `intent`, `usage` present.

### Implementation for User Story 1

- [X] T008 [US1] In `router.py`, add `POST /intent` accepting an `IntentPayload` body: read the threshold from `app.state` (via `Request`/dependency), call `decide(...)`, and on `True` return `IntentAccepted(intent=payload.intent, usage=None)` with HTTP 200. (Threshold-false branch added in US2.) (FR-001, FR-005)

### Tests for User Story 1

- [X] T009 [P] [US1] In `tests/test_gateway_endpoints.py`, add tests using `TestClient(create_app())`: valid confident intent → 200, body `outcome="accepted"`, `accepted=true`, echoes `intent`, and includes `usage` key (Scenario 1); confidence exactly equal to threshold → accepted (Scenario 2, inclusive boundary). (SC-001)

**Checkpoint**: MVP — the gateway accepts and acknowledges a confident intent end-to-end.

---

## Phase 4: User Story 2 - Reject an intent below the threshold (Priority: P2)

**Goal**: A valid intent with confidence below the threshold returns a 422 `ThresholdRejected` envelope reporting submitted confidence and required threshold; never accepted.

**Independent Test**: POST a valid intent below threshold; expect 422, `outcome="threshold_rejected"`, body includes `confidence` and `threshold`.

### Implementation for User Story 2

- [X] T010 [US2] In `router.py`, extend `POST /intent`: when `decide(...)` is `False`, return a `ThresholdRejected(intent=..., confidence=payload.confidence, threshold=<threshold>)` envelope with HTTP 422 (use `JSONResponse`/`status_code` or raise a mapped exception). (FR-004, FR-006). Same file as T008 → sequential.

### Tests for User Story 2

- [X] T011 [P] [US2] In `tests/test_gateway_logic.py`, add pure-logic tests for `decide(...)`: below/at/above threshold; and for `load_confidence_threshold(...)`: unset→0.5, blank→0.5, in-range value used, out-of-range raises, unparseable raises (Scenario 7). (SC-005, config edges)
- [X] T012 [P] [US2] In `tests/test_gateway_endpoints.py`, add endpoint tests: below-threshold intent → 422, `outcome="threshold_rejected"`, reports `confidence` and `threshold`, never `accepted` (Scenario 3); build apps under two `HEARTBEAT_CONFIDENCE_THRESHOLD` env values and submit the same mid-range intent → opposite decisions (Scenario 6). (SC-002, SC-005)

**Checkpoint**: Accept and threshold-reject both work and are distinguishable.

---

## Phase 5: User Story 3 - Reject malformed submissions (Priority: P3)

**Goal**: Submissions violating the `IntentPayload` contract return a 422 `ValidationRejected` envelope (via the handler) identifying the problem, before any threshold check.

**Independent Test**: POST payloads each violating one contract rule; expect 422, `outcome="validation_rejected"`, `errors` populated, no threshold comparison.

### Tests for User Story 3

- [X] T013 [P] [US3] In `tests/test_gateway_endpoints.py`, add tests: missing required field, confidence out of `[0,1]`, unknown extra field (`extra="forbid"`), wrong type → each returns 422 with `outcome="validation_rejected"` and a non-empty `errors` list; assert an out-of-range confidence yields validation_rejected (NOT threshold_rejected), proving validation precedes threshold (Scenario 4). (SC-003)

> Note: no new implementation task — US3 is served by the `RequestValidationError` handler built in T007. This phase verifies it.

**Checkpoint**: All three intent outcomes work and are distinguishable by `outcome` (SC-004).

---

## Phase 6: User Story 4 - Verify the gateway is online (Priority: P3)

**Goal**: `GET /health` returns a structured online status with no body and no side effects.

**Independent Test**: GET `/health`; expect 200, `status="online"`, `service="core-heartbeat"`.

### Implementation for User Story 4

- [X] T014 [US4] In `router.py`, add `GET /health` returning `HealthStatus()` (200), taking no request body and touching no intent state. (FR-014). Same file as T008/T010 → sequential.

### Tests for User Story 4

- [X] T015 [P] [US4] In `tests/test_gateway_endpoints.py`, add a test: `GET /health` → 200, `status="online"`, `service="core-heartbeat"` (Scenario 8). (SC-008)

**Checkpoint**: Both endpoints live.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Verify the shared envelope invariants and run full validation.

- [X] T016 [P] In `tests/test_gateway_endpoints.py`, add an envelope-consistency test: accepted, threshold_rejected, and validation_rejected responses each carry the `usage` field (present, null/empty) and a distinct `outcome` value — proving the shared envelope and distinguishability (SC-004, SC-007).
- [X] T017 Run the full quickstart validation: `./venv/bin/python -m pytest tests/test_gateway_logic.py tests/test_gateway_endpoints.py tests/test_intent_payload.py -v` (all green) and the manual smoke check from `quickstart.md` (uvicorn + curl /intent and /health).
- [X] T018 [P] Confirm `requirements-dev.txt` reflects the venv (re-run `./venv/bin/pip freeze` and diff for the test-client package); leave `requirements.txt` unchanged (no new runtime deps).

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: no deps — start immediately. T001→T002→T003 sequential (each depends on prior).
- **Foundational (Phase 2)**: depends on Setup. **Blocks all user stories.** T004 (models) and T005 (loader) are independent; T006 depends on nothing new; T007 depends on T004 (envelope) + T005 (loader) + T006 (router).
- **User Stories (Phase 3–6)**: depend on Foundational. US1→US2 share `POST /intent` in `router.py` (sequential: T008 before T010). US3 needs no new code (handler from T007). US4 (T014) also edits `router.py` → sequential after T010.
- **Polish (Phase 7)**: depends on all endpoints + tests existing.

### Within Each User Story

- `router.py` route tasks (T008, T010, T014) are sequential (same file).
- Test tasks target `tests/test_gateway_endpoints.py` (shared) and `tests/test_gateway_logic.py` (separate). Endpoint-test tasks appending to the same file should be serialized or merged; logic tests (T011) are independent.

### Parallel Opportunities

- Phase 2: T004 and T005 in parallel (different files: models.py vs router.py). T006 also router.py → after/with T005.
- T011 (logic tests) is independent of the endpoint-test file and can run alongside endpoint work.
- Phase 7: T016 and T018 are independent checks; T017 runs everything at the end.

---

## Parallel Example: Foundational

```bash
# Different files — safe together:
Task: "Add response models (Outcome, envelope, variants, HealthStatus) in models.py"   # T004
Task: "Implement load_confidence_threshold + decide in router.py"                        # T005/T006
# Then T007 (main.py) once T004+T005+T006 land.
```

---

## Implementation Strategy

### MVP First (US1)

1. Phase 1 Setup (test client).
2. Phase 2 Foundational (envelope, loader, factory, handler, router).
3. Phase 3 US1 (`POST /intent` accept path + tests).
4. **STOP and VALIDATE**: a confident intent is accepted and acknowledged end-to-end.

### Incremental Delivery

1. Setup + Foundational → app builds, endpoints reachable.
2. US1 → accept path (MVP).
3. US2 → threshold rejection + logic/config tests.
4. US3 → validation rejection verified (handler already built).
5. US4 → `/health`.
6. Polish → envelope-consistency + full run.

---

## Notes

- 422 is used for BOTH threshold_rejected and validation_rejected (confirmed); the `outcome` enum is the authoritative discriminator (SC-004).
- The `usage` field is present but unpopulated across all envelopes in this MVP (FR-013).
- Out-of-range confidence is a VALIDATION rejection (caught by IntentPayload/`extra`+`ge/le`), never a threshold rejection — verified in T013.
- No handler dispatch (FR-010); `router.py` ends at receive → validate → threshold-check → acknowledge.
- Commit after each phase; each checkpoint is independently testable.
