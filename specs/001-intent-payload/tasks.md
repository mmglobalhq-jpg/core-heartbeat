---
description: "Task list for Intent Payload implementation"
---

# Tasks: Intent Payload

**Input**: Design documents from `/specs/001-intent-payload/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: INCLUDED — the spec defines verifiable success criteria (SC-001…SC-005) and the user explicitly requested tests for strict/immutable behavior. Test tasks are first-class here.

**Organization**: Grouped by user story (US1 P1 → US2 P2 → US3 P3). Note: all three stories add fields to the **same** `IntentPayload` class in `models.py`, so field-implementation tasks across stories are sequential (same file), not parallel. Test tasks live in separate files and can parallelize.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: US1 / US2 / US3 — maps to the user stories in spec.md
- Exact file paths are included in each task

## Path Conventions

Existing flat single-project layout (per plan.md Structure Decision): `models.py`, `router.py`, `main.py` at repo root; tests under `tests/`.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Lock dependencies and create the test scaffold before any code.

- [X] T001 [P] Generate `requirements.txt` at repo root pinning the full runtime stack from the venv: `pydantic==2.13.4`, `fastapi==0.139.0`, plus their pinned transitive deps (capture via `./venv/bin/pip freeze`).
- [X] T002 [P] Add pytest as a dev dependency: install into the venv (`./venv/bin/pip install pytest`) and record it in `requirements-dev.txt` at repo root (pinned to the installed version), with a comment that it is dev-only.
- [X] T003 [P] Create the test package: `tests/__init__.py` and an empty `tests/test_intent_payload.py` with module docstring referencing `specs/001-intent-payload/quickstart.md`.
- [X] T004 Verify tooling: `./venv/bin/python -m pytest tests/ -q` runs (collects 0 tests, exits clean) and `./venv/bin/python -c "import pydantic, fastapi"` succeeds.

**Checkpoint**: Deps locked, pytest runnable, empty suite green.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Create the strict/immutable `IntentPayload` shell that ALL user stories add fields to. This encodes the gateway-wide decisions (`extra="forbid"`, `frozen=True`) once.

**⚠️ CRITICAL**: No user-story field work can begin until this shell exists.

- [X] T005 In `models.py`, add imports (`from typing import Any`, `from datetime import datetime, timezone`, `from pydantic import BaseModel, ConfigDict, Field`) and define the `IntentPayload(BaseModel)` class with `model_config = ConfigDict(extra="forbid", frozen=True)` and a class docstring pointing to `specs/001-intent-payload/data-model.md`. No fields yet.

**Checkpoint**: `from models import IntentPayload` imports; the class is strict + frozen and ready for fields.

---

## Phase 3: User Story 1 - Route a recognized intent to the correct handler (Priority: P1) 🎯 MVP

**Goal**: The payload carries a required, unique intent identity and a (possibly empty) name→value parameter map — everything the router needs to dispatch without the original request.

**Independent Test**: Construct a payload with a known `intent` and `entities`; confirm both are readable and intact, and that omitting `intent` fails at construction while omitting `entities` yields `{}`.

### Implementation for User Story 1

- [X] T006 [US1] In `models.py`, add `intent: str` (required) to `IntentPayload` — the unique intent identity (FR-001, FR-002; data-model VR-1).
- [X] T007 [US1] In `models.py`, add `entities: dict[str, Any] = Field(default_factory=dict)` to `IntentPayload` — name→value map, empty allowed (FR-005; data-model VR-3, research R2). Same file as T006, so runs after it.

### Tests for User Story 1

- [X] T008 [P] [US1] In `tests/test_intent_payload.py`, add tests: valid construction exposes `intent` + `entities` intact (Scenario 1); missing/empty `intent` raises `pydantic.ValidationError` (Scenario 2); omitted `entities` == `{}` and a populated map is preserved key-for-key (Scenario 4).

**Checkpoint**: MVP — a payload can be built with an identity + params and rejects a missing identity. This alone lets the router dispatch on `intent`.

---

## Phase 4: User Story 2 - Make confidence-aware routing decisions (Priority: P2)

**Goal**: The payload carries a bounded confidence scalar so downstream policy can accept/reject/escalate. Out-of-range values are rejected at construction (threshold policy stays in `router.py`).

**Independent Test**: Construct payloads across the confidence range; `0.0` and `1.0` accepted, `-0.1` and `1.1` rejected at construction.

### Implementation for User Story 2

- [X] T009 [US2] In `models.py`, add `confidence: float = Field(ge=0.0, le=1.0)` (required) to `IntentPayload` (FR-003, FR-004; data-model VR-2; research R1). Same file as US1 fields, runs after Phase 3.

### Tests for User Story 2

- [X] T010 [P] [US2] In `tests/test_intent_payload.py`, add a parametrized test over confidence values `0.0` and `1.0` (accepted) and `-0.1`, `1.1` (raise `ValidationError`) — Scenario 3, SC-003.

**Checkpoint**: Payload guarantees a valid bounded confidence; router can apply its own threshold.

---

## Phase 5: User Story 3 - Trace and audit an intent back to its origin (Priority: P3)

**Goal**: The payload carries the verbatim raw input plus metadata (creation timestamp, source) so any downstream decision is traceable and reproducible.

**Independent Test**: Construct a payload; recover `raw_input`, a tz-aware `timestamp`, and an identifiable `source` from the payload alone.

### Implementation for User Story 3

- [X] T011 [US3] In `models.py`, add `raw_input: str` (required, stored verbatim) to `IntentPayload` (FR-006; data-model VR-4).
- [X] T012 [US3] In `models.py`, add `source: str` (required, non-empty) to `IntentPayload` (FR-008; data-model VR-5, research R5). Same file as T011.
- [X] T013 [US3] In `models.py`, add `timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))` — tz-aware UTC, overridable (FR-007; data-model, research R4). Same file as T011/T012.

### Tests for User Story 3

- [X] T014 [P] [US3] In `tests/test_intent_payload.py`, add tests: `raw_input` recovered verbatim; `timestamp` auto-defaults to a tz-aware (UTC) value when omitted and is overridable when supplied; `source` required and non-empty (Scenario 6, SC-005).

**Checkpoint**: All six fields present; origin fully reconstructable from the payload.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Verify the gateway-wide guarantees that span all fields — strictness, immutability, and lossless serialization — plus final validation.

- [X] T015 [P] In `tests/test_intent_payload.py`, add a strictness test: constructing with an unknown/extra field raises `pydantic.ValidationError` (`extra="forbid"`; data-model VR-6, user decision).
- [X] T016 [P] In `tests/test_intent_payload.py`, add an immutability test: mutating any field on a constructed instance raises `ValidationError`/`FrozenInstanceError` (`frozen=True`; user decision).
- [X] T017 [P] In `tests/test_intent_payload.py`, add a round-trip test: `model_validate(p.model_dump(mode="json")) == p`, asserting `timestamp` survives as ISO-8601 and no field is lost (FR-010, SC-004; research R6).
- [X] T018 Run the full quickstart validation: `./venv/bin/python -m pytest tests/test_intent_payload.py -v` (all Scenarios 1–6 + strict/immutable/round-trip green) and the manual smoke check from `quickstart.md`.
- [X] T019 [P] Confirm `requirements.txt` and `requirements-dev.txt` reflect the actual venv (re-run `./venv/bin/pip freeze` and diff); commit them alongside `models.py` and the test suite.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: no dependencies — start immediately. T001–T003 are [P]; T004 gates on them.
- **Foundational (Phase 2)**: depends on Setup — **blocks all user stories** (creates the class shell).
- **User Stories (Phase 3–5)**: each depends on Foundational. Because they all edit `IntentPayload` in `models.py`, the *field* tasks are effectively sequential (P1 → P2 → P3). Their *test* tasks (T008, T010, T014) are independent and can be written in parallel once their fields exist.
- **Polish (Phase 6)**: depends on all fields existing (needs the complete model for round-trip/strict/immutable tests).

### Within Each User Story

- Field task(s) in `models.py` first, then that story's test task.
- Same-file field tasks (e.g. T006→T007, T011→T012→T013) run in sequence, not parallel.

### Parallel Opportunities

- Phase 1: T001, T002, T003 in parallel (distinct files); T004 after.
- Test tasks T008, T010, T014 target the same test file — treat as append-only and serialize edits if worked concurrently, or split into per-story test modules if true parallelism is wanted.
- Phase 6: T015, T016, T017, T019 are logically independent checks (T018 runs them all together at the end).

---

## Parallel Example: Setup

```bash
# Distinct files — safe to do together:
Task: "Generate requirements.txt from pip freeze"          # T001
Task: "Install pytest and record requirements-dev.txt"     # T002
Task: "Create tests/__init__.py and tests/test_intent_payload.py" # T003
```

---

## Implementation Strategy

### MVP First (User Story 1 only)

1. Phase 1: Setup (deps locked, pytest runnable).
2. Phase 2: Foundational (strict/frozen `IntentPayload` shell).
3. Phase 3: US1 (`intent` + `entities` + tests).
4. **STOP and VALIDATE**: router can dispatch on `intent`; missing identity rejected. Demo-able MVP.

### Incremental Delivery

1. Setup + Foundational → shell ready.
2. US1 → identity + params (MVP).
3. US2 → bounded confidence.
4. US3 → raw input + traceability metadata.
5. Polish → strict/immutable/round-trip guarantees verified end-to-end.

---

## Notes

- All six fields live in one `IntentPayload` class in `models.py`; the story split is about *capability slices*, not separate files.
- `extra="forbid"` + `frozen=True` are set once in Phase 2 and verified in Phase 6 — they are hard requirements, not options.
- Routing/handler mapping and confidence-threshold policy are intentionally absent (they belong to `router.py`, per spec Assumptions §2–§3).
- Commit after each phase; each checkpoint is independently testable.
