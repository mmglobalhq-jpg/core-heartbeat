---
description: "Task list for API-Driven Supervisor Node implementation"
---

# Tasks: API-Driven Supervisor Node

**Input**: Design documents from `/specs/004-api-supervisor/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/supervisor.md, quickstart.md

**Tests**: INCLUDED — the user requested supervisor + updated orchestrator/gateway tests; spec defines verifiable SC-001…SC-007. **All tests use a fake client — no real network calls, no API spend.**

**Organization**: Grouped by user story (US1 P1 → US2 P2 → US3 P3). The models, client factory, model-call helper, and supervisor rewrite are shared plumbing built in Foundational; US1 tests valid routing, US2 tests failure degradation, US3 tests usage capture. `orchestrator.py`/`models.py` tasks are sequential (same file); test files are separate.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependency on incomplete tasks)
- **[Story]**: US1 / US2 / US3 — maps to spec.md user stories
- Exact file paths included in each task

## Path Conventions

Flat single-project layout: `orchestrator.py`, `models.py`, `router.py`, `main.py` at repo root; tests under `tests/`.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Pin the google-genai runtime stack and fix the runtime/dev split (httpx becomes runtime).

- [X] T001 Regenerate `requirements.txt` from `./venv/bin/pip freeze`: add the google-genai stack (google-genai, google-auth, cryptography, cffi, pyasn1, pyasn1-modules, pycparser) and **MOVE httpx + httpcore into `requirements.txt`** (google-genai requires httpx at runtime). Keep everything else. Exclude only the pytest stack (pytest, iniconfig, pluggy, pygments).
- [X] T002 Update `requirements-dev.txt`: remove httpx and httpcore (now runtime); it should carry only the pytest stack (pytest, iniconfig, pluggy, pygments) plus `-r requirements.txt`.
- [X] T003 Verify the split + import: no dev-only package appears in `requirements.txt`; `./venv/bin/python -c "from google import genai; from google.genai import types, errors; print('ok')"` succeeds on 3.14; existing suite still green (`./venv/bin/python -m pytest tests/ -q`).

**Checkpoint**: google-genai pinned as runtime, httpx moved, imports clean, suite green.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build the decision/failure models, the injectable client + model-call helper, and the rewritten supervisor — the shared plumbing every user story exercises.

**⚠️ CRITICAL**: No user story can be tested until the supervisor calls the model (via an injectable client) and degrades safely.

- [X] T004 [P] In `models.py`, add `RoutingDecision(BaseModel)` with `next_node: Literal["local_llm", "tool_execution", "finish"]`, and `RoutingFailure(BaseModel)` with `category: Literal["missing_credential", "auth", "timeout", "network", "invalid_output"]` and `detail: str`. (FR-002, FR-008)
- [X] T005 [P] In `models.py`, document/allow `OrchestrationOutcome.status` value `"degraded"` (status is a free str; add a comment enumerating completed/halted_step_bound/error/degraded). (FR-008)
- [X] T006 In `orchestrator.py`, add imports (`from google import genai`, `from google.genai import types, errors`, `import httpx`, `Literal`/`json` as needed) and constants: `MODEL_NAME = "gemini-2.5-flash"`, `GEMINI_API_KEY_ENV = "GEMINI_API_KEY"`, `REQUEST_TIMEOUT_MS = 10_000`. Remove `NOOP_INTENTS`. Keep `MAX_STEPS`/`RECURSION_LIMIT` and the stub node usage constants.
- [X] T007 In `orchestrator.py`, implement `get_client() -> genai.Client | None`: read `GEMINI_API_KEY` from the env; return `None` if unset/blank; else construct and memoize a `genai.Client(api_key=...)`. Never log the key. (FR-003)
- [X] T008 In `orchestrator.py`, implement `request_routing_decision(state, client) -> tuple[RoutingDecision | None, RoutingFailure | None, TokenUsage]`: build a routing prompt from `state["intent"]` + `state["messages"]`; call `client.models.generate_content(model=MODEL_NAME, contents=..., config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=RoutingDecision, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS)))`; validate output via `RoutingDecision.model_validate` (parsed or json.loads(text)); map exceptions to categories (missing handled by caller; `errors.ClientError` 401/403 → auth; `httpx.TimeoutException` → timeout; `httpx.TransportError`/`ConnectError`/`errors.ServerError`/other `APIError` → network; JSON/`ValidationError`/out-of-vocab → invalid_output); extract `usage_metadata` → `TokenUsage` (zeros if absent). MUST NOT raise. (FR-002, FR-004, FR-006, FR-009)
- [X] T009 In `orchestrator.py`, rewrite `supervisor(state)`: (a) if `step >= MAX_STEPS` → finish, `status="halted_step_bound"`, no model call; (b) `client = get_client()`; if `None` → finish, `status="degraded"`, record `Message("routing failure: missing_credential")`; (c) else `decision, failure, usage = request_routing_decision(state, client)`; on failure → finish, `status="degraded"`, record `Message("routing failure: <category>")`, add usage; on success → `next = decision.next_node` (status `"completed"` iff finish), record a route Message, add usage; always increment step. Keep `route`, `build_graph`, worker nodes, and module-level `graph`/`run()` intact (recursion_limit unchanged). (FR-001, FR-005, FR-007, FR-010, FR-011)

**Checkpoint**: `run()` works with an injected fake client and degrades safely with no client; three-layer termination intact.

---

## Phase 3: User Story 1 - Model-driven routing with enforced schema (Priority: P1) 🎯 MVP

**Goal**: The Supervisor gets a schema-valid decision from the model and routes to exactly that step; out-of-vocab/unparseable is rejected.

**Independent Test**: With a fake client returning each allowed decision, the Supervisor routes to the matching node; invalid output is not accepted.

### Tests for User Story 1

- [X] T010 [P] [US1] In `tests/test_supervisor.py`, add a `FakeClient` (no network) and tests: `request_routing_decision` with a fake returning `{"next_node":"local_llm"}` / `"tool_execution"` / `"finish"` yields the matching `RoutingDecision`; a fake returning `{"next_node":"banana"}` and one returning non-JSON both yield `RoutingFailure(category="invalid_output")` and no decision. (SC-001)
- [X] T011 [P] [US1] In `tests/test_supervisor.py`, add supervisor-level routing tests via monkeypatched `get_client`: a fake returning `local_llm` sets `next="local_llm"`; `finish` sets `next="finish"`/`status="completed"`. (FR-010)

**Checkpoint**: MVP — model-driven routing works and rejects invalid decisions.

---

## Phase 4: User Story 2 - Graceful degradation on any model failure (Priority: P2)

**Goal**: Every failure mode resolves to a safe finish with a recorded category; the run terminates, no crash/hang.

**Independent Test**: A fake client raising each failure mode (and one returning invalid output) each yields a degraded finish with the right category; the run still returns a complete outcome.

### Tests for User Story 2

- [X] T012 [P] [US2] In `tests/test_supervisor.py`, add failure-mapping tests for `request_routing_decision`: fake raising `errors.ClientError` (401) → `auth`; raising `httpx.TimeoutException` → `timeout`; raising `httpx.ConnectError` → `network`; and (via monkeypatched `get_client` returning None / no `GEMINI_API_KEY`) the supervisor records `missing_credential`. Each returns a `RoutingFailure`, never raises. (SC-002, SC-005)
- [X] T013 [P] [US2] In `tests/test_supervisor.py`, add full-run degradation tests: with `get_client` monkeypatched to None, `run(intent)` returns `OrchestrationOutcome(status="degraded")`, `nodes_executed == []`, and a message recording the failure category; the call returns (no hang). (SC-002, SC-003, SC-005)

**Checkpoint**: All five failure categories degrade safely and observably.

---

## Phase 5: User Story 3 - Capture model token usage (Priority: P3)

**Goal**: Reported model token usage is added to the run's usage tracker; absence is a no-op.

**Independent Test**: A fake reporting known token counts increases the run's usage totals; a fake reporting none leaves them unchanged with no error.

### Tests for User Story 3

- [X] T014 [P] [US3] In `tests/test_supervisor.py`, add usage-capture tests: a fake response carrying `usage_metadata` (e.g. prompt/candidates/total counts) causes `request_routing_decision` to return a non-zero `TokenUsage` matching those counts; a fake with no `usage_metadata` returns `TokenUsage()` (zeros) and does not error. (SC-006)

**Checkpoint**: Model usage flows into the run totals when available.

---

## Phase 6: Update existing tests for the model-driven Supervisor

**Purpose**: Feature 003 tests assumed rule-based routing; the Supervisor now needs a client. Inject a fake client so existing behavior is reproduced deterministically. (Required cross-feature update per research R6.)

- [X] T015 In `tests/test_orchestrator.py`, add a scripted `FakeClient` (returns `["local_llm","tool_execution","finish"]` in order) and a fixture/monkeypatch of `orchestrator.get_client` to inject it. Update greet-plan tests so `nodes_executed == ["local_llm","tool_execution"]` and stub `total_tokens == 35` still hold (plus any fake model usage, which should be zero for the scripted fake). Reframe the old NOOP immediate-finish test as "fake returns finish on the first call → `nodes_executed == []`". Keep the determinism test (same scripted fake → identical outcome).
- [X] T016 In `tests/test_gateway_endpoints.py`, update the accepted-run integration tests to monkeypatch `orchestrator.get_client` with the scripted fake so `test_accept_confident_intent`, `test_all_outcomes_share_envelope`, and `test_accepted_intent_triggers_orchestration_with_usage` still see `nodes_executed == ["local_llm","tool_execution"]` and `usage.total_tokens == 35`. Add a new degraded test: with NO `GEMINI_API_KEY` and no fake injected, an accepted intent returns HTTP 200 with `orchestration.status == "degraded"` and `nodes_executed == []` (SC-003). Rejection tests remain unchanged (engine not triggered).

**Checkpoint**: Full suite green with the model-driven supervisor; no network calls.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Full validation and manifest confirmation.

- [X] T017 Run the full suite: `./venv/bin/python -m pytest tests/ -v` (all green — supervisor, updated orchestrator + gateway, unchanged 001/002 payload/logic). Confirm NO real network access occurred (all model calls via fake client).
- [X] T018 [P] Confirm `requirements.txt` contains the google-genai stack + httpx/httpcore and no pytest-only packages; `requirements-dev.txt` has only the pytest stack + `-r requirements.txt`. Re-diff against `./venv/bin/pip freeze`.
- [X] T019 [P] Grep the code to confirm the credential is never logged or embedded in any recorded message/detail (search `orchestrator.py` for GEMINI_API_KEY usage; ensure only read + passed to the SDK). (FR-003)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: T001→T002→T003 sequential (freeze → dev split → verify).
- **Foundational (Phase 2)**: depends on Setup. **Blocks all user stories.** T004/T005 (models.py) [P] with orchestrator scaffolding; within `orchestrator.py`: T006→T007→T008→T009 sequential.
- **US1/US2/US3 (Phases 3–5)**: depend on Foundational; each adds tests to `tests/test_supervisor.py` (append-only, serialize).
- **Phase 6**: depends on Foundational (needs the new supervisor); updates existing test files.
- **Polish (Phase 7)**: depends on all prior.

### Within Files

- `models.py`: T004→T005 (same file).
- `orchestrator.py`: T006→T007→T008→T009 (same file, sequential).
- `tests/test_supervisor.py`: T010–T014 append-only (serialize edits).

### Parallel Opportunities

- T004/T005 (models.py) can start alongside orchestrator scaffolding, but T008/T009 need T004's RoutingDecision/RoutingFailure.
- Polish T018/T019 are independent checks; T017 runs everything.

---

## Implementation Strategy

### MVP First (US1)

1. Phase 1 Setup (pin google-genai, move httpx).
2. Phase 2 Foundational (models + client + model-call + supervisor rewrite).
3. Phase 3 US1 (valid routing + invalid-output rejection).
4. **STOP and VALIDATE**: model-driven routing works with a fake client.

### Incremental Delivery

1. Setup + Foundational → supervisor is live-capable + safe.
2. US1 → valid decisions route, invalid rejected (MVP).
3. US2 → all failure modes degrade safely.
4. US3 → usage capture.
5. Phase 6 → existing tests updated (fake client).
6. Polish → full validation + credential-safety grep.

---

## Notes

- **No real network in tests** (hard requirement): every model interaction uses a fake client; `get_client` is monkeypatched. CI needs no `GEMINI_API_KEY` and spends nothing.
- **Three-layer termination preserved** (hard requirement): supervisor finish/degrade + `MAX_STEPS` guard (before any model call) + `recursion_limit`.
- **Credential never logged** (hard requirement, T019): only read and passed to the SDK.
- **Feature 003 test updates** (T015/T016): the NOOP shortcut is gone; determinism now comes from the injected fake client. Greet-plan trace + `total_tokens==35` still reproduce.
- `local_llm`/`tool_execution` remain stubs; only the Supervisor is live.
- Commit after each phase; each checkpoint is independently testable.
