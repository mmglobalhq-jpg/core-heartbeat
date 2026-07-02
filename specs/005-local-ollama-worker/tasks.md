---
description: "Task list for Local Ollama Worker Node implementation"
---

# Tasks: Local Ollama Worker Node

**Input**: Design documents from `/specs/005-local-ollama-worker/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/local_worker.md, quickstart.md

**Tests**: INCLUDED — spec defines verifiable SC-001…SC-005 and the user asked for a fake/mock test block. **All tests use `httpx.MockTransport` — no real network, no Ollama daemon, no spend.**

**Organization**: Grouped by user story (US1 P1 → US2 P2 → US3 P3). The `WorkerFailure` model, the `build_ollama_client()` factory, the async `generate_local()` helper, the async `local_llm` rewrite, and the async `run()`/gateway ripple are shared plumbing built in Foundational. US1 tests real-text routing, US2 tests usage capture, US3 tests failure degradation.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependency on incomplete tasks)
- **[Story]**: US1 / US2 / US3 — maps to spec.md user stories
- Exact file paths included in each task

## Path Conventions

Flat single-project layout: `orchestrator.py`, `models.py`, `router.py`, `main.py` at repo root; tests under `tests/`.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Confirm the runtime already carries everything needed — no new packages.

- [X] T001 Verify `httpx` (already runtime) exposes the async + mock seam: `./venv/bin/python -c "import httpx; httpx.AsyncClient; httpx.MockTransport; print('ok')"` succeeds on 3.14. Confirm no edit to `requirements.txt`/`requirements-dev.txt` is needed (httpx moved to runtime in feature 004).
- [X] T002 Baseline the suite before changes: `./venv/bin/python -m pytest tests/ -q` is green (79 passing) so cross-feature test updates in Phase 6 start from a known-good state.

**Checkpoint**: No new deps; async/mock primitives importable; suite green.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build the failure model, the injectable client + async model-call helper, the async `local_llm` node, and the async invocation ripple — the shared plumbing every user story exercises.

**⚠️ CRITICAL**: No user story can be tested until `local_llm` performs the (mocked) async call and degrades safely, and `run()`/gateway are async.

- [X] T003 [P] In `models.py`, add `WorkerFailure(BaseModel)` with `category: Literal["unreachable", "timeout", "invalid_output"]` and `detail: str`. Place it near `RoutingFailure` with a short docstring referencing feature 005. (FR-008)
- [X] T004 In `orchestrator.py`, add constants read at invoke time: `OLLAMA_URL_ENV="OLLAMA_URL"` (default `"http://localhost:11434/api/generate"`), `OLLAMA_MODEL_ENV="OLLAMA_MODEL"` (default `"qwen2.5:7b"`), `OLLAMA_TIMEOUT_MS_ENV="OLLAMA_TIMEOUT_MS"` (default `120_000`). Add small env-reader helpers (or inline `os.environ.get(..., default)`). Import `WorkerFailure` from `models`. Keep `MAX_STEPS`/`RECURSION_LIMIT`. (FR-010)
- [X] T005 In `orchestrator.py`, implement `build_ollama_client() -> httpx.AsyncClient`: construct an `httpx.AsyncClient` reading `OLLAMA_URL`/`OLLAMA_TIMEOUT_MS` at call time (timeout in seconds = ms/1000). This is the test seam — the node calls it at invoke time so `monkeypatch orchestrator.build_ollama_client` can inject a `MockTransport`-backed client. (FR-011)
- [X] T006 In `orchestrator.py`, add `_build_local_prompt(state) -> str` deriving the prompt from `state["intent"]` + `state["messages"]` (same style as the Supervisor's `_build_prompt`).
- [X] T007 In `orchestrator.py`, implement `async def generate_local(state, client) -> tuple[str | None, WorkerFailure | None, TokenUsage]`: `POST` `OLLAMA_URL` with JSON `{"model": OLLAMA_MODEL, "prompt": _build_local_prompt(state), "stream": False}`; require 2xx (else `invalid_output`, detail `"HTTP <code>"`); parse JSON, require `response` key → text (may be `""`); usage `TokenUsage(input=body.get("prompt_eval_count",0) or 0, output=body.get("eval_count",0) or 0, total=input+output)`. Map exceptions: `httpx.TimeoutException`→`timeout`; `httpx.ConnectError`/other `httpx.TransportError`→`unreachable`; JSON/`KeyError`/decode→`invalid_output`; catch-all→`unreachable`. MUST NOT raise; return exactly one of (text, failure) non-None. (FR-001, FR-005, FR-006, FR-007, FR-008)
- [X] T008 In `orchestrator.py`, rewrite `local_llm` as `async def`: `client = build_ollama_client()`; use `async with client:`; `text, failure, usage = await generate_local(state, client)`; on success record `Message(source="local_llm", content=text, step=state["step"])` + add `usage`; on failure record `Message(source="local_llm", content=f"local inference failure: {failure.category}", step=state["step"])` + `TokenUsage()`. Both paths append `"local_llm"` to `visited` and `step += 1`. Do NOT set `next`/`status`. Remove the old `LLM_USAGE` stub constant if now unused. (FR-002, FR-003, FR-004, FR-009)
- [X] T009 In `orchestrator.py`, make `run()` `async def` and use `await graph.ainvoke(initial, config={"recursion_limit": RECURSION_LIMIT})`; keep the outcome mapping and the defensive `except` unchanged. Leave `supervisor`, `tool_execution`, `route`, `build_graph`, and module-level `graph` intact (sync nodes run fine under `ainvoke`). (FR-012)
- [X] T010 In `router.py`, make `submit_intent` `async def` and `await run_orchestration(payload)`; everything else (threshold decide, envelopes, `/health`) unchanged.

**Checkpoint**: A routed run performs a mocked async Ollama call, records real text or a categorized failure, and always terminates; `run()`/gateway are async; three-layer termination intact.

---

## Phase 3: User Story 1 - Local model produces real inference output (Priority: P1) 🎯 MVP

**Goal**: A run routed to the local worker returns the model's real generated text (mocked) instead of the stub placeholder, and control returns to the Supervisor.

**Independent Test**: With a `MockTransport` returning `{"response": "..."}`, the worker records that text and the run terminates with `local_llm` in `nodes_executed`.

### Tests for User Story 1

- [X] T011 [P] [US1] In `tests/test_local_worker.py`, add a `_mock_client(handler)` helper building `httpx.AsyncClient(transport=httpx.MockTransport(handler))`, and a `_run_generate` helper wrapping `asyncio.run(generate_local(state, client))`. Test: a handler returning 200 `{"response":"hello from qwen","prompt_eval_count":26,"eval_count":290}` yields `text=="hello from qwen"` and no failure. (SC-001)
- [X] T012 [P] [US1] In `tests/test_local_worker.py`, add a node-level test: monkeypatch `orchestrator.build_ollama_client` to a mock returning 200 `{"response":"NODE TEXT"}`, `asyncio.run(orchestrator.local_llm(state))`, assert the returned update's `messages[0].content == "NODE TEXT"`, `source == "local_llm"`, and `"local_llm" in visited`. (SC-001, FR-003/FR-004)

**Checkpoint**: MVP — the local worker returns real (mocked) text and stays in the graph flow.

---

## Phase 4: User Story 2 - Local model token usage stays observable (Priority: P2)

**Goal**: Ollama's reported token counts are extracted and summed field-wise into the run's usage tracker; absence is a zero-contribution no-op.

**Independent Test**: A mock reporting known counts increases the run totals by exactly those amounts; a mock reporting none leaves them unchanged with no error.

### Tests for User Story 2

- [X] T013 [P] [US2] In `tests/test_local_worker.py`, add usage-extraction tests for `generate_local`: 200 `{"response":"x","prompt_eval_count":26,"eval_count":290}` → `TokenUsage(26, 290, 316)`; 200 `{"response":"x"}` (no counts) → `TokenUsage()` (zeros) and no error; 200 with only `prompt_eval_count` → output/total account for the missing count as zero. (SC-002, FR-005, FR-006)
- [X] T014 [P] [US2] In `tests/test_local_worker.py`, add a node-level usage test: with a mock returning known counts, `orchestrator.local_llm(state)`'s update carries the matching `TokenUsage` on the `usage` channel (so `add_usage` will sum it). (SC-002)

**Checkpoint**: Real (mocked) local usage flows into the run totals; missing counts are safe.

---

## Phase 5: User Story 3 - Local worker degrades safely when the model is unavailable (Priority: P3)

**Goal**: Every failure mode (unreachable, timeout, non-2xx, unusable body) resolves to a recorded, categorized failure; the run terminates, no crash/hang.

**Independent Test**: A mock raising each failure (and returning each bad response) yields a `WorkerFailure` of the right category and a terminating run.

### Tests for User Story 3

- [X] T015 [P] [US3] In `tests/test_local_worker.py`, add failure-mapping tests for `generate_local`: handler raising `httpx.ConnectError` → `unreachable`; raising `httpx.ReadTimeout` → `timeout`; returning 404 → `invalid_output` (detail `"HTTP 404"`); returning 200 with a body missing `response` → `invalid_output`; returning 200 non-JSON text → `invalid_output`. Each returns a `WorkerFailure`, `text is None`, never raises. (SC-003, FR-008)
- [X] T016 [P] [US3] In `tests/test_local_worker.py`, add a node-level degradation test: monkeypatch `build_ollama_client` to a mock raising `httpx.ConnectError`; `orchestrator.local_llm(state)`'s update records `Message(content="local inference failure: unreachable")`, `usage == TokenUsage()`, and `"local_llm" in visited`; it does not set `next`/`status`. (SC-003, FR-009)
- [X] T017 [US3] In `tests/test_local_worker.py`, add a full-run degradation test: scripted supervisor fake (feature 004 `get_client`) routes `local_llm` then `finish`, with `build_ollama_client` mocked to raise `httpx.ReadTimeout`; `asyncio.run(orchestrator.run(intent))` returns an `OrchestrationOutcome` that terminates, lists `local_llm` in `nodes_executed`, and whose messages include `local inference failure: timeout`; no exception propagates. (SC-003, SC-005, FR-009, FR-012)

**Checkpoint**: All local-worker failure categories degrade safely, observably, and terminate.

---

## Phase 6: Update existing tests for the live async worker

**Purpose**: Feature 003/004 tests assumed a synchronous `local_llm` stub with fixed `LLM_USAGE` (10/20/30) and a sync `run()`. Update them to inject a mocked Ollama client, expect real (mocked) text, updated usage totals, and drive the async path. (Required cross-feature update.)

- [X] T018 In `tests/test_orchestrator.py`, add a `MockTransport` Ollama fixture (default 200 `{"response":"[local] mocked", "prompt_eval_count":<a>, "eval_count":<b>}`) and monkeypatch `orchestrator.build_ollama_client`. Wrap `run()` calls in `asyncio.run(...)`. Update the greet-plan trace assertions: `nodes_executed == ["local_llm","tool_execution"]` still holds; `local_llm`'s message content is the mocked text (not the old stub string); update the `total_tokens` expectation to the new sum (mocked local counts + `TOOL_USAGE` 5 + any supervisor-fake usage). Keep the determinism test.
- [X] T019 In `tests/test_gateway_endpoints.py`, monkeypatch both `orchestrator.get_client` (scripted supervisor fake) and `orchestrator.build_ollama_client` (mock Ollama) for the accepted-run integration tests so `test_accept_confident_intent`, `test_all_outcomes_share_envelope`, and `test_accepted_intent_triggers_orchestration_with_usage` still see `nodes_executed == ["local_llm","tool_execution"]` and assert the new usage total + mocked local text. The async endpoint runs unchanged via `TestClient`. Add a degraded-local test: with `build_ollama_client` mocked to raise `httpx.ConnectError` (supervisor still routes to `local_llm`), an accepted intent returns HTTP 200, the run terminates, and orchestration messages include `local inference failure: unreachable`. Rejection tests unchanged (engine not triggered).

**Checkpoint**: Full suite green with the live async worker; no network, no daemon.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Full validation and no-network/config confirmation.

- [X] T020 Run the full suite: `./venv/bin/python -m pytest tests/ -v` (all green — new local-worker tests, updated orchestrator + gateway, unchanged 001/002/004). Confirm NO real network/daemon access occurred (all Ollama calls via `MockTransport`).
- [X] T021 [P] Confirm `requirements.txt`/`requirements-dev.txt` are unchanged and no new package was added (httpx already runtime); re-diff against `./venv/bin/pip freeze` if desired.
- [X] T022 [P] Grep `orchestrator.py` to confirm the three `OLLAMA_*` knobs are read from the env with the documented defaults and at invoke time (not captured at graph-build), and that no secret is logged (Ollama is keyless — verify `detail` carries only exception type/message or `"HTTP <code>"`). (FR-010)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: T001→T002 (verify → baseline).
- **Foundational (Phase 2)**: depends on Setup. **Blocks all user stories.** T003 (`models.py`) [P] alongside orchestrator scaffolding; within `orchestrator.py`: T004→T005→T006→T007→T008→T009 sequential (same file); T010 (`router.py`) after T009.
- **US1/US2/US3 (Phases 3–5)**: depend on Foundational; each appends to `tests/test_local_worker.py` (serialize edits within the file).
- **Phase 6**: depends on Foundational (needs async worker + async run); updates existing test files.
- **Polish (Phase 7)**: depends on all prior.

### Within Files

- `models.py`: T003 (single addition).
- `orchestrator.py`: T004→T005→T006→T007→T008→T009 (same file, sequential).
- `tests/test_local_worker.py`: T011–T017 append-only (serialize edits).

### Parallel Opportunities

- T003 (`models.py`) runs alongside orchestrator scaffolding, but T007/T008 need T003's `WorkerFailure`.
- Test-authoring tasks marked [P] touch the same new file, so they parallelize only conceptually — serialize the actual edits.
- Polish T021/T022 are independent checks; T020 runs everything.

---

## Implementation Strategy

### MVP First (US1)

1. Phase 1 Setup (verify deps, baseline).
2. Phase 2 Foundational (model + client factory + async helper + async node + async run/gateway).
3. Phase 3 US1 (real mocked text routes correctly).
4. **STOP and VALIDATE**: the local worker returns real (mocked) text and stays in the flow.

### Incremental Delivery

1. Setup + Foundational → worker is live-capable + safe + async.
2. US1 → real text (MVP).
3. US2 → usage capture.
4. US3 → all failure modes degrade safely.
5. Phase 6 → existing tests updated (mock client).
6. Polish → full validation + no-network/config confirmation.

---

## Notes

- **No real network/daemon in tests** (hard requirement): every Ollama interaction uses `httpx.MockTransport`; `build_ollama_client` is monkeypatched. CI needs no daemon and spends nothing.
- **Three-layer termination preserved** (hard requirement): a failed/looping local worker is bounded by the Supervisor's finish/degrade + `MAX_STEPS` + `recursion_limit`. The worker never sets routing.
- **Async ripple** (T009/T010): `run()`→async (`ainvoke`), `submit_intent`→async; sync supervisor/tool nodes run unchanged under `ainvoke`; direct `run()` tests use `asyncio.run()` (no `pytest-asyncio`).
- **Keyless**: Ollama is local/unauthenticated — no credential category, no secret to protect beyond avoiding leaking arbitrary body content in `detail`.
- **tool_execution stays a stub**; only the local worker becomes live this feature. Supervisor (004) unchanged.
- Commit after each phase; each checkpoint is independently testable.
