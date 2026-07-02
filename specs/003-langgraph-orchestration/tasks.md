---
description: "Task list for Orchestration Engine (LangGraph) implementation"
---

# Tasks: Orchestration Engine

**Input**: Design documents from `/specs/003-langgraph-orchestration/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/graph.md, quickstart.md

**Tests**: INCLUDED — the user requested engine + updated endpoint tests; spec defines verifiable SC-001…SC-007.

**Organization**: Grouped by user story (US1 P1 → US2 P2 → US3 P2). The engine (state, nodes, graph, run()) is shared plumbing built in Foundational so US1/US2 can exercise it; US3 wires it into the gateway. `orchestrator.py` tasks are sequential (same file); `models.py` and test files are separate.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependency on incomplete tasks)
- **[Story]**: US1 / US2 / US3 — maps to spec.md user stories
- Exact file paths included in each task

## Path Conventions

Flat single-project layout: `orchestrator.py` (new), `models.py`, `router.py`, `main.py` at repo root; tests under `tests/`.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Pin the new LangGraph runtime dependency and confirm it imports on 3.14.

- [X] T001 Regenerate `requirements.txt` to include the LangGraph runtime stack: run `./venv/bin/pip freeze`, then write `requirements.txt` as the full freeze MINUS the dev-only packages that belong in `requirements-dev.txt` (pytest, iniconfig, pluggy, pygments, httpx, httpcore, certifi). Keep langgraph and all its transitive deps (langchain-core, langgraph-checkpoint, langgraph-prebuilt, langgraph-sdk, langsmith, orjson, ormsgpack, tenacity, jsonpatch, jsonpointer, pyyaml, uuid-utils, xxhash, zstandard, requests, urllib3, charset-normalizer, requests-toolbelt, sniffio, distro, websockets, langchain-protocol).
- [X] T002 Verify the runtime/dev split: every package pinned in `requirements.txt` is importable and none of the dev-only packages (pytest/httpx/etc.) appear in `requirements.txt`; `requirements-dev.txt` still carries the pytest + httpx stack via `-r requirements.txt`.
- [X] T003 Confirm LangGraph imports on 3.14: `./venv/bin/python -c "from langgraph.graph import StateGraph, END; from langgraph.errors import GraphRecursionError; print('ok')"` succeeds, and the existing suite is still green (`./venv/bin/python -m pytest tests/ -q`).

**Checkpoint**: LangGraph pinned as a runtime dep, imports clean, existing suite green.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build the response/data models and the full orchestration engine — the shared plumbing every user story depends on.

**⚠️ CRITICAL**: No user story can be tested until GraphState, the nodes, the compiled graph, and `run()` exist.

- [X] T004 [P] In `models.py`, add the orchestration models: `Message` (source: str, content: str, step: int); `TokenUsage` (input_tokens: int = 0, output_tokens: int = 0, total_tokens: int = 0); `OrchestrationOutcome` (status: str, nodes_executed: list[str], messages: list[Message], usage: TokenUsage, steps: int). Per data-model.md.
- [X] T005 [P] In `models.py`, extend `IntentAccepted` with `orchestration: OrchestrationOutcome | None = None` (keep all existing fields; `usage` inherited from GatewayResponse remains `dict | None`). (FR-011)
- [X] T006 Create `orchestrator.py` with imports and constants: `MAX_STEPS = 8`, `RECURSION_LIMIT = 25`, `NOOP_INTENTS = {"ping", "noop"}`, fixed increments `LLM_USAGE = TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30)` and `TOOL_USAGE = TokenUsage(input_tokens=5, output_tokens=0, total_tokens=5)`; define `add_usage(left, right) -> TokenUsage` (field-wise sum; `left is None` -> right) and the `GraphState` TypedDict with channels: `intent: IntentPayload` (no reducer), `messages: Annotated[list[Message], operator.add]`, `usage: Annotated[TokenUsage, add_usage]`, `visited: Annotated[list[str], operator.add]`, `step: Annotated[int, operator.add]`, `next: str`, `status: str`. (FR-001, FR-008, FR-009)
- [X] T007 In `orchestrator.py`, implement the node functions: `supervisor(state)` — increments step, sets `next`/`status` per the deterministic policy (NOOP_INTENTS or step>=MAX_STEPS -> finish; else first of [local_llm, tool_execution] not in `visited`, else finish), logs a supervisor Message; `local_llm(state)` and `tool_execution(state)` — append one Message with their `source`, add their fixed TokenUsage, append name to `visited`, increment step. All deterministic, no real inference/tools. (FR-002, FR-003, FR-004, FR-005, FR-014)
- [X] T008 In `orchestrator.py`, implement routing + graph build: `route(state) -> str` returning `state["next"]`; `build_graph()` wiring entry point `supervisor`, `add_conditional_edges("supervisor", route, {"local_llm": "local_llm", "tool_execution": "tool_execution", "finish": END})`, edges `local_llm -> supervisor` and `tool_execution -> supervisor`, compiled; module-level `graph = build_graph()`. (FR-006, FR-002)
- [X] T009 In `orchestrator.py`, implement `run(payload: IntentPayload) -> OrchestrationOutcome`: build initial state (intent=payload, empty messages/visited, zero usage, step=0, next="", status=""), `graph.invoke(state, config={"recursion_limit": RECURSION_LIMIT})`, map final state -> OrchestrationOutcome (status, nodes_executed=[worker names in visited], messages, usage, steps=step). Wrap invoke in try/except so a node error or GraphRecursionError yields `OrchestrationOutcome(status="error", ...)` with partial/zeroed data rather than propagating. (FR-007, FR-010, error edge case)

**Checkpoint**: `from orchestrator import run` works; a run terminates and returns an OrchestrationOutcome.

---

## Phase 3: User Story 1 - Orchestrate an accepted intent to a terminating outcome (Priority: P1) 🎯 MVP

**Goal**: `run(payload)` starts at the Supervisor, does work, and halts with a structured outcome; never loops forever.

**Independent Test**: `run()` on an accepted intent returns an OrchestrationOutcome with a terminal status and halts.

### Tests for User Story 1

- [X] T010 [P] [US1] In `tests/test_orchestrator.py`, add tests: `run()` on a normal intent (e.g. "greet") returns an `OrchestrationOutcome` with terminal `status="completed"`; the outcome reports `nodes_executed` and a non-empty `messages` history; the call halts (asserts by returning, not hanging). (SC-001)

**Checkpoint**: MVP — the engine turns an accepted intent into a terminating, structured outcome.

---

## Phase 4: User Story 2 - Cyclic routing with accumulating state (Priority: P2)

**Goal**: Supervisor routes to workers which return to it; message history and usage accumulate one contribution per step; the run is bounded and deterministic.

**Independent Test**: A multi-hop intent yields ordered per-node messages, usage totals equal to the sum of increments, bounded steps, and identical results across repeated runs.

### Tests for User Story 2

- [X] T011 [P] [US2] In `tests/test_orchestrator.py`, add cyclic/accumulation tests for the "greet" plan: `nodes_executed == ["local_llm", "tool_execution"]`; `messages` has one ordered entry per node execution with correct `source` provenance; `usage.total_tokens == 35` (and input/output sums match) — SC-003, SC-004.
- [X] T012 [P] [US2] In `tests/test_orchestrator.py`, add termination + determinism tests: no run exceeds `MAX_STEPS` (`steps <= 8`); a NOOP intent ("ping"/"noop") finishes immediately with `status="completed"`, `nodes_executed == []`, `usage.total_tokens == 0` (immediate-finish edge); `run()` called twice on the same intent returns identical outcomes (SC-002, SC-007, immediate-finish edge).

**Checkpoint**: Cyclic, bounded, deterministic accumulation proven.

---

## Phase 5: User Story 3 - Gateway returns the orchestration outcome and usage (Priority: P2)

**Goal**: `POST /intent` triggers the engine on acceptance and returns the outcome + populated `usage`; rejections never trigger the engine.

**Independent Test**: Accepted intent → response includes `orchestration` and populated `usage`; rejected/invalid intent → engine not triggered, `usage` null.

### Implementation for User Story 3

- [X] T013 [US3] In `router.py`, update the `submit_intent` accept branch: import `run` from `orchestrator`, call `outcome = run(payload)`, and build `IntentAccepted(intent=payload.intent, orchestration=outcome, usage=outcome.usage.model_dump())` returned with HTTP 200. Leave the threshold/validation branches untouched (engine NOT called). (FR-011, FR-012, FR-013)

### Tests for User Story 3

- [X] T014 [US3] In `tests/test_gateway_endpoints.py`, UPDATE the existing accepted-path assertions (intended Feature 002 breaking change): `test_accept_confident_intent` and `test_accept_exactly_at_threshold` now expect `usage` to be a populated dict (not None) and an `orchestration` object present with `nodes_executed`; update `test_all_outcomes_share_envelope` so accepted has a populated `usage` dict while threshold_rejected and validation_rejected still have `usage is None`. (SC-005, SC-006)
- [X] T015 [P] [US3] In `tests/test_gateway_endpoints.py`, add new integration tests: accepted intent → 200 with `orchestration.nodes_executed == ["local_llm","tool_execution"]` and `usage["total_tokens"] == 35` (SC-005); below-threshold and invalid intents → engine NOT triggered (`usage` null, no orchestration run data), rejection behavior unchanged (SC-006).

**Checkpoint**: Gateway drives orchestration and returns real results + usage; rejections unchanged.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Full-suite validation and manifest confirmation.

- [X] T016 Run the full quickstart validation: `./venv/bin/python -m pytest tests/ -v` (all suites green — orchestrator, updated endpoints, unchanged 001/002 logic + payload) and the manual smoke check from `quickstart.md` (uvicorn + curl "greet" → populated usage total_tokens 35).
- [X] T017 [P] Confirm `requirements.txt` includes the LangGraph stack and no dev-only packages, and `requirements-dev.txt` is unchanged (still `-r requirements.txt` + pytest/httpx). Re-diff against `./venv/bin/pip freeze`.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: no deps — T001→T002→T003 sequential (freeze → split → verify).
- **Foundational (Phase 2)**: depends on Setup. **Blocks all user stories.** T004/T005 (models.py) are [P] with the orchestrator work but T007–T009 depend on T004 (Message/TokenUsage) and T006 (GraphState). Within `orchestrator.py`: T006→T007→T008→T009 sequential (same file).
- **US1 (Phase 3)**: depends on Foundational (needs `run()`).
- **US2 (Phase 4)**: depends on Foundational; independent of US1 (different assertions, same engine).
- **US3 (Phase 5)**: depends on Foundational (needs `run()`); T013 edits `router.py`; T014 edits existing endpoint tests (sequential with T015 on the same file, or merge).
- **Polish (Phase 6)**: depends on all prior.

### Within Files

- `orchestrator.py`: T006→T007→T008→T009 sequential.
- `models.py`: T004 and T005 both edit it — T005 after T004 (needs OrchestrationOutcome type).
- `tests/test_gateway_endpoints.py`: T014 (update) and T015 (add) touch the same file — serialize.

### Parallel Opportunities

- T004/T005 (models.py) can proceed alongside starting `orchestrator.py` scaffolding, but T007–T009 need T004's types.
- Test-writing tasks T010, T011, T012 all target `tests/test_orchestrator.py` — append-only; serialize edits or split logically.
- Polish T016/T017 are independent checks (T016 runs everything).

---

## Parallel Example: Foundational models

```bash
# models.py additions can land while orchestrator scaffolding begins:
Task: "Add Message/TokenUsage/OrchestrationOutcome to models.py"   # T004
Task: "Extend IntentAccepted with orchestration field"             # T005 (after T004)
```

---

## Implementation Strategy

### MVP First (US1)

1. Phase 1 Setup (pin langgraph).
2. Phase 2 Foundational (models + full engine).
3. Phase 3 US1 (engine terminates with an outcome + test).
4. **STOP and VALIDATE**: `run()` on an accepted intent halts with a structured outcome.

### Incremental Delivery

1. Setup + Foundational → engine exists.
2. US1 → terminating outcome (MVP).
3. US2 → cyclic/bounded/deterministic accumulation.
4. US3 → gateway integration + populated usage (updates Feature 002 tests).
5. Polish → full-suite validation.

---

## Notes

- **Three-layer termination is a hard requirement** (FR-007): Supervisor finish + `MAX_STEPS=8` + `recursion_limit=25`. Verified feasible in research R1.
- **Determinism is a hard requirement** (FR-014, SC-007): fixed usage increments (llm 10/20/30, tool 5/0/5) → full run `total_tokens == 35`.
- **Intended Feature 002 breaking change** (T014): accepted `usage` goes from null to a populated dict; rejections stay null. This is a required task, not a regression.
- Engine only runs for accepted intents (FR-013); router threshold/validation branches are untouched.
- Nodes stay stubbed — no real inference or tools (out of scope).
- Commit after each phase; each checkpoint is independently testable.
