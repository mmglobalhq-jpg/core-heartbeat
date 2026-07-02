# Implementation Plan: Orchestration Engine

**Branch**: `003-langgraph-orchestration` | **Date**: 2026-07-01 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/003-langgraph-orchestration/spec.md`

## Summary

Add a LangGraph-based orchestration engine in a new `orchestrator.py`. Define a `GraphState` (TypedDict with reducers) carrying the `IntentPayload`, an append-only `messages` history, and an additive `usage` token tracker. Build a cyclic `StateGraph`: entry point = **Supervisor**, which uses conditional edges to route to a stubbed **local_llm** node, a stubbed **tool_execution** node, or finish (`END`); both worker nodes edge back to the Supervisor. Termination is guaranteed two ways — a deterministic Supervisor finish decision plus an explicit `MAX_STEPS` graceful bound, backed by LangGraph's `recursion_limit` as a hard catch. The compiled graph is built once at module load. `router.py`'s `POST /intent` invokes it after acceptance and maps the final state into an `OrchestrationOutcome`, populating the response envelope's `usage` field (reserved-but-null since feature 002). Nodes are deterministic stubs — no real inference or tools.

## Technical Context

**Language/Version**: Python 3.14.4 (venv)

**Primary Dependencies**: **NEW** langgraph 1.2.7 (+ langchain-core 1.4.8, langgraph-checkpoint 4.1.1, langgraph-prebuilt, langgraph-sdk, langsmith, orjson, and transitive deps) — verified installing and running on 3.14 (research R1). Existing: FastAPI 0.139.0, Pydantic 2.13.4. Reuses `IntentPayload` (001) and the gateway (002).

**Storage**: N/A — orchestration state lives only for the duration of a run; no cross-request persistence/checkpointing in this MVP.

**Testing**: pytest + FastAPI TestClient (feature 002). New `tests/test_orchestrator.py` for the engine; `tests/test_gateway_endpoints.py` updated for populated `usage`.

**Target Platform**: Linux server (WSL2 dev); ASGI app via uvicorn.

**Project Type**: Single project — small web service; orchestration added as one new module.

**Performance Goals**: Not a stated objective. Runs are short bounded loops of deterministic stubs (sub-millisecond).

**Constraints**: Guaranteed termination (FR-007) via Supervisor finish + `MAX_STEPS` + `recursion_limit`; append-only ordered `messages` (FR-008); additive lossless `usage` (FR-009); deterministic/reproducible (FR-014); engine only triggered for accepted intents (FR-013); accepted response `usage` now populated (FR-012).

**Scale/Scope**: One new module (`orchestrator.py`), ~4 new response models in `models.py`, a small edit to `router.py`'s accept branch, plus tests. LangGraph stack added to `requirements.txt`.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Constitution is the unpopulated template — no ratified principles, no enforceable gates. **Gate status: PASS (vacuously).**

Applied defaults (consistent with 001/002): simplicity/YAGNI (stubbed nodes, no real inference/tools/persistence — all out of scope), test-first (acceptance scenarios → engine + endpoint tests), no scope creep (three nodes, deterministic routing).

*Post-Phase 1 re-check*: Design adds one module, response models, and one router edit; the only new **runtime** dependency is the LangGraph stack (justified — it is the chosen engine). **PASS.**

## Project Structure

### Documentation (this feature)

```text
specs/003-langgraph-orchestration/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   └── graph.md          # Phase 1 — state schema, node/edge wiring, routing/termination contract
├── checklists/
│   └── requirements.md  # from /speckit-specify
└── tasks.md             # Phase 2 (/speckit-tasks — NOT created here)
```

### Source Code (repository root)

```text
orchestrator.py  # NEW: GraphState (TypedDict + reducers), add_usage reducer, node
                 #   functions (supervisor, local_llm, tool_execution), routing fn,
                 #   build_graph() + module-level compiled `graph`, run(payload) -> OrchestrationOutcome.
                 #   Constants: MAX_STEPS, RECURSION_LIMIT, per-node usage increments, NOOP_INTENTS.
models.py        # + Message, TokenUsage, OrchestrationOutcome; extend IntentAccepted
                 #   with an `orchestration` field; usage populated from the run.
router.py        # POST /intent accept branch calls orchestrator.run(payload) and
                 #   builds IntentAccepted with orchestration + populated usage.
main.py          # unchanged (factory still valid).

tests/
├── test_orchestrator.py       # NEW — engine: termination, cycle, history/usage accumulation, determinism
├── test_gateway_endpoints.py  # UPDATED — accepted response now carries orchestration + populated usage
├── test_gateway_logic.py      # unchanged
└── test_intent_payload.py     # unchanged
```

**Structure Decision**: Orchestration goes in a **new `orchestrator.py`** (keeps the graph wiring isolated from routing and schemas, flat layout preserved). Response/data models stay in `models.py`. The graph is compiled once at module import (`graph = build_graph()`) and reused per request; `run(payload)` wraps `graph.invoke(...)` and maps the final `GraphState` to an `OrchestrationOutcome`, so `router.py` stays thin and the mapping is unit-testable.

## Complexity Tracking

> No constitution violations to justify — section intentionally empty. The LangGraph dependency is the feature's explicit premise, not unjustified complexity.
