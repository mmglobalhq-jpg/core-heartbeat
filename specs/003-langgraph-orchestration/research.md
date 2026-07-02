# Phase 0 Research: Orchestration Engine

All Technical Context unknowns resolved. No open NEEDS CLARIFICATION.

## R1 — CRITICAL: LangGraph on Python 3.14 (dependency viability)

- **Decision**: Adopt **langgraph 1.2.7**. Verified in the venv on Python 3.14.4: it installs cleanly, `from langgraph.graph import StateGraph, END` and `from langgraph.errors import GraphRecursionError` import, and a minimal cyclic graph (Supervisor→worker→Supervisor with conditional edges) compiles, runs, and terminates. An infinite `a→a` graph raises `GraphRecursionError` at `recursion_limit`, confirming the hard termination guarantee.
- **Transitive deps pulled** (all installed 3.14 wheels): langchain-core 1.4.8, langgraph-checkpoint 4.1.1, langgraph-prebuilt 1.1.0, langgraph-sdk 0.4.2, langsmith 0.9.5, langchain-protocol 0.0.18, orjson 3.11.9, ormsgpack 1.12.2, tenacity 9.1.4, jsonpatch 1.33, jsonpointer 3.1.1, pyyaml 6.0.3, uuid-utils 0.16.2, xxhash 3.8.0, zstandard 0.25.0, requests 2.34.2, urllib3 2.7.0, charset-normalizer 3.4.7, requests-toolbelt 1.0.0, sniffio 1.3.1, distro 1.9.0, websockets 15.0.1.
- **Rationale**: This was the feature's only blocking risk. Resolving it up front (rather than discovering an install failure mid-implementation) is why it is R1. No LLM provider packages are needed — nodes are stubbed.
- **Alternatives considered**: Hand-rolled state machine (no dependency) — rejected: the user chose LangGraph as the standard engine and it works on 3.14. Older langgraph pins — unnecessary; latest installs fine.
- **Risk note**: langgraph pulls a sizeable transitive tree (requests/urllib3/websockets/langsmith). Acceptable for the chosen engine; pinned in `requirements.txt`.

## R2 — GraphState representation & reducers

- **Decision**: `GraphState` is a `TypedDict` with `Annotated` channel reducers:
  - `intent: IntentPayload` — no reducer (last-value-wins); set once at input, read by nodes.
  - `messages: Annotated[list[Message], operator.add]` — append-only; nodes return `{"messages": [Message(...)]}` and LangGraph concatenates (FR-008).
  - `usage: Annotated[TokenUsage, add_usage]` — additive; a custom `add_usage(left, right)` reducer field-wise sums token counts (FR-009).
  - `visited: Annotated[list[str], operator.add]` — node-name log the Supervisor reads to plan the next route.
  - `step: Annotated[int, operator.add]` — increments per node; drives the `MAX_STEPS` bound.
  - `next: str` — the Supervisor's routing decision (last-value-wins), consumed by the conditional edge.
  - `status: str` — terminal status set when the Supervisor finishes (last-value-wins).
- **Rationale**: TypedDict + reducers is LangGraph's idiomatic state model and makes accumulation semantics (append vs sum) explicit and testable. Keeping `intent` as a plain channel reuses the frozen `IntentPayload` unchanged. Deriving routing from `visited`/`step` keeps the Supervisor deterministic.
- **Alternatives considered**: Pydantic `BaseModel` state — supported by LangGraph but heavier and awkward with per-field reducers; TypedDict is the common choice. Storing usage as a bare dict with a dict-merge reducer — a typed `TokenUsage` model is clearer and `model_dump()`s straight into the response `usage` field.

## R3 — Deterministic Supervisor routing & termination

- **Decision**: The Supervisor node computes a deterministic decision and writes it to `next`; a conditional-edge function returns `state["next"]`, mapped to `local_llm` / `tool_execution` / `END`. Policy:
  - If `intent.intent` is in `NOOP_INTENTS` (e.g. `{"ping", "noop"}`) → finish immediately (edge case "immediate finish"), `status="completed"`.
  - Else run the deterministic plan **local_llm then tool_execution** (drive off `visited`): route to `local_llm` if not yet visited, else `tool_execution` if not yet visited, else finish (`status="completed"`).
  - Safety bound: if `step >= MAX_STEPS`, finish with `status="halted_step_bound"` regardless (FR-007, SC-002).
- **Termination (three layers)**: (1) the plan is finite; (2) `MAX_STEPS` graceful bound; (3) `recursion_limit` on `invoke` as the hard catch (raises `GraphRecursionError`). `MAX_STEPS` is set comfortably below `recursion_limit` so normal runs halt gracefully, never via the exception.
- **Rationale**: Deterministic routing makes runs reproducible (FR-014, SC-007) and every scenario testable. Reading the intent (`NOOP_INTENTS`) honors "Supervisor reads the intent and decides." Three layers satisfy "cycles cannot run forever" defensively.
- **Alternatives considered**: Model/LLM-driven routing — out of scope (stubbed feature). Relying on `recursion_limit` alone — would terminate via an exception rather than a clean status; the explicit `MAX_STEPS` gives a graceful terminal status.
- **Constants**: `MAX_STEPS = 8`, `RECURSION_LIMIT = 25` (both configurable later; values are planning defaults ample for the 2-hop plan which uses ~5 supersteps).

## R4 — Stubbed node behavior & fixed usage increments

- **Decision**: `local_llm` and `tool_execution` are pure deterministic stubs. Each appends one `Message` (with its `source`), records itself in `visited`, increments `step`, and adds a **fixed** `TokenUsage`:
  - `local_llm`: `TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30)`, content `"[stub] local inference result"`.
  - `tool_execution`: `TokenUsage(input_tokens=5, output_tokens=0, total_tokens=5)`, content `"[stub] tool executed"`.
- **Rationale**: Fixed increments make accumulation exactly assertable (a full 2-hop run → `total_tokens = 35`, `messages` length known) and the whole run reproducible (SC-003, SC-004, SC-007). No real inference/tools (out of scope, FR-014).
- **Alternatives considered**: Randomized/timestamped stub output — would break determinism/reproducibility. Rejected.

## R5 — OrchestrationOutcome & gateway response integration

- **Decision**: `run(payload) -> OrchestrationOutcome` invokes the compiled graph with an initial state (`intent=payload`, empty `messages`/`visited`, zero `usage`, `step=0`) and `config={"recursion_limit": RECURSION_LIMIT}`, then maps the final state to `OrchestrationOutcome(status, nodes_executed=visited, messages, usage, steps)`. In `router.py`, the accept branch calls `run(payload)` and returns `IntentAccepted(intent=..., orchestration=outcome, usage=outcome.usage.model_dump())` — the envelope `usage` **mirrors** the run's accumulated usage (FR-011, FR-012, SC-005).
- **Rationale**: A thin `run()` keeps `router.py` unchanged in shape (still returns `IntentAccepted`) while satisfying FR-011/012. Mirroring `usage` into the standardized envelope field connects to feature 002's reserved slot; the detailed record also lives inside `orchestration`.
- **CROSS-FEATURE IMPACT**: Feature 002's endpoint tests assert `accepted.usage is None`. That is now **intentionally** false for accepted intents (FR-012). `tests/test_gateway_endpoints.py` MUST be updated: accepted → `usage` is a populated dict + `orchestration` present; threshold/validation rejections still have `usage is None` (engine not triggered, FR-013). This is a required task, not a regression.
- **Error handling**: `run()` wraps `graph.invoke` so a node exception or `GraphRecursionError` yields an `OrchestrationOutcome(status="error", ...)` with whatever partial usage/messages are available, rather than propagating. The gateway returns it in the accepted envelope (HTTP 200) with `orchestration.status="error"` (edge case "node raises"). Stubs don't raise, so this is defensive.
- **Alternatives considered**: New top-level outcome enum value for orchestration errors — unnecessary; the acceptance succeeded and the error is surfaced structurally inside `orchestration`. A 5xx for orchestration errors — deferred; MVP surfaces within the envelope per the spec's edge case.

## R6 — requirements.txt regeneration strategy

- **Decision**: Regenerate `requirements.txt` from `pip freeze` (now includes the LangGraph stack) **minus the dev-only packages** already pinned in `requirements-dev.txt` (pytest, iniconfig, pluggy, pygments, and the httpx test-client stack: httpx, httpcore, certifi). Verify the split by asserting every `requirements.txt` pin imports and no dev-only package leaks in.
- **Rationale**: LangGraph is a runtime dependency (the engine runs in-process serving requests), so it belongs in `requirements.txt`; the test client and pytest remain dev-only. `packaging` is a shared transitive dep and stays wherever `pip freeze` lists it (runtime), since langchain-core depends on it.
- **Alternatives considered**: Adding only `langgraph` unpinned — loses reproducibility; the whole resolved tree is pinned instead.

## Cross-cutting: what stays OUT

Per spec Assumptions/FR-014 — no real local-model inference, no real tool integrations, no cross-request persistence/checkpointing, no streaming, no human-in-the-loop interrupts, no async/background execution (synchronous within the request), no model-driven routing.
