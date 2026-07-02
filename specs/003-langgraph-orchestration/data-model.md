# Phase 1 Data Model: Orchestration Engine

New models in `models.py` and the `GraphState` in `orchestrator.py`. `IntentPayload` (001) is reused unchanged as the run input.

## `GraphState` (TypedDict, in `orchestrator.py`)

The state threaded through a run. Channel reducers in brackets.

| Channel | Type | Reducer | Purpose | Spec ref |
|---------|------|---------|---------|----------|
| `intent` | `IntentPayload` | last-value-wins | The originating intent; set at input, read by nodes. | FR-001 |
| `messages` | `list[Message]` | `operator.add` (append) | Ordered, append-only history; one entry per node step. | FR-001, FR-008 |
| `usage` | `TokenUsage` | `add_usage` (field-wise sum) | Additive token accumulator. | FR-001, FR-009 |
| `visited` | `list[str]` | `operator.add` | Node-name log driving deterministic routing. | FR-003 |
| `step` | `int` | `operator.add` | Step counter enforcing `MAX_STEPS`. | FR-007 |
| `next` | `str` | last-value-wins | Supervisor's routing decision, read by the conditional edge. | FR-003 |
| `status` | `str` | last-value-wins | Terminal status set when the Supervisor finishes. | FR-010 |

**Reducer `add_usage(left, right)`**: returns a `TokenUsage` whose fields are the element-wise sum (`left is None` → `right`). Guarantees FR-009 (totals = sum of contributions).

## `Message` (Pydantic, in `models.py`)

An ordered history entry appended as a node executes.

| Field | Type | Notes |
|-------|------|-------|
| `source` | `str` | Which node produced it (`"supervisor"` / `"local_llm"` / `"tool_execution"`). Provenance (FR-008). |
| `content` | `str` | Deterministic stub content. |
| `step` | `int` | Step index within the run. |

## `TokenUsage` (Pydantic, in `models.py`)

The additive usage accumulator; `model_dump()`s straight into the response envelope `usage` field.

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `input_tokens` | `int` | `0` | Summed across steps. |
| `output_tokens` | `int` | `0` | Summed across steps. |
| `total_tokens` | `int` | `0` | Summed across steps. |

Fixed per-node increments (R4): `local_llm` → (10, 20, 30); `tool_execution` → (5, 0, 5).

## `OrchestrationOutcome` (Pydantic, in `models.py`)

The structured result returned on termination (FR-010).

| Field | Type | Notes |
|-------|------|-------|
| `status` | `str` | Terminal status: `"completed"` / `"halted_step_bound"` / `"error"`. |
| `nodes_executed` | `list[str]` | Worker nodes that ran (from `visited`). |
| `messages` | `list[Message]` | The full ordered history. |
| `usage` | `TokenUsage` | Accumulated totals. |
| `steps` | `int` | Total node steps taken. |

## `IntentAccepted` (extended, in `models.py`)

Feature 002's success envelope gains the orchestration result. **Existing fields unchanged**; new field added:

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `orchestration` | `OrchestrationOutcome \| None` | `None` | The run result for an accepted intent (FR-011). |
| `usage` | `dict \| None` | (inherited) | **Now populated** for accepted intents from `orchestration.usage` (FR-012). |

## Graph wiring (in `orchestrator.py`)

```
entry point: supervisor
supervisor --(conditional edges on state["next"])--> { local_llm | tool_execution | END }
local_llm      --> supervisor        (cyclic)
tool_execution --> supervisor        (cyclic)
```

- Compiled once: `graph = build_graph()` at module load.
- Invoked with `config={"recursion_limit": RECURSION_LIMIT}`.
- Constants: `MAX_STEPS = 8`, `RECURSION_LIMIT = 25`, `NOOP_INTENTS = {"ping", "noop"}`.

## Decision / termination logic

- **Supervisor** (`supervisor(state)`): increments `step`; if `intent.intent in NOOP_INTENTS` or `step >= MAX_STEPS` → set `next="finish"`, `status` accordingly; else pick the first of `[local_llm, tool_execution]` not in `visited`, else `next="finish"`. Logs a supervisor `Message`.
- **Routing edge** (`route(state) -> str`): returns `state["next"]`; mapping `{"local_llm": "local_llm", "tool_execution": "tool_execution", "finish": END}`.
- **Worker nodes**: append one `Message`, add fixed `TokenUsage`, append name to `visited`, `step += 1`, edge back to supervisor.

## Relationships

- `IntentPayload` (001) → the `intent` channel (unchanged, FR reuse).
- `TokenUsage` → embedded in `OrchestrationOutcome` and mirrored into the envelope `usage` (002 reserved field, FR-012).
- `OrchestrationOutcome` → embedded in `IntentAccepted.orchestration` (FR-011).

## State transitions

Per run: `supervisor → (local_llm | tool_execution)* → supervisor → END`, bounded by `MAX_STEPS`/`recursion_limit`. No cross-request state; each request starts fresh (FR-014 determinism).
