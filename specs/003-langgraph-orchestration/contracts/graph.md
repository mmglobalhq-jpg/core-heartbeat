# Contract: Orchestration Engine & Gateway Integration

## Engine interface (`orchestrator.py`)

```
run(payload: IntentPayload) -> OrchestrationOutcome
```

- **Precondition**: `payload` is a fully-valid, accepted `IntentPayload` (the gateway calls `run` only after acceptance).
- **Postcondition**: returns an `OrchestrationOutcome` with a terminal `status`; the call always returns (never loops forever, never raises for stub runs — errors are captured into `status="error"`).
- **Determinism**: `run(p)` called twice with an equal `p` returns equal `OrchestrationOutcome` (FR-014, SC-007).

Module-level: `graph = build_graph()` — the compiled `StateGraph`, built once.

## Graph contract

- **Entry point**: `supervisor` (FR-002).
- **Nodes**: `supervisor`, `local_llm` (stub), `tool_execution` (stub).
- **Edges**:
  - `supervisor` → conditional on `state["next"]` → `local_llm` | `tool_execution` | `END`.
  - `local_llm` → `supervisor`; `tool_execution` → `supervisor` (cyclic, FR-006).
- **Invoke config**: `{"recursion_limit": RECURSION_LIMIT}` (=25).
- **Termination guarantees** (FR-007, SC-001/002):
  - Supervisor finishes when the deterministic plan is done (`status="completed"`).
  - Supervisor finishes at `step >= MAX_STEPS` (=8) with `status="halted_step_bound"`.
  - `recursion_limit` raises `GraphRecursionError` if all else failed — caught by `run()` → `status="error"`.

## State channel contract

| Channel | Accumulation | Invariant |
|---------|--------------|-----------|
| `messages` | append (`operator.add`) | one entry per node execution, in order, with `source` provenance (SC-003) |
| `usage` | sum (`add_usage`) | final totals == Σ per-step increments (SC-004) |
| `visited` | append | records worker nodes run, drives routing |
| `step` | sum | bounded by `MAX_STEPS` |

Fixed usage increments (deterministic): `local_llm`=(10,20,30), `tool_execution`=(5,0,5). A full 2-hop run → `total_tokens = 35`.

## Deterministic run traces (reference)

| Intent identity | Trace | `nodes_executed` | `total_tokens` | `status` |
|-----------------|-------|------------------|----------------|----------|
| e.g. `"greet"` | supervisor→local_llm→supervisor→tool_execution→supervisor→END | `[local_llm, tool_execution]` | 35 | completed |
| `"ping"` / `"noop"` | supervisor→END | `[]` | 0 | completed |

## Gateway integration contract (`router.py`, `POST /intent`)

- **On acceptance** (valid + confidence ≥ threshold): call `run(payload)`, return `IntentAccepted` with:
  - `outcome = "accepted"`, `accepted = true`, `intent`, `detail` (unchanged fields).
  - `orchestration` = the `OrchestrationOutcome` (FR-011).
  - `usage` = `orchestration.usage` as a dict — **populated** (FR-012, SC-005), previously `null`.
  - HTTP 200.
- **On rejection** (threshold/validation): engine is NOT called; `usage` stays `null`; responses unchanged (FR-013, SC-006).

### Breaking change to feature 002 responses (intended)

`tests/test_gateway_endpoints.py` assertions that `accepted.usage is None` MUST be updated: accepted responses now carry a populated `usage` dict and an `orchestration` object. Rejection responses are unchanged (`usage is None`).
