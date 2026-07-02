# Phase 1 Data Model: Local Ollama Worker Node

All additions are in `models.py`; the graph state (`GraphState` in
`orchestrator.py`) is unchanged — the local worker reuses the existing `messages`,
`usage`, `visited`, and `step` channels and their reducers.

## New model

### `WorkerFailure`

A categorized, credential-free record of why a local inference step could not
produce output (feature 005; parallel to feature 004's `RoutingFailure`).

| Field | Type | Rules |
|-------|------|-------|
| `category` | `Literal["unreachable", "timeout", "invalid_output"]` | one of the three; see failure mapping in research R5 |
| `detail` | `str` | short, network-safe diagnostic (`f"{type(exc).__name__}: {exc}"[:200]` or `"HTTP <code>"`); no secrets (Ollama is keyless) |

Recorded into the run's message history as
`Message(source="local_llm", content=f"local inference failure: {category}", step=…)`.
Not returned as a standalone field on `OrchestrationOutcome`; observability comes
through the message history (which *is* part of the outcome).

## Reused / unchanged models

- **`TokenUsage`** — unchanged. The worker maps Ollama's `prompt_eval_count` →
  `input_tokens`, `eval_count` → `output_tokens`, sum → `total_tokens`, and
  returns it on the node's `usage` channel; the existing `add_usage` reducer sums
  it field-wise into the run total (FR-005). Absent counts → `TokenUsage()`
  (zeros) (FR-006).
- **`Message`** — unchanged; the worker appends real generated text (or a failure
  line) with `source="local_llm"`.
- **`OrchestrationOutcome`** — unchanged shape. `nodes_executed` still includes
  `local_llm` when it ran (success *or* failure), because it appends to `visited`
  either way. `status` continues to be set by the Supervisor / `run()`.
- **`GraphState`** — unchanged. No new channels; the worker writes only
  `messages`, `usage`, `visited`, `step`.

## Node output contract (updated `local_llm`)

`local_llm` transitions from a fixed stub to an async node. Its returned state
update:

**On success**
```
{
  "messages": [Message(source="local_llm", content=<generated text>, step=<step>)],
  "usage":    TokenUsage(input=prompt_eval_count, output=eval_count, total=sum),
  "visited":  ["local_llm"],
  "step":     1,
}
```

**On failure (WorkerFailure)**
```
{
  "messages": [Message(source="local_llm", content="local inference failure: <category>", step=<step>)],
  "usage":    TokenUsage(),          # zeros
  "visited":  ["local_llm"],         # it still executed
  "step":     1,
}
```

In both cases the node returns control to the Supervisor via the existing
`local_llm → supervisor` edge; it never sets `next`/`status` (research R6).

## Configuration (environment)

| Var | Default | Meaning |
|-----|---------|---------|
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | local generate endpoint (FR-010) |
| `OLLAMA_MODEL` | `qwen2.5:7b` | target model (FR-010) |
| `OLLAMA_TIMEOUT_MS` | `120000` | per-call time bound (FR-007, SC-005) |

Read at node-invoke time (not captured at graph-build time) so tests can override
via monkeypatch/env, matching feature 004's dynamic `get_client()` seam.
