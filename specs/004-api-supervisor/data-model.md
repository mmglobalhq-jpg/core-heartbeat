# Phase 1 Data Model: API-Driven Supervisor Node

New models in `models.py`; `GraphState` and node logic in `orchestrator.py` are modified. `OrchestrationOutcome` gains a status value.

## `RoutingDecision` (Pydantic, in `models.py`) — the strict output schema

Passed to the model as `response_schema`; also used to re-validate the response.

| Field | Type | Notes |
|-------|------|-------|
| `next_node` | `Literal["local_llm", "tool_execution", "finish"]` | The only accepted decisions (FR-002, SC-001). Values match the graph's conditional-edge keys exactly. |

## `RoutingFailure` (Pydantic, in `models.py`) — categorized degradation

| Field | Type | Notes |
|-------|------|-------|
| `category` | `Literal["missing_credential", "auth", "timeout", "network", "invalid_output"]` | The failure taxonomy (research R4). |
| `detail` | `str` | Short human-readable context (never contains the credential). |

## Failure taxonomy → trigger (research R4)

| Category | Trigger |
|----------|---------|
| `missing_credential` | `GEMINI_API_KEY` unset/blank (`get_client()` → None) |
| `auth` | `genai.errors.ClientError` 401/403 |
| `timeout` | `httpx.TimeoutException` (bounded by `REQUEST_TIMEOUT_MS`) |
| `network` | `httpx.TransportError`/`ConnectError`, `genai.errors.ServerError`, other non-auth `APIError` |
| `invalid_output` | JSON decode error, Pydantic `ValidationError`, out-of-vocab value |

## `OrchestrationOutcome.status` (extended)

Existing values `completed` / `halted_step_bound` / `error`; **add** `degraded` — a run that terminated safely after a Supervisor routing failure (FR-008). Distinguishes a failed-routing finish from a normal completion.

## `GraphState` (modified, `orchestrator.py`)

Unchanged channels from feature 003 (`intent`, `messages` [append], `usage` [add_usage], `visited` [append], `step` [add], `next`, `status`). No new channels are strictly required — the failure is recorded via a `messages` entry and the terminal `status="degraded"`. (`visited` still tracks worker executions for `nodes_executed`.)

## Constants (`orchestrator.py`)

| Name | Value | Purpose |
|------|-------|---------|
| `MODEL_NAME` | `"gemini-2.5-flash"` | Target model. |
| `GEMINI_API_KEY_ENV` | `"GEMINI_API_KEY"` | Credential env var. |
| `REQUEST_TIMEOUT_MS` | a few seconds (e.g. `10_000`) | Bounds each model call (FR-006/SC-004). |
| `MAX_STEPS` / `RECURSION_LIMIT` | `8` / `25` (unchanged) | Termination layers 2 & 3. |

## Functions (`orchestrator.py`)

- **`get_client() -> genai.Client | None`**: reads `GEMINI_API_KEY`; `None` if missing/blank; else a memoized `genai.Client`.
- **`request_routing_decision(state, client) -> tuple[RoutingDecision | None, RoutingFailure | None, TokenUsage]`**: builds context (intent + message history), calls `client.models.generate_content(model=MODEL_NAME, contents=..., config=GenerateContentConfig(response_mime_type="application/json", response_schema=RoutingDecision, http_options=HttpOptions(timeout=REQUEST_TIMEOUT_MS)))`, validates → `RoutingDecision`; maps exceptions/invalid output → `RoutingFailure`; extracts `usage_metadata` → `TokenUsage` (zeros when absent). Never raises to the caller.
- **`supervisor(state)`** (rewritten): step-bound guard → model call → route or degrade (research R6).

## Supervisor decision flow

```
if step >= MAX_STEPS:            -> next="finish", status="halted_step_bound"   (no model call)
client = get_client()
if client is None:               -> next="finish", status="degraded", record missing_credential
decision, failure, usage = request_routing_decision(state, client)
if failure is not None:          -> next="finish", status="degraded", record failure.category, add usage
else:                            -> next=decision.next_node, add usage
                                    (status="completed" iff next=="finish")
```

Routing edge mapping unchanged: `{"local_llm": local_llm, "tool_execution": tool_execution, "finish": END}`.

## Relationships

- `RoutingDecision` → drives the `next` channel → the conditional edge (FR-010).
- `RoutingFailure` → recorded in `messages` + terminal `status="degraded"` (FR-008); resolves to `finish` (FR-005).
- Model `TokenUsage` → `usage` channel via `add_usage` → run totals → gateway envelope (FR-009).
- `local_llm`/`tool_execution` → unchanged stubs (FR-011).

## State transitions

Per Supervisor visit: model call → route to a worker (which returns to Supervisor) or finish. Guaranteed to terminate by: a `finish` decision, a failure→finish, `MAX_STEPS`, and `recursion_limit`. Determinism in tests via injected fake client.
