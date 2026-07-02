# Contract: API-Driven Supervisor Node

## Model call contract (`request_routing_decision`)

```
request_routing_decision(state: GraphState, client)
    -> tuple[RoutingDecision | None, RoutingFailure | None, TokenUsage]
```

- **Never raises** to its caller — all exceptions are mapped to a `RoutingFailure`.
- Exactly one of `(decision, failure)` is non-None.
- Returns a `TokenUsage` (zeros when the model reports none or on failure).
- Makes **one** model call per invocation, bounded by `REQUEST_TIMEOUT_MS`.

### Model request
- `model = "gemini-2.5-flash"`.
- `config`: `response_mime_type="application/json"`, `response_schema=RoutingDecision`, `http_options=HttpOptions(timeout=REQUEST_TIMEOUT_MS)`.
- `contents`: a routing prompt derived from `state["intent"]` and `state["messages"]`.

### Output validation
- Parse `response.parsed` (or `json.loads(response.text)`), then `RoutingDecision.model_validate(...)`.
- `next_node` MUST be one of `local_llm | tool_execution | finish`. Anything else → `RoutingFailure(category="invalid_output")`.

### Failure mapping
| Exception / condition | `RoutingFailure.category` |
|-----------------------|---------------------------|
| `client is None` (missing key) | `missing_credential` |
| `genai.errors.ClientError` 401/403 | `auth` |
| `httpx.TimeoutException` | `timeout` |
| `httpx.TransportError` / `ConnectError` / `ServerError` / other `APIError` | `network` |
| JSON error / `ValidationError` / out-of-vocab | `invalid_output` |

## Supervisor node contract

- **On step bound** (`step >= MAX_STEPS`): finish, `status="halted_step_bound"`, no model call.
- **On success**: `next = decision.next_node`; if `finish`, `status="completed"`; usage accumulated.
- **On any failure**: `next="finish"`, `status="degraded"`, a `Message` recording `routing failure: <category>` appended, usage accumulated (may be zero). The run terminates.
- **Guarantee**: the Supervisor never raises, never hangs (bounded timeout), and always yields a routable `next`. Three-layer termination preserved (finish decision + `MAX_STEPS` + `recursion_limit`).

## Credential handling

- Read `GEMINI_API_KEY` from the environment in `get_client()`.
- Missing/blank key → `get_client()` returns `None` → `missing_credential` degraded finish (service stays up).
- The key is passed to the SDK and MUST NOT be logged or included in any recorded message/detail.

## Test contract (no network — FR-012, SC-007)

A fake client is injected via `monkeypatch orchestrator.get_client`. It must support the shapes below without any real call:

| Fake behavior | Expected result |
|---------------|-----------------|
| returns valid `{"next_node": "local_llm"}` | routes to local_llm |
| returns valid `{"next_node": "tool_execution"}` | routes to tool_execution |
| returns valid `{"next_node": "finish"}` | run finishes, `status="completed"` |
| returns `{"next_node": "banana"}` / non-JSON | `invalid_output` → degraded finish |
| raises `genai.errors.ClientError` 401 | `auth` → degraded finish |
| raises `httpx.TimeoutException` | `timeout` → degraded finish |
| raises `httpx.ConnectError` | `network` → degraded finish |
| `get_client()` returns None (no key) | `missing_credential` → degraded finish |
| reports `usage_metadata` tokens | added to run usage |
| scripted `[local_llm, tool_execution, finish]` | greet-plan trace reproduced; `nodes_executed == [local_llm, tool_execution]` |

## Downstream (unchanged)

- Conditional edge mapping, worker nodes, `run()` mapping to `OrchestrationOutcome`, and the gateway integration are unchanged except that a degraded run surfaces `status="degraded"` and (usually) `nodes_executed == []`. Accepted intents still return HTTP 200 with the orchestration outcome + usage (FR of feature 003 preserved).
