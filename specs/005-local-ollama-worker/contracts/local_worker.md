# Contract: Local Ollama Worker Node

## Model-call helper (`generate_local`)

```
async def generate_local(state: GraphState, client: httpx.AsyncClient)
    -> tuple[str | None, WorkerFailure | None, TokenUsage]
```

- **Never raises** to its caller — every exception maps to a `WorkerFailure`.
- Exactly one of `(text, failure)` is non-None.
- Returns a `TokenUsage` (zeros when the response reports none or on failure).
- Makes **one** POST per invocation, bounded by `OLLAMA_TIMEOUT_MS`.

### Request
- `POST` to `OLLAMA_URL` (default `http://localhost:11434/api/generate`).
- JSON body: `{"model": OLLAMA_MODEL, "prompt": <built>, "stream": false}`.
- `prompt` derived from `state["intent"]` + `state["messages"]` (same construction
  style as the Supervisor's routing prompt).
- Timeout: `OLLAMA_TIMEOUT_MS` applied to the request.

### Response handling
- Require 2xx; otherwise → `WorkerFailure(category="invalid_output", detail="HTTP <code>")`.
- Parse JSON; require a `response` field (the generated text).
- `text = body["response"]` (may be an empty string — that is a valid success).
- Usage: `TokenUsage(input=body.get("prompt_eval_count", 0) or 0,`
  `output=body.get("eval_count", 0) or 0, total=input+output)`.

### Failure mapping
| Exception / condition | `WorkerFailure.category` |
|-----------------------|--------------------------|
| `httpx.TimeoutException` (any subclass) | `timeout` |
| `httpx.ConnectError` / other `httpx.TransportError` | `unreachable` |
| non-2xx HTTP status | `invalid_output` |
| body not JSON / missing `response` / decode error | `invalid_output` |
| any other exception (safety net) | `unreachable` |

## Node contract (`local_llm`, now async)

- **On success**: append a `Message(source="local_llm", content=<text>)`, add the
  extracted `TokenUsage`, append `"local_llm"` to `visited`, `step += 1`.
- **On failure**: append `Message(source="local_llm", content="local inference`
  `failure: <category>")`, add `TokenUsage()` (zeros), append `"local_llm"` to
  `visited`, `step += 1`.
- **Never** sets `next`/`status` — routing stays with the Supervisor (research R6).
- Returns control to the Supervisor via the existing `local_llm → supervisor` edge.

## Client factory & seam

```
def build_ollama_client() -> httpx.AsyncClient
```
- Constructs an `httpx.AsyncClient` (reads `OLLAMA_URL`/`OLLAMA_TIMEOUT_MS` at call
  time). The node calls it at invoke time, so tests monkeypatch
  `orchestrator.build_ollama_client` to return a client wired with
  `httpx.MockTransport` — **no network, no daemon** (FR-011, SC-004).

## Invocation (`run`, now async)

```
async def run(payload: IntentPayload) -> OrchestrationOutcome
```
- Uses `await graph.ainvoke(initial, config={"recursion_limit": RECURSION_LIMIT})`.
- Sync `supervisor`/`tool_execution` nodes run unchanged under `ainvoke`.
- `router.submit_intent` becomes `async def` and `await`s `run_orchestration`.
- Existing outcome mapping unchanged.

## Termination guarantee (unchanged, FR-012)

Three layers preserved: Supervisor finish/degrade decision + `MAX_STEPS` guard
(before any model call) + `recursion_limit`. A failing/looping local worker cannot
run unbounded — repeated routes to it are capped by the step bound and recursion
limit.

## Test contract (no network — FR-011, SC-004)

A `MockTransport` handler is injected via `monkeypatch orchestrator.build_ollama_client`.
It must support the shapes below without any real call:

| Fake behavior | Expected result |
|---------------|-----------------|
| 200 `{"response":"hi","prompt_eval_count":26,"eval_count":290}` | text `"hi"`, usage `26/290/316` |
| 200 `{"response":""}` (no counts) | text `""`, usage `0/0/0`, no error |
| 200 body missing `response` | `WorkerFailure(invalid_output)` |
| 200 non-JSON body | `WorkerFailure(invalid_output)` |
| 404 / 500 status | `WorkerFailure(invalid_output)`, detail `"HTTP 404"` |
| handler raises `httpx.ConnectError` | `WorkerFailure(unreachable)` |
| handler raises `httpx.ReadTimeout` | `WorkerFailure(timeout)` |
| full run: supervisor→local_llm (mock 200)→…→finish | `nodes_executed` includes `local_llm`; run usage includes the mocked counts; terminates |
| full run: supervisor→local_llm (mock raises)→…→finish | failure `Message` recorded; run still terminates; `local_llm` in `nodes_executed` |
