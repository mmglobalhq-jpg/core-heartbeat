# Phase 0 Research: Local Ollama Worker Node

## R1 — Async node execution under the compiled graph

**Decision**: Make `local_llm` an `async def` node and drive the graph with
`graph.ainvoke(...)` from an `async def run()`. The `supervisor` node stays a
plain sync function; `tool_execution` stays a sync stub.

**Rationale**: The user requires an *asynchronous* HTTP call. LangGraph executes
a mix of sync and async node callables under `ainvoke` — sync nodes run in the
default executor thread, async nodes are awaited on the loop. Verified empirically
against the actual compiled-graph shape (sync supervisor ↔ async worker, cyclic,
`recursion_limit=25`): `ainvoke` completed correctly and termination held. The
graph is still compiled once at import; only the invocation entry point changes
from `.invoke` to `.ainvoke`.

**Alternatives considered**:
- *Sync `httpx.Client` in a sync node* — simplest, no ripple, but not the
  asynchronous call requested; rejected on the explicit requirement.
- *Sync node calling `asyncio.run(_call())` internally* — breaks with
  "event loop is already running" once the gateway endpoint is async; fragile,
  rejected.

## R2 — Ripple of an async `run()` into the gateway and tests

**Decision**: `run()` becomes `async`; `router.submit_intent` becomes
`async def` and `await`s it. Tests that call `run()` directly wrap it in
`asyncio.run(...)` — **no `pytest-asyncio` dependency added**.

**Rationale**: An async graph invocation must be awaited by an async caller.
FastAPI natively supports async endpoints and the existing `TestClient`
(Starlette) drives them synchronously, so the gateway integration tests need no
new machinery. For the handful of orchestrator unit tests that invoke `run()`
directly, `asyncio.run()` keeps them plain sync `def` tests with zero new deps —
consistent with the project's "minimal, pinned deps" posture. Verified
`asyncio.run()` drives `ainvoke` end-to-end.

**Alternatives considered**:
- *Add `pytest-asyncio`* — cleaner `async def test_...` ergonomics but a new dev
  dependency for a couple of call sites; rejected (YAGNI).

## R3 — HTTP client & the no-network test seam

**Decision**: Isolate the call behind an async helper
`generate_local(state, client) -> tuple[str | None, WorkerFailure | None, TokenUsage]`
that takes an injected `httpx.AsyncClient`, plus a module-level factory
`build_ollama_client() -> httpx.AsyncClient`. Tests inject a client wired with
`httpx.MockTransport` (a handler that returns canned JSON or raises transport
errors). `get_client`-style monkeypatching of `build_ollama_client` is the seam.

**Rationale**: This mirrors feature 004's `get_client()` /
`request_routing_decision(state, client)` split exactly, so the codebase stays
uniform. `httpx.MockTransport` gives fully deterministic, no-network responses and
can simulate connect errors and timeouts by raising inside the handler — covering
the entire US3 failure matrix without a live daemon. `httpx` is already a runtime
dependency (required by google-genai), so **no new package is needed**.

**Alternatives considered**:
- *`unittest.mock` patch of `client.post`* — works but `MockTransport` is the
  idiomatic, transport-level httpx fake and models real request/response flow more
  faithfully; preferred.
- *A separate fake-client class (as 004 did for the GenAI SDK)* — 004 needed it
  because the GenAI SDK isn't httpx-shaped; here the call *is* httpx, so
  `MockTransport` is the lighter, truer seam.

## R4 — Ollama `/api/generate` contract & usage extraction

**Decision**: POST `{"model": OLLAMA_MODEL, "prompt": <built>, "stream": false}`
to `OLLAMA_URL`. Parse the single JSON response: `response` → generated text;
`prompt_eval_count` → input tokens; `eval_count` → output tokens;
`total = input + output`. Missing counts default to 0.

**Rationale**: `stream: false` yields one complete JSON object carrying both the
text and the token counters, so a single await gets everything (spec Assumptions:
non-streaming). Ollama's documented non-stream fields are `response`,
`prompt_eval_count`, and `eval_count`; summing the two counts into
`TokenUsage(input, output, total)` feeds the existing `add_usage` field-wise
reducer unchanged (FR-005). Absent counters → zeros satisfies FR-006.

**Alternatives considered**:
- *Streaming (`stream: true`)* — would require aggregating chunks and only the
  final chunk carries counts; unnecessary complexity for a worker node. Rejected.

## R5 — Failure taxonomy for the local worker

**Decision**: Add `WorkerFailure(category, detail)` with
`category ∈ {unreachable, timeout, invalid_output}`. Mapping:

| Condition | category |
|-----------|----------|
| `httpx.TimeoutException` (connect/read/write/pool) | `timeout` |
| `httpx.ConnectError` / other `httpx.TransportError` | `unreachable` |
| non-2xx HTTP status | `invalid_output` |
| body not JSON / missing `response` field / decode error | `invalid_output` |

`generate_local` **never raises**; every exception maps to a `WorkerFailure`
(mirrors `request_routing_decision`). The node records a categorized message and
returns control to the Supervisor; the run keeps terminating via the three-layer
guard.

**Rationale**: Directly serves Constitution Principle IV (Fail-Safe Transparency)
and spec US3's three scenarios (unreachable / timeout / unusable response). A
missing model (`404`) and a `500` both surface as `invalid_output` — an
observable, non-fatal degradation rather than a crash. No credential category
exists because Ollama is local and unauthenticated.

**Alternatives considered**:
- *Reuse feature 004's `RoutingFailure`* — its `Literal` categories
  (auth/missing_credential/…) don't fit a local, keyless HTTP worker; a dedicated
  `WorkerFailure` keeps each taxonomy honest.
- *A dedicated `http_status` category* — folded into `invalid_output` to keep the
  set to the three the spec enumerates; the status code lives in `detail`.

## R6 — Worker failure does not hijack routing

**Decision**: On failure the worker records a `Message`
(`source="local_llm"`, `content="local inference failure: <category>"`), adds zero
usage, still lists itself in `visited`, and returns normally. It does **not** set
`next`/`status` — routing remains the Supervisor's job.

**Rationale**: Worker nodes are not routing authorities (only the Supervisor is,
per feature 004). A failed local call returns to the Supervisor, which decides the
next step; runaway ret/routing is bounded by `MAX_STEPS` + `recursion_limit`
(three-layer termination, FR-012). The failure is observable in the outcome's
message history, satisfying Principle IV without inventing a new terminal path.

## R7 — Configuration & timeout defaults

**Decision**: `OLLAMA_URL` (default `http://localhost:11434/api/generate`),
`OLLAMA_MODEL` (default `qwen2.5:7b`), `OLLAMA_TIMEOUT_MS` (default `120_000`),
all read from the environment — matching the `HEARTBEAT_CONFIDENCE_THRESHOLD` /
`GEMINI_API_KEY` env pattern.

**Rationale**: Zero-config in the common case (defaults are the pre-pulled model
and the standard local endpoint), overridable for other hosts/models (FR-010). A
7B local generation can take longer than the Supervisor's 10 s cloud call, so the
per-call bound defaults higher (120 s) while still bounding the run predictably
(SC-005); the timeout is passed to the httpx request so a hung daemon degrades to
`timeout` rather than hanging.
