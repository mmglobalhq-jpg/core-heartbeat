# Phase 0 Research: API-Driven Supervisor Node

All Technical Context unknowns resolved. No open NEEDS CLARIFICATION.

## R1 — CRITICAL: Google GenAI SDK on Python 3.14 (dependency viability)

- **Decision**: Adopt **google-genai 2.10.0**. Verified in the venv on Python 3.14.4: installs cleanly; `from google import genai`, `from google.genai import types, errors` import. Structured-output surface confirmed: `types.GenerateContentConfig` exposes `response_mime_type`, `response_schema`, and `http_options`; `types.HttpOptions` exists (timeout); `errors.APIError` / `errors.ClientError` / `errors.ServerError` exist; `types.GenerateContentResponse` exists (carries `usage_metadata`).
- **Transitive deps pulled**: google-auth 2.55.1, cryptography 49.0.0, cffi 2.0.0, pyasn1 0.6.3, pyasn1-modules 0.4.2, pycparser 3.0. Already-present deps reused: anyio, distro, pydantic, requests, sniffio, tenacity, typing-extensions, websockets.
- **IMPORTANT — httpx becomes runtime**: `google-genai` **requires httpx** (its HTTP transport). So `httpx` (and `httpcore`) move from dev-only into `requirements.txt`. `certifi` is already runtime. The pytest stack stays dev-only.
- **Rationale**: This was the feature's blocking risk (as with LangGraph in 003). Resolving install + API surface up front avoids mid-implementation surprises. No real API call is made during research (paid + needs a key).
- **Alternatives considered**: Google's older `google-generativeai` package — superseded by `google-genai`, which the user specified. REST via raw httpx — rejected; the SDK gives typed structured output + error classes.

## R2 — Model client construction & dependency injection

- **Decision**: `MODEL_NAME = "gemini-2.5-flash"`. A module-level `get_client() -> genai.Client | None` reads `GEMINI_API_KEY` from the environment; returns `None` when unset/blank (so a missing key becomes a categorized failure, not a construction crash), else `genai.Client(api_key=...)`. The `supervisor` node calls `get_client()` **at invoke time**, and the model call is isolated in `request_routing_decision(state, client)`. Tests `monkeypatch orchestrator.get_client` (or pass a fake) to inject a fake client.
- **Rationale**: Dynamic `get_client()` lookup means the module-level compiled `graph` still works while tests fully control the client with no network (FR-012, SC-007). Reading the key in `get_client()` (not at import) keeps the service bootable without a key and lets it degrade per request (spec assumption). Credential is passed to the SDK and never logged (FR-003).
- **Alternatives considered**: Building the graph with an injected client (`build_graph(client)`) — also viable but complicates the module-level `graph`/`run()`; monkeypatching `get_client` is simpler and covers the endpoint path too. Global mutable client set by tests — less clean than monkeypatch.

## R3 — Strict structured output & decision validation

- **Decision**: Define `RoutingDecision(BaseModel)` with `next_node: Literal["local_llm", "tool_execution", "finish"]` and pass it as `config.response_schema` with `response_mime_type="application/json"`. Read the parsed result (`response.parsed` when available, else `json.loads(response.text)` validated through `RoutingDecision.model_validate`). Any value outside the Literal, unparseable JSON, or a Pydantic `ValidationError` → **not accepted**; treated as an `invalid_output` failure (R4).
- **Rationale**: The Literal makes the vocabulary un-representable-if-wrong at the schema layer, and re-validating with Pydantic guards against a model that ignores the schema (SC-001). The three literals exactly match the graph's conditional-edge mapping keys (`local_llm`/`tool_execution`/`finish`→END), so a valid decision routes with no translation (FR-010).
- **Alternatives considered**: Free-text parsing with regex — brittle; rejected. Enum instead of Literal — equivalent; Literal is lighter for a Pydantic response schema.

## R4 — Failure taxonomy, timeout, and safe degradation

- **Decision**: Bound the call with `http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS)` (a few seconds, configurable). Wrap `request_routing_decision` to map exceptions to a categorized `RoutingFailure`:
  | Category | Trigger |
  |----------|---------|
  | `missing_credential` | `get_client()` returned `None` (no `GEMINI_API_KEY`) |
  | `auth` | `genai.errors.ClientError` with status 401/403 |
  | `timeout` | `httpx.TimeoutException` (and subclasses) |
  | `network` | `httpx.TransportError`/`ConnectError`, `genai.errors.ServerError`, other non-auth `APIError` |
  | `invalid_output` | JSON decode error, Pydantic `ValidationError`, out-of-vocab value |
  Any failure → `supervisor` sets `next="finish"`, `status="degraded"`, appends a `Message(source="supervisor", content="routing failure: <category>", …)`. The run terminates and returns a complete outcome (`OrchestrationOutcome.status="degraded"`), observable by the caller (FR-005/FR-008, SC-002/003/005).
- **Rationale**: Explicit categories make failures observable and testable (a fake client raises each type). Folding non-auth API/server errors into `network` keeps the taxonomy small while covering real transient failures. `status="degraded"` distinguishes a failed-routing finish from a normal `completed`.
- **Alternatives considered**: Retjuries/backoff — out of scope (fail fast per spec). A single generic "error" category — loses observability required by FR-008.

## R5 — Usage capture

- **Decision**: When the response carries `usage_metadata`, map its token counts (`prompt_token_count`, `candidates_token_count`, `total_token_count`) into a `TokenUsage` and return it from `request_routing_decision`; `supervisor` returns it on the `usage` channel so the `add_usage` reducer accumulates it. When absent (or on failure), return `TokenUsage()` (zeros) — no error (FR-009, SC-006).
- **Rationale**: Reuses the existing additive usage channel from feature 003; the envelope `usage` field then reflects real model tokens plus stub increments. Guarding for absent metadata avoids errors on failure paths.
- **Alternatives considered**: Separate model-usage field — unnecessary; the existing `TokenUsage` accumulator already threads to the response.

## R6 — Supervisor control flow & termination preservation

- **Decision**: `supervisor(state)`:
  1. If `step >= MAX_STEPS` → finish (`status="halted_step_bound"`), **no model call** (bounds cost + preserves layer-2 termination).
  2. Else `client = get_client()`; if `None` → degraded finish (`missing_credential`).
  3. Else `decision, failure, usage = request_routing_decision(state, client)`; on failure → degraded finish; on success → `next = decision.next_node` (and `status="completed"` when it's `finish`).
  Return increments `step` and adds `usage`. Routing edge + worker nodes + `recursion_limit` unchanged. The deterministic `NOOP_INTENTS` shortcut is **removed** (routing is now model-driven).
- **Rationale**: Keeping the `MAX_STEPS` pre-check preserves the three-layer termination (Supervisor finish + step bound + recursion limit) AND caps model calls per run. Removing `NOOP_INTENTS` honors FR-001 (model decides). Determinism now comes from the injected fake client, not a fixed rule.
- **Test impact**: Existing feature-003 tests that assumed rule-based routing must inject a fake client. With a fake scripted `["local_llm","tool_execution","finish"]`, the greet-plan trace and `total_tokens==35` (stub) reproduce; the old "noop immediate finish" is reframed as "fake returns finish first call → nodes_executed==[]". Gateway integration tests monkeypatch `get_client` for accepted runs. This is required work, flagged for tasks.

## R7 — requirements split update

- **Decision**: Regenerate `requirements.txt` from `pip freeze` minus the **dev-only** set, which is now just the pytest stack (`pytest`, `iniconfig`, `pluggy`, `pygments`). **Move `httpx` and `httpcore` into `requirements.txt`** (google-genai runtime dep). Add the google-genai stack (google-genai, google-auth, cryptography, cffi, pyasn1, pyasn1-modules, pycparser). Verify no dev-only package leaks into `requirements.txt` and that `google-genai` imports from a clean runtime set.
- **Rationale**: httpx is no longer test-only; it is a transitive runtime dependency of the SDK. Keeping the split honest matters for reproducible deploys.
- **Alternatives considered**: Leaving httpx in dev — wrong now; a production install from `requirements.txt` would be missing an SDK dependency.

## Cross-cutting: what stays OUT

Per spec Assumptions — no retries/backoff, no streaming, no multi-model fallback, no prompt-quality tuning, no billing/cost management. `local_llm`/`tool_execution` remain deterministic stubs; only the Supervisor becomes live.
