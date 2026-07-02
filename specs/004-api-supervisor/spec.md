# Feature Specification: API-Driven Supervisor Node

**Feature Branch**: `004-api-supervisor`

**Created**: 2026-07-01

**Status**: Draft

**Input**: User description: "Replace the stubbed Supervisor node in the orchestration engine with a live call to a hosted LLM (Gemini 2.5 Flash) via the official Google GenAI SDK. Read the API key from the environment (GEMINI_API_KEY). Enforce a strict JSON output schema so the model reliably returns a routing decision constrained to {local_llm, tool_execution, finish}. Safely catch API timeouts / auth errors / network errors / missing key / malformed output and return a clean failure state so the graph terminates gracefully rather than crashing. Preserve the three-layer termination guarantees. Capture model token usage into the usage tracker. local_llm and tool_execution remain stubbed."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Model-driven routing with an enforced decision schema (Priority: P1)

The Supervisor, when it needs to decide the next step, consults a hosted language model instead of a fixed rule. It presents the current orchestration state (the intent and accumulated message history) and receives a routing decision constrained to exactly one of the allowed next steps — route to the local model node, route to the tool node, or finish. A valid decision drives the graph exactly as that step.

**Why this priority**: This is the feature's essence — turning the Supervisor from a hardcoded router into a model-driven one while keeping the routing contract intact. It is the smallest slice that delivers the new capability.

**Independent Test**: With a substituted model client returning each allowed decision in turn, confirm the Supervisor routes to the matching next step and the graph proceeds accordingly.

**Acceptance Scenarios**:

1. **Given** the model returns a valid decision of "route to local model", **When** the Supervisor runs, **Then** the graph proceeds to the local model node.
2. **Given** the model returns a valid decision of "route to tool", **When** the Supervisor runs, **Then** the graph proceeds to the tool node.
3. **Given** the model returns a valid decision of "finish", **When** the Supervisor runs, **Then** the run terminates.
4. **Given** the model returns any value outside the allowed set of decisions, **When** the Supervisor evaluates it, **Then** that value is NOT accepted as a routing decision (it is treated as a failure, per US2).

---

### User Story 2 - Graceful degradation on any model failure (Priority: P2)

Because the Supervisor now depends on an external service, it must never let that dependency crash or hang the graph. Any failure — a missing or invalid credential, an authentication error, a request timeout, a network error, or a model response that fails schema validation — is caught and resolved to a safe terminal decision (finish). The run terminates, the failure is recorded so it is observable, and the caller still receives a well-formed outcome.

**Why this priority**: Reliability of the whole orchestration hinges on the Supervisor failing safe. Without this, a single external hiccup could hang or crash a request. It builds on US1 but is independently testable and essential to shipping a live dependency.

**Independent Test**: With a substituted client that raises each failure mode (and one that returns invalid output), confirm every case resolves to a safe finish, the run terminates, and the failure category is recorded — with no crash or hang.

**Acceptance Scenarios**:

1. **Given** the credential is missing, **When** the Supervisor runs, **Then** it resolves to finish, records a "missing credential" failure, and the run terminates gracefully.
2. **Given** the model call raises an authentication error, **When** the Supervisor runs, **Then** it resolves to finish, records an "authentication" failure, and terminates.
3. **Given** the model call times out, **When** the Supervisor runs, **Then** it resolves to finish within the bounded wait, records a "timeout" failure, and terminates.
4. **Given** a network error occurs, **When** the Supervisor runs, **Then** it resolves to finish, records a "network" failure, and terminates.
5. **Given** the model returns malformed or out-of-vocabulary output, **When** the Supervisor evaluates it, **Then** it resolves to finish, records an "invalid output" failure, and terminates.
6. **Given** any of the above, **When** the run ends, **Then** the caller receives a complete orchestration outcome (degraded) rather than an unhandled error.

---

### User Story 3 - Capture model token usage (Priority: P3)

When the model call reports token usage, the Supervisor adds those counts to the run's existing usage tracker, so the usage/cost accounting reflects the real model calls (not only the stub increments).

**Why this priority**: Usage accounting is valuable for cost visibility and completes the usage thread, but the engine functions and stays safe without it.

**Independent Test**: With a substituted client reporting known token counts, confirm those counts are added to the run's usage totals; with a client reporting none, confirm the totals are unaffected and no error occurs.

**Acceptance Scenarios**:

1. **Given** the model reports token usage for a call, **When** the Supervisor completes, **Then** those tokens are added to the run's usage totals.
2. **Given** the model reports no usage, **When** the Supervisor completes, **Then** the usage totals are unchanged and no error occurs.

---

### Edge Cases

- **Missing credential**: treated as a failure → safe finish, recorded; the service does not crash on startup or per request.
- **Invalid credential / auth error**: safe finish, recorded.
- **Timeout**: the Supervisor waits no longer than the configured bound, then safe finish, recorded.
- **Network error**: safe finish, recorded.
- **Valid JSON but out-of-vocabulary decision** (e.g. an unknown node name): rejected, not acted on → safe finish, recorded.
- **Unparseable / non-conforming output**: rejected → safe finish, recorded.
- **Model returns a valid decision but the step bound is reached**: the existing step bound still finishes the run (termination guarantees layered).
- **Repeated Supervisor visits**: each visit is a separate model call; the step bound caps the number of calls per run (bounds cost and latency).
- **Model reports no usage**: usage tracker is left unchanged; no error.
- **Degraded run**: a run whose Supervisor failed still returns a complete outcome, distinguishable as degraded from a normal completion.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The Supervisor MUST derive its routing decision from a hosted language model (not a fixed rule), given the current orchestration state (intent + accumulated message history).
- **FR-002**: The Supervisor MUST constrain the model response to a strict schema whose decision is exactly one of the allowed next steps (route-to-local-model, route-to-tool, finish); any response outside this vocabulary or that fails to parse/validate MUST NOT be accepted as a decision.
- **FR-003**: The Supervisor MUST read the model credential from a designated environment variable and MUST NOT hardcode or log it.
- **FR-004**: The Supervisor MUST catch the failure modes — missing/invalid credential, authentication error, request timeout, network error, and malformed/invalid model output — without crashing or hanging the graph.
- **FR-005**: On any such failure, the Supervisor MUST resolve to a safe terminal decision (finish) so the run terminates.
- **FR-006**: The Supervisor MUST bound the time it waits for a model response (a request timeout), so an unresponsive model cannot hang the run.
- **FR-007**: The existing termination guarantees (Supervisor finish decision, step bound, recursion limit) MUST continue to hold; a degraded or failed model call MUST still lead to termination.
- **FR-008**: The run MUST record that a Supervisor model-routing failure occurred and its category, so the degradation is observable in the run outcome (never silent).
- **FR-009**: The Supervisor MUST add the token usage reported by the model call to the run's usage tracker when such usage is available, and MUST NOT error when it is not.
- **FR-010**: A valid model decision MUST drive routing to exactly the corresponding next step; the downstream routing/termination behavior is otherwise unchanged from the prior Supervisor.
- **FR-011**: The local-model and tool nodes MUST remain stubbed; only the Supervisor becomes model-driven in this feature.
- **FR-012**: The Supervisor MUST be testable deterministically by substituting a fake/mock model client so tests make no real network calls, covering valid decisions, each failure mode, and invalid-output handling.

### Key Entities *(include if feature involves data)*

- **Routing Decision**: The strict, schema-constrained result of a Supervisor model call. A single field naming the next step, restricted to the allowed set (route-to-local-model, route-to-tool, finish). Out-of-set or unparseable values are invalid and rejected.
- **Supervisor Routing Context**: The information presented to the model to decide — the intent and the accumulated message history of the run.
- **Routing Failure**: A recorded, categorized degradation (missing credential, authentication, timeout, network, invalid output) with an associated safe fallback decision (finish). Makes the failure observable in the outcome.
- **Model Usage**: Token counts reported by the model call, added to the run's existing usage tracker when available.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of routing decisions acted upon are within the allowed vocabulary; no out-of-vocabulary or unparseable model response is ever acted upon as a decision.
- **SC-002**: 100% of runs terminate even when the model call fails (any failure mode) — no crash and no hang, verified across all five failure categories.
- **SC-003**: When the model is unavailable or misconfigured, a submitted intent still yields a complete (degraded) orchestration outcome — the caller never receives an unhandled error.
- **SC-004**: The Supervisor never waits longer than the configured timeout bound for a model response.
- **SC-005**: 100% of Supervisor model-routing failures are recorded with a category in the run outcome (observable, never silent).
- **SC-006**: When the model reports token usage, it is reflected in the run's usage totals; when it reports none, totals are unchanged and no error occurs.
- **SC-007**: The feature is fully verifiable without real network access — valid-decision, each failure mode, and invalid-output paths are covered by deterministic tests using a substituted model client.

## Assumptions

- The hosted model is Gemini 2.5 Flash accessed via the official Google GenAI SDK (implementation detail resolved in planning); the SDK's installability on Python 3.14 will be verified in planning before committing to it.
- The credential is read from the `GEMINI_API_KEY` environment variable.
- A default request timeout (a few seconds) bounds each call; the exact value is a planning detail and configurable.
- **No automatic retries** in this MVP — a failure resolves straight to a safe finish (fail fast). Retry/backoff is a future enhancement.
- On failure, the run degrades to finish and still returns an orchestration outcome (consistent with the orchestration feature); the gateway still returns its accepted envelope. A degraded run is distinguishable from a normal completion via the recorded failure.
- The credential is handled gracefully at call time (missing key → safe finish), not as a hard startup crash, so the service stays up and degrades rather than failing to boot.
- Each Supervisor visit issues one model call; the existing step bound caps the number of model calls per run (bounding cost and latency).
- The local-model and tool nodes remain deterministic stubs; only the Supervisor becomes live.
- Because routing is now model-driven, the Supervisor is not strictly deterministic in production; determinism for tests is achieved by substituting a fake model client.
- Real billing/cost management, prompt-quality tuning, response streaming, and multi-model fallback are out of scope for this feature.
