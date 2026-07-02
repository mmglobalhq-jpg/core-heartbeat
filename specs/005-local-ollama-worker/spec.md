# Feature Specification: Local Ollama Worker Node

**Feature Branch**: `005-local-ollama-worker`

**Created**: 2026-07-02

**Status**: Draft

**Input**: User description: "Local Ollama Worker Node — Replace the stubbed local_llm node in orchestrator.py with an asynchronous HTTP call to the local Ollama service running on http://localhost:11434/api/generate. Configure it to invoke the qwen2.5:7b model we just pulled. Ensure token usage returned by Ollama's response is extracted and summed correctly into the existing graph state's add_usage tracker so it remains observable. Use a Mock/Fake block for tests so we don't accidentally depend on the local daemon being up during CI validation."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Local model produces real inference output (Priority: P1)

When the Supervisor routes a run to the local worker, the worker sends the run's
prompt to the locally hosted model and returns that model's actual generated text
as the worker's result, replacing the previous fixed placeholder string. The run
then continues through the graph exactly as before, carrying the real output in
its message history.

**Why this priority**: This is the core of the feature — turning the stubbed
local worker into a live local-inference step. Without it, nothing else in this
feature has value. It is the minimum shippable slice: a routed run comes back
with genuine local model output.

**Independent Test**: With the local model responder simulated (no real daemon),
route a run to the local worker and confirm the worker's recorded message
contains the responder's returned text rather than the old placeholder, and that
the run still reaches a terminating outcome.

**Acceptance Scenarios**:

1. **Given** an accepted intent whose Supervisor routes to the local worker,
   **When** the local model returns generated text, **Then** the worker records a
   message carrying that generated text and control returns to the Supervisor.
2. **Given** the same run, **When** the outcome is returned to the gateway,
   **Then** the local worker appears in the executed-nodes list exactly as it did
   when it was a stub.

---

### User Story 2 - Local model token usage stays observable (Priority: P2)

The token counts reported by the local model for a call (prompt tokens and
generated tokens) are extracted from its response and added into the run's
existing cumulative usage tracker, so the totals returned to the gateway reflect
the real cost of the local inference step alongside the Supervisor's usage.

**Why this priority**: Observability of cost is a standing project principle
(usage must remain visible). The feature is only complete if real local usage is
captured, not silently dropped — but the run can still function (US1) before this
is wired, so it ranks below P1.

**Independent Test**: With a simulated responder reporting known prompt/generated
token counts, run a routed intent and confirm the run's usage totals increase by
exactly those counts (field-wise), and that a response reporting no counts leaves
the totals unchanged without error.

**Acceptance Scenarios**:

1. **Given** a local model response reporting known input and output token
   counts, **When** the worker completes, **Then** the run's cumulative usage
   increases by exactly those input, output, and total amounts.
2. **Given** a local model response that omits token counts, **When** the worker
   completes, **Then** the run's usage is unchanged for this step and no error is
   raised.

---

### User Story 3 - Local worker degrades safely when the model is unavailable (Priority: P3)

If the local model service is unreachable, slow to the point of timing out, or
returns an unusable response, the worker does not crash or hang the run. Instead
it records an explicitly categorized failure in the run's message history and
returns control so the run still terminates with a complete, observable outcome.

**Why this priority**: Fail-safe transparency is a project principle — no silent
or fatal failures. This protects every run from a flaky or absent local daemon.
It ranks P3 because the happy path (US1/US2) delivers the feature's value first,
but this must ship for the feature to be trustworthy.

**Independent Test**: With a simulated responder that raises a connection error,
a timeout, and a malformed response in turn, run a routed intent for each and
confirm each run terminates with a recorded, categorized local-worker failure and
no unhandled exception.

**Acceptance Scenarios**:

1. **Given** the local model service is unreachable, **When** the worker runs,
   **Then** a categorized failure is recorded, the run terminates, and no
   exception propagates to the gateway.
2. **Given** the local model call exceeds its time bound, **When** the worker
   runs, **Then** the call is abandoned, a timeout failure is recorded, and the
   run terminates.
3. **Given** the local model returns a response the worker cannot interpret,
   **When** the worker runs, **Then** an invalid-output failure is recorded and
   the run terminates.

---

### Edge Cases

- **Model not pulled / unknown model**: the configured model name is not present
  on the local service — treated as a categorized failure (per US3), not a crash.
- **Empty generated text**: the model returns a successful response with empty
  output — recorded as a message with empty content; the run continues normally.
- **Usage present but partial**: only one of prompt/generated counts is reported —
  the present count is added, the missing one contributes zero.
- **Concurrent runs**: multiple accepted intents route to the local worker at once
  — each run's usage and messages remain isolated to that run.
- **Test isolation**: the full test suite passes with the local service stopped —
  no test may reach the real daemon or the network.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The local worker node MUST obtain generated text from the locally
  hosted model service instead of returning a fixed placeholder string.
- **FR-002**: The local worker MUST build the model request from the current run's
  intent and message history so the model responds to the run's actual context.
- **FR-003**: The local worker MUST record the model's generated text as a message
  attributed to the local worker in the run's message history.
- **FR-004**: The local worker MUST return control to the Supervisor after
  completing, preserving the existing graph flow and the node's presence in the
  executed-nodes list.
- **FR-005**: The local worker MUST extract the model's reported input (prompt) and
  output (generated) token counts and add them, field-wise, into the run's
  existing cumulative usage tracker.
- **FR-006**: When the model response omits token counts, the worker MUST treat
  the contribution as zero and MUST NOT error.
- **FR-007**: The local worker MUST bound each model call in time so an unresponsive
  service cannot hang a run.
- **FR-008**: The local worker MUST categorize and record any failure — service
  unreachable, timeout, or unusable response — as an observable entry in the run's
  history rather than crashing or silently swallowing it.
- **FR-009**: On any local-worker failure, the run MUST still terminate with a
  complete outcome returned to the gateway.
- **FR-010**: The target model identity and the local service endpoint MUST be
  configurable, defaulting to the pre-pulled local model and the standard local
  endpoint so no configuration is required in the common case.
- **FR-011**: The automated test suite MUST validate the local worker without
  contacting the real local service or the network, using a substituted responder,
  so validation passes whether or not the daemon is running.
- **FR-012**: The three-layer termination guarantee (Supervisor finish/degrade,
  step bound, recursion limit) MUST remain intact; the local worker MUST NOT
  introduce an unbounded or non-terminating path.

### Key Entities *(include if feature involves data)*

- **Local model request**: the prompt/context sent to the local model, derived
  from the run's intent and message history.
- **Local model response**: the model's returned generated text plus its reported
  token usage (input/prompt count and output/generated count).
- **Local worker failure**: a categorized record (unreachable / timeout /
  invalid-output) describing why a local inference step could not produce output,
  carried into the run's observable history.
- **Usage tracker**: the run's existing cumulative token accumulator, into which
  the local model's reported usage is summed.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A run routed to the local worker returns the model's real generated
  text in its history in 100% of successful cases — zero occurrences of the former
  placeholder string remain.
- **SC-002**: Reported local model token usage is reflected in the run's totals in
  100% of cases where the model reports counts; the run total equals the field-wise
  sum of every node's usage.
- **SC-003**: 100% of local-worker failure modes (unreachable, timeout, unusable
  response) result in a terminating run with a recorded, categorized failure and
  zero unhandled exceptions reaching the gateway.
- **SC-004**: The complete automated test suite passes with the local model service
  stopped, with zero real network or daemon calls made during the run.
- **SC-005**: A run whose model call exceeds the configured time bound completes
  (terminates and returns) within a bounded, predictable duration rather than
  hanging indefinitely.

## Assumptions

- **Fail-safe over fatal**: Per the project constitution (Fail-Safe Transparency),
  an unavailable or misbehaving local model degrades to a recorded, categorized
  failure and lets the run terminate; it does not abort the request. The failure
  categories mirror the Supervisor's taxonomy (unreachable/timeout/invalid-output).
- **Non-streaming call**: The worker requests a single complete response (not a
  token stream), so one response carries both the generated text and the token
  counts.
- **Prompt construction** reuses the run's intent and message history, consistent
  with how the Supervisor already builds its prompt.
- **Defaults**: The local endpoint defaults to the standard local Ollama address
  (`http://localhost:11434/api/generate`) and the model defaults to the already-
  pulled `qwen2.5:7b`; both are overridable via environment configuration, matching
  the credential/threshold configuration pattern used elsewhere in the service.
- **Scope**: Only the local worker (`local_llm`) becomes live in this feature; the
  external tool node (`tool_execution`) remains a stub and the Supervisor's
  model-driven routing (feature 004) is unchanged.
- **Testing substitution**: Tests inject a fake/mock responder in place of the real
  HTTP call — the same no-network, no-spend discipline used for the Supervisor's
  fake client in feature 004.
- **Runtime dependency**: A local Ollama daemon with `qwen2.5:7b` pulled is assumed
  present in the real runtime environment, but is explicitly NOT required for tests
  or for the service to boot (a missing daemon degrades per FR-008/FR-009).
