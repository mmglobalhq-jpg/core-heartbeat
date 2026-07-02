# Feature Specification: Gateway Routing Interface

**Feature Branch**: `002-gateway-routing`

**Created**: 2026-07-01

**Status**: Draft

**Input**: User description: "Gateway routing interface for core-heartbeat. Build the FastAPI application instance in main.py and the routing endpoints in router.py. The primary endpoint accepts a POST request whose body is an IntentPayload. The router validates the payload and checks the confidence score against an environment-driven threshold. It returns a structured response confirming the outcome; the response schema must include an optional usage/metadata dictionary so token counts and cost tracking can be passed through later. Include a basic GET /health endpoint so other local systems can verify the gateway is online. Payloads that fail validation or fall below the confidence threshold are rejected with a structured, meaningful error response. Handler dispatch is explicitly out of scope for this MVP."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Submit a valid, confident intent and get an acknowledgment (Priority: P1)

An upstream producer submits a well-formed intent, whose confidence meets the acceptance threshold, to the gateway's primary submission endpoint. The gateway confirms the intent was received and validated by returning a structured success response that echoes the intent's identity and an explicit accepted status.

**Why this priority**: This is the gateway's reason to exist for this MVP — proving an intent can be received, validated, threshold-checked, and acknowledged over the wire. Without it there is no gateway. It is the smallest end-to-end slice that delivers value.

**Independent Test**: Submit a well-formed intent with confidence at or above the threshold; confirm the response is a structured success that reports the intent identity and an "accepted" status.

**Acceptance Scenarios**:

1. **Given** a well-formed intent with confidence above the threshold, **When** it is submitted to the gateway, **Then** the gateway returns a structured success response echoing the intent identity and an accepted status.
2. **Given** a well-formed intent with confidence exactly equal to the threshold, **When** it is submitted, **Then** it is accepted (the threshold is inclusive) and acknowledged.
3. **Given** a well-formed intent carrying parameters, **When** it is accepted, **Then** the success response confirms receipt without requiring the original submission to be re-sent.
4. **Given** an accepted intent, **When** the caller inspects the success response, **Then** the response exposes the shared envelope's optional usage/metadata field (a map) that MAY be empty or omitted in this MVP but is present in the contract for future token-count and cost pass-through.

---

### User Story 2 - Reject an intent below the confidence threshold (Priority: P2)

An upstream producer submits a well-formed intent whose confidence is below the acceptance threshold. The gateway declines to acknowledge it as accepted and instead returns a structured error response that clearly states the reason (confidence below threshold) and includes the offending and required values, so the caller can understand and react.

**Why this priority**: Threshold enforcement is the gateway's core policy decision for this MVP and prevents acting on low-certainty intents. It builds directly on US1 but is a distinct, independently testable behavior.

**Independent Test**: Submit a well-formed intent with confidence below the threshold; confirm the response is a structured rejection that names the threshold reason and reports both the submitted confidence and the required threshold.

**Acceptance Scenarios**:

1. **Given** a well-formed intent with confidence below the threshold, **When** it is submitted, **Then** the gateway returns a structured error response indicating the intent was not accepted due to insufficient confidence.
2. **Given** a below-threshold rejection, **When** the caller inspects the error, **Then** it reports the submitted confidence value and the threshold it failed to meet.
3. **Given** the acceptance threshold has been configured to a different value, **When** an intent is submitted, **Then** the accept/reject decision reflects the currently configured threshold.

---

### User Story 3 - Reject a malformed or invalid submission (Priority: P3)

An upstream producer submits a payload that does not satisfy the intent contract (missing a required field, an out-of-range confidence, an unknown extra field, or wrong types). The gateway rejects it before any threshold check with a structured validation error that identifies what was wrong, so the caller can correct and resubmit.

**Why this priority**: Robust input validation protects the gateway and downstream consumers and gives callers actionable feedback, but it is a guard around the primary flow rather than the primary flow itself.

**Independent Test**: Submit payloads that each violate one aspect of the intent contract; confirm each is rejected with a structured validation error identifying the problem, and that no threshold decision or acceptance occurs.

**Acceptance Scenarios**:

1. **Given** a submission missing a required field, **When** it is submitted, **Then** the gateway returns a structured validation error identifying the missing field and does not accept the intent.
2. **Given** a submission with a confidence outside the allowed range, **When** it is submitted, **Then** the gateway returns a structured validation error and performs no threshold comparison.
3. **Given** a submission carrying an unknown extra field, **When** it is submitted, **Then** the gateway rejects it as invalid (strict contract) rather than silently ignoring the field.

---

### User Story 4 - Verify the gateway is online (Priority: P3)

An operator or a neighboring local system needs to confirm the gateway process is up and reachable before sending intents to it. It calls a lightweight liveness endpoint and receives a simple, structured "online" status without submitting or affecting any intent.

**Why this priority**: Liveness checking is valuable for local integration and orchestration but is independent of the core submit/validate flow; the gateway delivers its primary value without it.

**Independent Test**: Call the liveness endpoint with no body; confirm a structured response reporting an online/healthy status, with no side effects on intent processing.

**Acceptance Scenarios**:

1. **Given** the gateway is running, **When** the liveness endpoint is called, **Then** it returns a structured response indicating the gateway is online.
2. **Given** the liveness endpoint is called, **When** the response is inspected, **Then** it requires no request body and does not create, validate, or acknowledge any intent.

---

### Edge Cases

- **Confidence exactly at threshold**: accepted — the threshold is inclusive (>=).
- **Empty request body / non-parseable submission**: rejected as a structured validation error, not an unhandled failure.
- **Valid contract but below threshold**: rejected as a policy rejection (US2), which is distinct from a contract validation error (US3); the two error kinds are distinguishable by the caller.
- **Missing/blank configured threshold**: the gateway falls back to a defined default threshold rather than failing to start or accepting everything.
- **Threshold configured out of the valid 0–1 range**: treated as a misconfiguration and surfaced clearly rather than silently clamping.
- **Extra parameters present but intent otherwise valid and confident**: accepted; parameters do not affect the accept/reject decision in this MVP.
- **Usage/metadata not provided by the gateway yet**: the response's usage/metadata field is present in the schema but empty or omitted; callers must tolerate its absence and not depend on populated values in this MVP.
- **Liveness endpoint called while no intents are in flight**: returns online regardless; it reflects process reachability, not intent-processing state.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The gateway MUST expose a primary submission endpoint that accepts an intent submission from an upstream producer.
- **FR-002**: The gateway MUST validate each submission against the intent contract before any other processing, rejecting submissions that are missing required fields, contain out-of-range values, include unknown extra fields, or use wrong types.
- **FR-003**: For submissions that pass validation, the gateway MUST compare the submission's confidence against a configured acceptance threshold.
- **FR-004**: The gateway MUST accept a validated submission whose confidence is greater than or equal to the threshold and reject one whose confidence is below the threshold.
- **FR-005**: On acceptance, the gateway MUST return a structured success response that includes the intent identity and an explicit accepted status confirming the intent was received and validated.
- **FR-006**: On a below-threshold rejection, the gateway MUST return a structured error response that states the reason and includes both the submitted confidence and the required threshold.
- **FR-007**: On a validation failure, the gateway MUST return a structured error response that identifies what was invalid.
- **FR-008**: The gateway MUST make validation errors and threshold rejections distinguishable from one another and from successful acceptances.
- **FR-009**: The acceptance threshold MUST be configurable without changing code, and MUST fall back to a defined default when not explicitly configured.
- **FR-010**: The gateway MUST NOT dispatch to, invoke, or select any intent-specific handler in this MVP; its responsibility ends at receive → validate → threshold-check → acknowledge.
- **FR-011**: The gateway MUST leave the submitted intent unmodified; acknowledgment reflects the intent as received.
- **FR-012**: The acceptance threshold MUST be sourced from an environment variable at startup, falling back to the defined default when the variable is unset or blank (FR-009 refined: the configuration source is the environment).
- **FR-013**: Every structured gateway response — both the success (acknowledgment) response and both kinds of rejection response (validation and threshold) — MUST share a consistent envelope that includes an optional usage/metadata field, expressed as a name→value map. The field MAY be empty or omitted in this MVP and exists to carry future pass-through data such as token counts and cost. The gateway MUST NOT be required to populate it in this MVP.
- **FR-014**: The gateway MUST expose a lightweight liveness endpoint that reports an online/healthy status, requires no request body, and has no effect on intent processing.

### Key Entities *(include if feature involves data)*

- **Intent Submission**: The incoming intent to be evaluated. Conforms to the established intent contract (identity, confidence, parameters, raw input, source, timestamp). Consumed, not persisted, by this feature.
- **Response Envelope**: The shared structure of every gateway response (success and both rejection kinds). Common attributes include an outcome indicator (distinguishing accepted / threshold-rejected / validation-rejected) and an **optional usage/metadata map** (may be empty/omitted in this MVP; reserved for future token-count and cost pass-through).
- **Acknowledgment Response**: The success variant of the envelope, returned on acceptance. Adds: intent identity, accepted status, and confirmation the intent was received and validated.
- **Rejection Response**: The error variant of the envelope, returned on failure. Two distinguishable kinds — validation failure (what was invalid) and threshold rejection (submitted confidence vs required threshold).
- **Acceptance Threshold**: The configurable confidence cutoff (a value in 0–1) governing accept/reject, sourced from an environment variable at startup, with a defined default when unconfigured.
- **Liveness Status**: The structured result of the liveness endpoint, reporting that the gateway process is online/reachable. Carries no intent data.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of well-formed submissions with confidence at or above the threshold receive a structured success response reporting the intent identity and accepted status.
- **SC-002**: 100% of well-formed submissions with confidence below the threshold receive a structured rejection that reports both the submitted confidence and the required threshold, and are never reported as accepted.
- **SC-003**: 100% of submissions violating the intent contract receive a structured validation error and never reach the threshold comparison.
- **SC-004**: A caller can distinguish the three outcomes — accepted, threshold-rejected, and validation-rejected — from the response alone in 100% of cases.
- **SC-005**: Changing the configured threshold changes the accept/reject boundary with no code change, verifiable by submitting the same intent under two different configured thresholds and observing opposite decisions.
- **SC-006**: No submission results in an unstructured or unhandled failure; every submission yields one of the three defined structured outcomes.
- **SC-007**: Every gateway response (success and both rejection kinds) carries the shared envelope's usage/metadata field in its schema; a caller can read it (finding it empty/omitted in this MVP) without the response shape changing when it is later populated.
- **SC-008**: The liveness endpoint returns an online status in 100% of calls while the gateway is running, requires no request body, and never alters intent-processing state.

## Assumptions

- The intent contract referenced here is the already-implemented `IntentPayload` model (strict, immutable, confidence bounded to 0–1). This feature reuses it as the submission body and does not redefine it.
- Default acceptance threshold is **0.5** when not explicitly configured; the threshold is read from an **environment variable** at startup (falling back to the default when unset or blank).
- All gateway responses share a **consistent envelope** carrying the optional usage/metadata field (success and both rejection kinds), so token-count and cost pass-through is uniform. The field is **optional and unpopulated** in this MVP — the schema reserves it so later features can populate it without a breaking response-shape change.
- The threshold comparison is **inclusive** at the boundary (confidence `>= threshold` is accepted).
- A below-threshold outcome is a **rejection** (structured error), not an "accepted-but-flagged" success — per the feature description.
- Validation is delegated to the intent contract's own construction-time rules; the gateway surfaces those failures as structured errors rather than re-implementing field checks.
- Handler dispatch, intent→handler mapping, persistence, authentication/authorization, and rate limiting are out of scope for this MVP.
- The gateway exposes exactly two endpoints in this MVP: the primary intent submission endpoint and a lightweight liveness endpoint. Deep/dependency health checks (beyond process reachability) are out of scope.
- Concurrency, throughput, and latency targets are not specified for this MVP; correctness of the three outcomes is the objective.
