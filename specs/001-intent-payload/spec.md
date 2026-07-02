# Feature Specification: Intent Payload

**Feature Branch**: `001-intent-payload`

**Created**: 2026-07-01

**Status**: Draft

**Input**: User description: "IntentPayload data structure — the core payload that carries a parsed user/system intent through the core-heartbeat service. It is produced at the API boundary and consumed by the router to dispatch to the right handler. Should capture the intent identity, a confidence signal, any extracted entities/parameters, the originating raw input, and metadata (e.g. timestamp, source). Exact fields to be defined by this spec."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Route a recognized intent to the correct handler (Priority: P1)

An incoming request is parsed into a structured intent at the service boundary. That structured intent carries everything the router needs — the intent's identity and any extracted parameters — so the router can dispatch to exactly one handler without re-parsing or reaching back to the original request.

**Why this priority**: This is the reason the payload exists. Without a reliable, self-contained intent structure, the router cannot make a dispatch decision. This story alone delivers the core value: parsed input in, correct handler selected.

**Independent Test**: Construct an intent payload with a known intent identity and a set of parameters, hand it to the router, and confirm the router selects the handler mapped to that identity and receives the parameters intact.

**Acceptance Scenarios**:

1. **Given** a payload with a recognized intent identity and complete parameters, **When** the router processes it, **Then** exactly one handler is selected and receives all parameters unchanged.
2. **Given** a payload with an intent identity that maps to no handler, **When** the router processes it, **Then** the request is rejected/routed to a fallback and no handler is invoked.

---

### User Story 2 - Make confidence-aware routing decisions (Priority: P2)

The parsed intent includes a confidence signal describing how certain the upstream parser is about the identified intent. Downstream logic can use this signal to accept, reject, or escalate an intent (e.g. ask for clarification) rather than blindly dispatching a low-certainty guess.

**Why this priority**: Confidence-aware handling prevents acting on unreliable interpretations, but the system can still function (dispatching every recognized intent) without it. Valuable, not foundational.

**Independent Test**: Construct payloads spanning the confidence range and confirm each can be evaluated against a decision threshold, with out-of-range values rejected at construction.

**Acceptance Scenarios**:

1. **Given** a payload with a confidence at or above the acceptance threshold, **When** it is evaluated, **Then** it is eligible for normal dispatch.
2. **Given** a payload with a confidence below the acceptance threshold, **When** it is evaluated, **Then** it is flagged for fallback/clarification rather than normal dispatch.
3. **Given** a confidence value outside the valid range, **When** the payload is constructed, **Then** construction fails with a clear validation error.

---

### User Story 3 - Trace and audit an intent back to its origin (Priority: P3)

Each intent payload carries metadata — when it was created, where it came from, and the original raw input — so any downstream decision can be traced, logged, and reproduced during debugging or auditing.

**Why this priority**: Traceability is important for operability but not required to route a single request. It pays off across many requests over time.

**Independent Test**: Construct a payload, inspect its metadata and raw input, and confirm they capture the origin and timing needed to reconstruct how the intent was produced.

**Acceptance Scenarios**:

1. **Given** a constructed payload, **When** its metadata is inspected, **Then** it reports a creation timestamp and an identifiable source.
2. **Given** a constructed payload, **When** its raw input is inspected, **Then** the original unparsed input that produced the intent is available.

---

### Edge Cases

- **Missing intent identity**: A payload cannot be constructed without an intent identity; construction fails with a validation error rather than producing an unroutable payload.
- **No extracted entities**: A recognized intent may legitimately carry zero parameters (e.g. a "ping" intent). An empty parameter set is valid, not an error.
- **Out-of-range confidence**: Confidence outside its defined bounds is rejected at construction.
- **Unknown intent identity at routing time**: A structurally valid payload whose identity maps to no handler is handled by a defined fallback, not an unhandled error.
- **Oversized or malformed raw input**: The payload preserves raw input for traceability but does not itself re-interpret it; excessively large input handling is delegated to the boundary that produces the payload.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The payload MUST carry a single intent identity that uniquely names the recognized intent and is sufficient for the router to select a handler.
- **FR-002**: The payload MUST require an intent identity; a payload without one MUST be rejected at construction.
- **FR-003**: The payload MUST carry a confidence signal, expressed as a normalized value between 0 and 1 inclusive, indicating certainty in the identified intent.
- **FR-004**: The payload MUST reject confidence values outside the 0–1 inclusive range at construction with a clear validation error.
- **FR-005**: The payload MUST carry a collection of extracted entities/parameters keyed by name, and MUST permit that collection to be empty.
- **FR-006**: The payload MUST preserve the originating raw input that produced the intent, for traceability.
- **FR-007**: The payload MUST carry a creation timestamp indicating when the intent was produced.
- **FR-008**: The payload MUST carry a source identifier indicating the origin of the intent (e.g. which channel or upstream component produced it).
- **FR-009**: The payload MUST be validated at construction so that any consumer receiving a payload can trust its structural contract without re-validating.
- **FR-010**: The payload MUST be serializable to and reconstructable from a plain data representation without loss, so it can cross the service boundary.
- **FR-011**: Consumers MUST be able to read the intent identity and parameters without access to the original request context.

### Key Entities *(include if feature involves data)*

- **Intent Payload**: The self-contained record of one parsed intent moving through the service. Attributes: intent identity (required), confidence (required, 0–1), extracted entities/parameters (required, may be empty), raw input (required), creation timestamp (required), source (required).
- **Extracted Entities**: A named collection of parameters pulled from the input that a handler needs to act on the intent. Related to Intent Payload as a contained value.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of payloads accepted by the boundary carry all required fields (identity, confidence, parameters, raw input, timestamp, source); payloads missing any required field are rejected at construction.
- **SC-002**: 100% of payloads with a valid intent identity are routed to a single deterministic outcome (a mapped handler or a defined fallback), with no unhandled cases.
- **SC-003**: 100% of confidence values outside the 0–1 range are rejected at construction rather than reaching the router.
- **SC-004**: Any payload can be serialized and reconstructed with zero field loss, verified by round-trip equality.
- **SC-005**: For any dispatched intent, its origin can be reconstructed from the payload's metadata and raw input alone in 100% of cases.

## Assumptions

- The upstream parser (at the service boundary) is responsible for producing the intent identity and confidence; this payload defines the contract it must satisfy, not the parsing logic itself.
- The mapping from intent identity to handler lives in the router and is out of scope for this payload; the payload only guarantees the identity is present and readable.
- The acceptance/rejection confidence threshold is a routing policy decision owned by the router, not encoded in the payload; the payload only guarantees a valid, bounded confidence value.
- Confidence is a single normalized scalar in [0, 1]; per-entity or multi-hypothesis confidence is out of scope for v1.
- Parameters are a flat name→value collection; deeply nested or typed parameter schemas per intent are out of scope for v1.
- Raw input is retained verbatim for traceability; the payload does not sanitize, truncate, or re-parse it.
- A single intent identity per payload (no multi-intent payloads) is sufficient for v1.
