# Feature Specification: Orchestration Engine

**Feature Branch**: `003-langgraph-orchestration`

**Created**: 2026-07-01

**Status**: Draft

**Input**: User description: "LangGraph Orchestration Engine for core-heartbeat handlers. Define a GraphState carrying the IntentPayload, a message history list, and a token usage tracker, and a basic cyclic graph with three nodes: a Supervisor that reads the intent and routes; a Local-LLM node (stubbed local inference); and a Tool-Execution node (stubbed). The Supervisor is the routing hub with a defined entry point and a termination condition so cycles cannot run forever. The gateway's router.py is updated so an accepted, validated payload triggers the compiled graph and returns the orchestration outcome and accumulated usage. Node bodies are stubbed/deterministic for this feature; real inference and real tools are out of scope."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Orchestrate an accepted intent to a terminating outcome (Priority: P1)

An accepted, validated intent is handed to the orchestration engine, which starts at the Supervisor, routes the work through one or more nodes, and halts with a structured outcome describing what happened. The engine never runs forever.

**Why this priority**: This is the engine's reason to exist — turning an accepted intent into a completed, bounded orchestration run with a result. Without a graph that starts, does work, and terminates, nothing downstream is possible. It is the smallest end-to-end slice that delivers value.

**Independent Test**: Feed an accepted intent to the engine and confirm it returns a structured outcome with a terminal status, and that the run halts (does not loop indefinitely).

**Acceptance Scenarios**:

1. **Given** an accepted intent, **When** it is submitted to the engine, **Then** execution begins at the Supervisor and ends with a structured outcome carrying a terminal status.
2. **Given** the engine has run, **When** the outcome is inspected, **Then** it reports which nodes executed and the final accumulated state (message history and usage).
3. **Given** any accepted intent, **When** it is orchestrated, **Then** the run terminates within the defined step bound and never loops forever.

---

### User Story 2 - Route cyclically through nodes with accumulating state (Priority: P2)

The Supervisor acts as a routing hub: it inspects the current state and the intent, dispatches to the Local-LLM node or the Tool-Execution node, and those nodes return control to the Supervisor, which may route again or finish. Across these steps the message history and usage totals grow, and a bound guarantees the loop always ends.

**Why this priority**: Cyclic routing with a coordinator is what makes the engine an orchestrator rather than a single function call, enabling multi-step work. It builds on US1 and is independently testable, but the engine delivers baseline value (US1) even with a single hop.

**Independent Test**: Drive an intent that causes multiple Supervisor visits; confirm the message history and usage accumulate one contribution per node step, control returns to the Supervisor between node executions, and the run still terminates.

**Acceptance Scenarios**:

1. **Given** a run that visits a node, **When** that node finishes, **Then** control returns to the Supervisor, which decides the next route or to finish.
2. **Given** a multi-step run, **When** each node executes, **Then** the message history gains an ordered entry and the usage tracker increases, with no loss or reordering.
3. **Given** the Supervisor would otherwise keep routing, **When** the step bound is reached, **Then** the run terminates with a clear terminal status indicating the bound stopped it.
4. **Given** the Supervisor decides to finish immediately (no node work needed), **When** the run ends, **Then** a valid outcome is still returned with empty/zero accumulated state.

---

### User Story 3 - Gateway returns the orchestration outcome and usage (Priority: P2)

When the gateway accepts a valid, sufficiently-confident intent, it triggers the orchestration engine with that intent and returns the engine's outcome together with the accumulated usage — instead of only a static acknowledgment. The usage totals populate the response's usage field, which was previously reserved but empty.

**Why this priority**: This is the user-visible payoff — submitting an intent now actually drives orchestration and returns a real result and usage accounting. It depends on the engine (US1) existing.

**Independent Test**: Submit an accepted intent to the gateway and confirm the response contains the orchestration outcome and a populated usage field; submit a rejected/invalid intent and confirm the engine is not triggered.

**Acceptance Scenarios**:

1. **Given** a valid intent whose confidence meets the threshold, **When** it is submitted to the gateway, **Then** the gateway triggers the engine and the response includes the orchestration outcome and the accumulated usage.
2. **Given** the same accepted submission, **When** the response is inspected, **Then** the usage field is populated (no longer null/empty) and reflects the run's totals.
3. **Given** an intent that is rejected (below threshold) or invalid (contract violation), **When** it is submitted, **Then** the orchestration engine is NOT triggered and the existing rejection behavior is unchanged.

---

### Edge Cases

- **Immediate finish**: the Supervisor decides to finish before dispatching any node — a valid outcome is returned with empty message history and zero usage.
- **Step-bound reached**: a run that would keep routing halts at the bound with a terminal status that identifies the bound as the stop reason (never an infinite loop).
- **Intent with no specific route**: the Supervisor applies a defined default (e.g., finish, or a default node) rather than failing or hanging.
- **No usage produced**: if no node does billable work, the usage tracker stays at zero and the outcome/usage field is still present.
- **Node raises during orchestration**: the run ends with a structured terminal error status; the gateway surfaces it within its existing response envelope rather than hanging or crashing.
- **Rejected/invalid intents**: never reach the engine — orchestration is only triggered after acceptance (unchanged from the gateway feature).
- **Determinism**: given the same intent, the stubbed engine produces the same outcome and usage (reproducible), since nodes are deterministic placeholders.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The engine MUST maintain an orchestration state that carries the originating intent, an ordered message history, and a usage/token tracker, threaded through every step of a run.
- **FR-002**: The engine MUST begin every run at a single, well-defined entry point (the Supervisor).
- **FR-003**: The Supervisor MUST inspect the intent and current state and decide the next action: route to a specific node, or finish the run.
- **FR-004**: The engine MUST provide a Local-LLM node representing local model inference, which in this feature produces deterministic placeholder output (no real model call), appends an entry to the message history, and updates the usage tracker.
- **FR-005**: The engine MUST provide a Tool-Execution node representing an external tool/action, which in this feature produces deterministic placeholder output (no real tool call), appends an entry to the message history, and updates the usage tracker.
- **FR-006**: Non-Supervisor nodes MUST return control to the Supervisor after executing, making the graph cyclic.
- **FR-007**: The engine MUST guarantee termination: every run halts, enforced by a bounded step/iteration limit in addition to the Supervisor's ability to finish, so cycles cannot run forever.
- **FR-008**: The message history MUST be append-only within a run and preserve the order and node provenance of each step.
- **FR-009**: The usage tracker MUST accumulate additively across steps such that its totals equal the sum of per-step contributions, with no loss.
- **FR-010**: On termination, the engine MUST return a structured outcome that includes a terminal status, the nodes that executed, the resulting message history, and the accumulated usage.
- **FR-011**: The gateway MUST, upon accepting a valid intent that meets the confidence threshold, trigger the engine with that intent and include the orchestration outcome and accumulated usage in its response.
- **FR-012**: The gateway MUST populate the previously-reserved usage field of its response with the run's accumulated usage totals for accepted intents.
- **FR-013**: The gateway MUST NOT trigger the engine for intents that fail validation or fall below the confidence threshold; existing rejection behavior is unchanged.
- **FR-014**: The engine's node bodies MUST be stubbed/deterministic in this feature; the same intent MUST yield the same outcome and usage.

### Key Entities *(include if feature involves data)*

- **Orchestration State**: The state threaded through a run. Attributes: the originating intent (the established intent contract), an ordered message history, a usage/token tracker, and control fields sufficient to route and to enforce the step bound.
- **Message**: An ordered entry appended to the history as a node executes. Attributes: source/role (which node produced it), content, and its position/step in the run.
- **Usage Tracker**: The accumulator of token counts / cost across steps. Monotonically increases; its totals are the sum of per-step contributions and are intended to flow into the gateway response's usage field.
- **Node**: A unit of work in the graph. Three kinds: Supervisor (routing hub / entry point / decides finish), Local-LLM (stubbed inference), Tool-Execution (stubbed tool/action).
- **Orchestration Outcome**: The structured result returned on termination. Attributes: terminal status (e.g., completed / halted-at-bound / error), nodes executed, resulting message history, and accumulated usage.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of accepted intents produce a terminating orchestration outcome with a terminal status; no run loops indefinitely.
- **SC-002**: No orchestration run exceeds the configured step bound; runs that would otherwise continue halt at the bound with a clear terminal status.
- **SC-003**: The message history contains exactly one ordered entry per node execution, in execution order, with correct node provenance.
- **SC-004**: The usage tracker's final totals equal the sum of the per-step contributions in 100% of runs (no double-counting, no loss).
- **SC-005**: For an accepted intent, the gateway response includes the orchestration outcome and a populated usage field (previously null/empty) in 100% of cases.
- **SC-006**: For rejected or invalid intents, the engine is triggered 0% of the time and the response matches the pre-existing rejection behavior.
- **SC-007**: Given the same intent submitted twice, the engine returns identical outcome and usage (deterministic/reproducible) in 100% of cases.

## Assumptions

- The orchestration engine is a stateful, cyclic graph (the chosen engine technology is an implementation detail resolved in planning). It builds on the `IntentPayload` model (feature 001) and the gateway (feature 002), whose response envelope already reserves the `usage` field.
- The Supervisor's routing and finish decisions are **deterministic** for this feature (derived from the intent and a simple rule), so runs are reproducible and testable. Intelligent/model-driven routing is out of scope.
- A **default step bound** (a small fixed integer) guarantees termination and is configurable later; its exact value is a planning detail. The bound is a safety net in addition to the Supervisor's normal finish decision.
- Node outputs are deterministic placeholders and each node contributes a fixed, known usage increment, so accumulation and reproducibility are testable.
- Orchestration executes **synchronously** within the gateway request for this MVP; the gateway blocks until the run terminates. Async execution, streaming, and background processing are out of scope.
- A message is a simple ordered entry with a source/role and content; rich message typing is out of scope for this feature.
- Real local-model inference, real tool/action integrations, cross-request persistence/checkpointing, human-in-the-loop interrupts, and concurrency/throughput tuning are out of scope.
- Only accepted intents reach the engine; validation and threshold behavior from the gateway feature are unchanged.
