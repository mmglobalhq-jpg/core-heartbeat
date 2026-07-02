<!--
SYNC IMPACT REPORT
==================
Version change: (template, unratified) → 1.0.0
Bump rationale: Initial ratification. The template placeholders are replaced with
the project's first concrete governing principles; per semantic-versioning policy
the first adopted constitution is 1.0.0.

Modified principles (placeholder → adopted):
  - [PRINCIPLE_1] → I. Quality-Prioritized Orchestration
  - [PRINCIPLE_2] → II. Purpose-Driven Resource Allocation
  - [PRINCIPLE_3] → III. Adaptive Cost-Awareness
  - [PRINCIPLE_4] → IV. Fail-Safe Transparency
  - [PRINCIPLE_5] → (removed; project adopts 4 principles, not 5)

Added sections:
  - Operational Constraints (replaces [SECTION_2])
  - Development Workflow & Quality Gates (replaces [SECTION_3])
  - Governance (concretized)

Removed sections:
  - Fifth principle slot (template offered up to 5; only 4 adopted)

Templates requiring updates:
  - ✅ .specify/templates/plan-template.md — Constitution Check gate is generic
        ("[Gates determined based on constitution file]"); resolves against this
        file dynamically, no edit required.
  - ✅ .specify/templates/spec-template.md — no principle-specific constraints to
        propagate; unchanged.
  - ✅ .specify/templates/tasks-template.md — task categories already cover the
        observability/testing discipline these principles imply; unchanged.

Follow-up TODOs: none. Ratification date set to the repository's first commit
(2026-07-01); if a different formal adoption date is preferred, amend and bump PATCH.
-->

# core-heartbeat Constitution

## Core Principles

### I. Quality-Prioritized Orchestration

The Supervisor MUST select the model or tool best suited to the task's complexity.
For high-reasoning tasks — including complex code generation and architectural
analysis — the system MUST use high-capability API-based models to ensure
correctness, regardless of the local-versus-cloud distinction. Capability, not
locality, is the deciding factor when correctness is at stake.

**Rationale**: Routing a hard-reasoning task to an under-powered model to save a
few tokens trades a small, certain cost for a large, probable one. Correctness on
high-value work is non-negotiable, so the routing decision optimizes for fitness
to the task first.

### II. Purpose-Driven Resource Allocation

The system MUST employ local inference for routine automation, repetitive
monitoring, and structured data tasks where local capability is proven sufficient.
API inference is the standard for tasks where quality, nuance, or reasoning depth
is the primary constraint. "Proven sufficient" means demonstrated adequacy for
that class of task, not an assumption.

**Rationale**: Local and API inference are complementary tools with distinct
strengths. Matching each task class to the tier that reliably handles it keeps
the cheap path cheap and the demanding path correct.

### III. Adaptive Cost-Awareness

Cost-awareness is a policy of efficiency, NOT a barrier to quality. The system
MUST prioritize cost savings on low-value tasks — heartbeats, routine polling —
while authorizing appropriate spend for high-value tasks — coding, complex
planning — where the cost of failure exceeds the cost of tokens. Cost MUST NOT be
used to justify a wrong answer on work where correctness is the primary
constraint.

**Rationale**: The metric that matters is total cost of outcomes, not token spend
in isolation. Spending tokens to avoid an expensive failure is a saving, not an
expense; starving a high-value task to conserve tokens is the false economy this
principle forbids.

### IV. Fail-Safe Transparency

The system MUST NEVER fail silently. Every degradation — whether from network
latency, model timeout, authentication failure, missing credential, or invalid
model output — MUST be explicitly categorized in the `OrchestrationOutcome` and
returned to the gateway. A run MUST always terminate with an observable outcome;
a swallowed or uncategorized failure is a defect.

**Rationale**: An orchestration engine that hides failures is untrustworthy and
undebuggable. Explicit, categorized degradation is what lets callers reason about
what happened and lets operators find the cause. This principle is the contract
behind the failure taxonomy and the guaranteed-termination design.

## Operational Constraints

- **Guaranteed termination**: Every orchestration run MUST terminate. The
  three-layer defense (Supervisor finish/degrade decision, the `MAX_STEPS` step
  bound checked before any model call, and LangGraph's `recursion_limit`) MUST
  remain intact. Removing a layer requires a constitution amendment.
- **Credential safety**: API credentials MUST only be read from the environment
  and passed to the SDK. They MUST NEVER be logged, embedded in a recorded
  message, or included in any failure `detail`.
- **Bootable without credentials**: A missing API key MUST degrade to a
  categorized `missing_credential` outcome, never a crash. The service MUST stay
  bootable and the gateway responsive without any model credential configured.
- **Bounded model calls**: Every model call MUST be time-bounded so a hung
  provider degrades (as `timeout`) rather than blocking a run.

## Development Workflow & Quality Gates

- **Spec-driven**: Features are developed through the Spec Kit flow
  (specify → plan → tasks → implement). Design artifacts live under `specs/`.
- **Deterministic, network-free tests**: The test suite MUST pass with no real
  network access and no API spend. Model interactions MUST be exercised through
  an injected fake client. CI MUST require no API credential.
- **Green before commit**: The full suite MUST be green before a feature is
  considered complete. Each Spec Kit phase checkpoint SHOULD be independently
  testable.
- **Constitution Check gate**: Every `plan.md` MUST pass the Constitution Check
  against these principles before Phase 0 research and again after Phase 1
  design. Violations MUST be justified in the plan's Complexity Tracking or the
  design MUST change.

## Governance

This constitution supersedes other development practices where they conflict.

- **Amendments** MUST be made by editing `.specify/memory/constitution.md`,
  accompanied by an updated Sync Impact Report and a version bump per the policy
  below. Dependent templates and guidance MUST be re-checked for consistency in
  the same change.
- **Versioning policy** (semantic):
  - **MAJOR**: Backward-incompatible governance changes — a principle removed or
    redefined in a way that invalidates existing compliance.
  - **MINOR**: A new principle or section added, or material expansion of
    guidance.
  - **PATCH**: Clarifications, wording, and non-semantic refinements.
- **Compliance review**: Plans and reviews MUST verify adherence to these
  principles. Any complexity or deviation MUST be explicitly justified against the
  principle it strains; unjustified deviations block the change.

**Version**: 1.0.0 | **Ratified**: 2026-07-01 | **Last Amended**: 2026-07-02
