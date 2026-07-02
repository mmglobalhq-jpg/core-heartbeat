# Implementation Plan: API-Driven Supervisor Node

**Branch**: `004-api-supervisor` | **Date**: 2026-07-01 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/004-api-supervisor/spec.md`

## Summary

Replace the deterministic Supervisor node in `orchestrator.py` with a live call to **Gemini 2.5 Flash** via the official **Google GenAI SDK** (`google-genai`). The Supervisor builds a routing context from the intent + message history, calls the model with a **strict structured-output schema** (`RoutingDecision.next_node ∈ {local_llm, tool_execution, finish}`), and routes on the validated decision. A module-level `get_client()` reads `GEMINI_API_KEY`; the model call is isolated in `request_routing_decision(state, client)` so tests substitute a **fake client (no network)**. All failure modes — missing key, auth, timeout, network, invalid/out-of-vocab output — are caught and resolved to a **safe `finish`** with a recorded, categorized `RoutingFailure` (status `degraded`), so the graph always terminates. `MAX_STEPS`/`recursion_limit` remain. Model token usage (`usage_metadata`) is added to the run's `TokenUsage`. `local_llm`/`tool_execution` stay stubbed.

## Technical Context

**Language/Version**: Python 3.14.4 (venv)

**Primary Dependencies**: **NEW** google-genai 2.10.0 (+ google-auth 2.55.1, cryptography 49.0.0, cffi, pyasn1, pyasn1-modules, pycparser) — verified installing and importing on 3.14, structured-output + error API confirmed (research R1). **httpx/httpcore move to runtime** (google-genai requires httpx). Existing: langgraph 1.2.7, FastAPI, Pydantic.

**Storage**: N/A.

**Testing**: pytest + FastAPI TestClient. Model calls are exercised via an injected fake client — **no real network calls, no API spend** in tests (FR-012, SC-007).

**Target Platform**: Linux server (WSL2 dev); ASGI app via uvicorn. Now makes outbound HTTPS calls to the Gemini API at runtime.

**Project Type**: Single project — orchestration module gains a live external dependency in one node.

**Performance Goals**: Bounded per-call latency via a request timeout (FR-006/SC-004); number of model calls per run capped by `MAX_STEPS`.

**Constraints**: Strict decision vocabulary (FR-002/SC-001); always terminate even on failure (FR-005/FR-007/SC-002); degraded runs still return a complete outcome (SC-003); failures recorded/observable (FR-008/SC-005); credential from env, never logged (FR-003); no real network in tests (FR-012).

**Scale/Scope**: Rewrite of one node + a model-call helper + a client factory in `orchestrator.py`; ~2 new models in `models.py`; requirements split update; test additions + updates to existing orchestrator/gateway tests (supervisor now needs a client).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Constitution is the unpopulated template — no ratified principles, no enforceable gates. **Gate status: PASS (vacuously).**

Applied defaults (consistent with 001–003): simplicity/YAGNI (only the Supervisor becomes live; nodes stay stubbed; no retries/streaming/fallback), test-first (each failure mode + valid decision has a deterministic test via fake client), fail-safe (external dependency degrades to a safe terminal decision).

*Post-Phase 1 re-check*: Adds one runtime dependency stack (google-genai) — justified as the feature's premise. No new architectural surface beyond the Supervisor rewrite and two models. **PASS.**

## Project Structure

### Documentation (this feature)

```text
specs/004-api-supervisor/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   └── supervisor.md     # Phase 1 — model-call contract, decision schema, failure taxonomy
├── checklists/
│   └── requirements.md  # from /speckit-specify
└── tasks.md             # Phase 2 (/speckit-tasks — NOT created here)
```

### Source Code (repository root)

```text
orchestrator.py  # MODIFIED: get_client() (reads GEMINI_API_KEY, memoized, None if missing);
                 #   request_routing_decision(state, client) -> (RoutingDecision|None, RoutingFailure|None, TokenUsage);
                 #   supervisor() rewritten to call the model, map failures to safe finish (status "degraded"),
                 #   record the failure, capture usage. MAX_STEPS guard kept (no model call past the bound).
                 #   local_llm/tool_execution UNCHANGED (stubs). Constants: MODEL_NAME="gemini-2.5-flash",
                 #   REQUEST_TIMEOUT_MS, GEMINI_API_KEY_ENV.
models.py        # + RoutingDecision (next_node: Literal[...]) used as response_schema;
                 #   + RoutingFailure (category: Literal[...], detail). OrchestrationOutcome.status gains "degraded".
requirements.txt # + google-genai stack; httpx/httpcore moved here (runtime).
requirements-dev.txt # httpx/httpcore removed (now runtime); keeps pytest stack.

tests/
├── test_supervisor.py         # NEW — request_routing_decision + supervisor via fake client:
│                              #   each valid decision, each failure mode, invalid output, usage capture
├── test_orchestrator.py       # UPDATED — full runs now inject a fake client (monkeypatch get_client);
│                              #   greet-plan trace reproduced; noop reframed as model-returns-finish
├── test_gateway_endpoints.py  # UPDATED — integration tests inject a fake client for accepted runs
└── (001/002 logic + payload unchanged)
```

**Structure Decision**: The model call is isolated behind `get_client()` and `request_routing_decision(state, client)`, both module-level in `orchestrator.py`. The `supervisor` node calls `get_client()` **dynamically at invoke time** (not captured at graph-build time), so tests can `monkeypatch orchestrator.get_client` to a fake even though `graph` is compiled once at import. This keeps the compiled-once graph while making the whole stack (unit + endpoint) testable with zero network calls.

## Complexity Tracking

> No constitution violations to justify. The google-genai dependency and outbound API call are the feature's explicit premise.
