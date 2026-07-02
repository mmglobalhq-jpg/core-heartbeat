# Implementation Plan: Local Ollama Worker Node

**Branch**: `005-local-ollama-worker` | **Date**: 2026-07-02 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/005-local-ollama-worker/spec.md`

## Summary

Replace the stubbed `local_llm` node in `orchestrator.py` with a live,
**asynchronous** call to the local **Ollama** service (`POST` `OLLAMA_URL`,
default `http://localhost:11434/api/generate`) invoking **`qwen2.5:7b`**
(non-streaming). The call is isolated in an async helper
`generate_local(state, client) -> (text | None, WorkerFailure | None, TokenUsage)`
that takes an injected `httpx.AsyncClient`; a `build_ollama_client()` factory is
the test seam (tests inject `httpx.MockTransport` — **no network, no daemon**).
Ollama's `prompt_eval_count`/`eval_count` map to `TokenUsage` and sum into the
run via the existing `add_usage` reducer (usage stays observable). Every failure —
unreachable, timeout, non-2xx, unusable body — becomes a categorized
`WorkerFailure` recorded in the run's message history; the node returns control to
the Supervisor and the run always terminates (three-layer guard intact). Making
the node async ripples the invocation to `graph.ainvoke` in an `async def run()`
and an `async def submit_intent` in the gateway; direct `run()` tests use
`asyncio.run()` (no `pytest-asyncio`). `tool_execution` stays a stub; the
Supervisor (feature 004) is unchanged.

## Technical Context

**Language/Version**: Python 3.14.4 (venv)

**Primary Dependencies**: **No new packages.** `httpx 0.28.1` is already a runtime
dep (pulled in by google-genai in feature 004) and provides both
`AsyncClient` and `MockTransport`. Existing: langgraph 1.2.7, FastAPI 0.139,
Pydantic 2.13, google-genai 2.10 (Supervisor, unchanged).

**Storage**: N/A.

**Testing**: pytest + FastAPI `TestClient`. The Ollama call is exercised via an
injected `httpx.MockTransport` — **no real network, no daemon** (FR-011, SC-004).
Direct `run()` tests wrap the async call in `asyncio.run()`.

**Target Platform**: Linux server (WSL2 dev); ASGI app via uvicorn. Adds an
outbound HTTP call to a **local** service at runtime.

**Project Type**: Single project — one orchestration node becomes live + async.

**Performance Goals**: Per-call latency bounded by `OLLAMA_TIMEOUT_MS`
(default 120 s); calls per run capped by `MAX_STEPS`; run terminates predictably
(SC-005).

**Constraints**: Real model text replaces the stub (FR-001/SC-001); usage summed
field-wise and observable (FR-005/SC-002); all failures degrade to a recorded,
categorized, terminating outcome (FR-008/FR-009/SC-003); bounded call, no hang
(FR-007/SC-005); no network/daemon in tests (FR-011/SC-004); three-layer
termination preserved (FR-012).

**Scale/Scope**: Rewrite of one node (async) + a model-call helper + a client
factory in `orchestrator.py`; `run()`→async; 1 new model in `models.py`
(`WorkerFailure`); gateway endpoint → async; new `tests/test_local_worker.py` +
updates to existing orchestrator/gateway tests (stub usage assertions change).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Evaluated against the ratified **core-heartbeat Constitution v1.0.0**:

- **I. Quality-Prioritized Orchestration** — PASS. The local worker path becomes a
  real capability; routing authority stays with the Supervisor. This feature does
  not downgrade any high-reasoning path.
- **II. Purpose-Driven Resource Allocation** — PASS / directly advances. Local
  inference for a worker node is exactly the "local for routine/structured tasks"
  intent; this is the first realization of that principle.
- **III. Adaptive Cost-Awareness** — PASS. Local inference is the low-cost path;
  no cloud spend added. Usage remains tracked so cost stays visible.
- **IV. Fail-Safe Transparency** — PASS / central. Every degradation
  (unreachable/timeout/invalid_output) is categorized in the run's outcome
  (message history) and returned to the gateway; no silent or fatal failure.
- **Operational Constraints** — PASS. Three-layer termination preserved (no new
  unbounded path); bounded call via `OLLAMA_TIMEOUT_MS`; bootable without the
  daemon (a missing daemon degrades to `unreachable`); no credential (Ollama is
  local/keyless) so the credential-safety rule is N/A.
- **Dev Workflow & Quality Gates** — PASS. Network-free deterministic tests via
  `MockTransport`; spec-driven flow; green-before-commit; this plan passes the
  Constitution Check gate.

**Gate status: PASS.** No violations → Complexity Tracking empty.

*Post-Phase 1 re-check*: Design adds one model (`WorkerFailure`), one async helper,
one client factory, and flips the invocation to `ainvoke`/async endpoint. No new
dependency, no new architectural surface beyond the node rewrite. Still **PASS**.

## Project Structure

### Documentation (this feature)

```text
specs/005-local-ollama-worker/
├── plan.md              # This file
├── research.md          # Phase 0 output (R1–R7)
├── data-model.md        # Phase 1 output (WorkerFailure + node output contract)
├── quickstart.md        # Phase 1 output (validation guide)
├── contracts/
│   └── local_worker.md  # Phase 1 — generate_local contract, failure map, test matrix
├── checklists/
│   └── requirements.md  # from /speckit-specify
└── tasks.md             # Phase 2 (/speckit-tasks — NOT created here)
```

### Source Code (repository root)

```text
orchestrator.py  # MODIFIED:
                 #   + constants OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_MS (env-read at invoke time)
                 #   + build_ollama_client() -> httpx.AsyncClient   (test seam)
                 #   + async generate_local(state, client) -> (text|None, WorkerFailure|None, TokenUsage)
                 #       (never raises; failure map per contract; usage from prompt_eval_count/eval_count)
                 #   * local_llm  -> async def: calls build_ollama_client() + generate_local;
                 #       records real text or categorized failure; adds usage; appends "local_llm" to visited
                 #   * run()      -> async def: await graph.ainvoke(...); outcome mapping unchanged
                 #   UNCHANGED: supervisor (feature 004), tool_execution (stub), route/build_graph, MAX_STEPS/RECURSION_LIMIT
models.py        # + WorkerFailure(category: Literal["unreachable","timeout","invalid_output"], detail: str)
router.py        # * submit_intent -> async def; await run_orchestration(payload); rest unchanged
main.py          # UNCHANGED
requirements*.txt# UNCHANGED (httpx already runtime)

tests/
├── test_local_worker.py       # NEW — generate_local + local_llm via httpx.MockTransport:
│                              #   success text+usage, empty/no-counts, missing field, non-JSON,
│                              #   404/500, ConnectError->unreachable, ReadTimeout->timeout
├── test_orchestrator.py       # UPDATED — full runs now also inject a MockTransport Ollama client;
│                              #   greet-plan trace: local_llm carries mocked text; usage totals updated
│                              #   for real (mocked) counts; run() calls wrapped in asyncio.run()
├── test_gateway_endpoints.py  # UPDATED — accepted-run tests inject the mock Ollama client;
│                              #   assert local_llm text + updated usage totals; async endpoint via TestClient
├── test_supervisor.py         # UNCHANGED (Supervisor untouched)
└── (001/002 payload + logic unchanged)
```

**Structure Decision**: Mirror feature 004's isolation pattern. The HTTP call is
behind `generate_local(state, client)` with `build_ollama_client()` as the
factory; the `local_llm` node calls the factory **at invoke time** (not captured
at graph-build), so tests `monkeypatch orchestrator.build_ollama_client` to a
`MockTransport`-backed client even though `graph` is compiled once at import. The
only structural shift is sync→async: `local_llm` and `run()` become async and the
graph is driven with `ainvoke`; the sync `supervisor`/`tool_execution` nodes run
unchanged under `ainvoke` (verified). This keeps compile-once + zero-network
tests while delivering a genuinely asynchronous local call.

## Complexity Tracking

> No constitution violations to justify. No new dependency (httpx is already
> runtime). The async invocation and the local HTTP call are the feature's
> explicit premise; the `WorkerFailure` model is the minimum needed to satisfy
> Fail-Safe Transparency.
