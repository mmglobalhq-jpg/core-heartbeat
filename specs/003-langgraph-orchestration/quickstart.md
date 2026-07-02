# Quickstart: Orchestration Engine

Validation guide proving the orchestration engine (`orchestrator.py`) and its gateway integration satisfy the spec. Run from the repository root.

## Prerequisites

- Python 3.14 venv (`./venv`).
- Runtime deps incl. the LangGraph stack:

```bash
./venv/bin/pip install -r requirements.txt
```

- Dev deps (pytest + test client):

```bash
./venv/bin/pip install -r requirements-dev.txt
```

## Artifacts this validates

- State & models: [data-model.md](./data-model.md)
- Engine & integration contract: [contracts/graph.md](./contracts/graph.md)
- Requirements: [spec.md](./spec.md) (FR-001…FR-014, SC-001…SC-007)

## Scenario 1 — A run terminates with an outcome (US1, SC-001)

Call `run(payload)` with an accepted intent (e.g. identity `"greet"`).

- **Expected**: returns an `OrchestrationOutcome` with a terminal `status` (`"completed"`); the call halts (no infinite loop).

## Scenario 2 — Cyclic multi-step with accumulation (US2, SC-003/004)

Run the `"greet"` intent (plan: local_llm then tool_execution).

- **Expected**: `nodes_executed == ["local_llm", "tool_execution"]`; `messages` has one ordered entry per node step with correct `source`; `usage.total_tokens == 35` (10+20+30 → then 5 → summed).

## Scenario 3 — Immediate finish (edge case)

Run a no-op intent (identity `"ping"` or `"noop"`).

- **Expected**: `status="completed"`, `nodes_executed == []`, `usage.total_tokens == 0`, minimal/no worker messages.

## Scenario 4 — Bounded termination (SC-002)

Confirm no run exceeds `MAX_STEPS`; a run that would keep routing halts at the bound.

- **Expected**: `steps <= MAX_STEPS`; if the bound stops it, `status="halted_step_bound"`. The hard `recursion_limit` never triggers for the deterministic plan.

## Scenario 5 — Determinism (SC-007)

Call `run(payload)` twice with the same intent.

- **Expected**: identical `OrchestrationOutcome` (same status, nodes, messages, usage) both times.

## Scenario 6 — Gateway returns outcome + populated usage (US3, SC-005)

Submit an accepted intent to `POST /intent`.

- **Expected**: HTTP 200; body `outcome="accepted"`, includes an `orchestration` object, and `usage` is a **populated dict** (no longer null) equal to the run's totals.

## Scenario 7 — Engine not triggered for rejections (SC-006)

Submit a below-threshold intent and an invalid intent.

- **Expected**: threshold_rejected / validation_rejected exactly as before; `usage` stays `null`; no `orchestration` field with run data.

## Run the automated suite

```bash
./venv/bin/python -m pytest tests/ -v
```

- **Expected**: all suites green — `test_orchestrator.py` (engine), updated `test_gateway_endpoints.py` (accepted now carries orchestration + usage; rejections unchanged), plus the unchanged 001/002 logic tests.

## Manual smoke check (optional)

```bash
./venv/bin/uvicorn main:app --port 8000 &
curl -s -X POST localhost:8000/intent -H 'content-type: application/json' \
  -d '{"intent":"greet","confidence":0.9,"raw_input":"hi","source":"cli"}' | python -m json.tool
```

- **Expected**: an `accepted` envelope whose `usage` is populated (e.g. `total_tokens: 35`) and whose `orchestration.nodes_executed` is `["local_llm","tool_execution"]`.
