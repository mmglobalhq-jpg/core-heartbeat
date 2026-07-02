# Quickstart: API-Driven Supervisor Node

Validation guide proving the model-driven Supervisor and its failure handling satisfy the spec. Run from the repository root. **Tests use a fake client — no real API calls, no key needed, no spend.**

## Prerequisites

- Python 3.14 venv (`./venv`).
- Runtime deps incl. the Google GenAI stack:

```bash
./venv/bin/pip install -r requirements.txt
```

- Dev deps (pytest):

```bash
./venv/bin/pip install -r requirements-dev.txt
```

- For a **live** smoke check only: export a real key `GEMINI_API_KEY=...` (optional; the automated suite never needs it).

## Artifacts this validates

- Models & flow: [data-model.md](./data-model.md)
- Supervisor & failure contract: [contracts/supervisor.md](./contracts/supervisor.md)
- Requirements: [spec.md](./spec.md) (FR-001…FR-012, SC-001…SC-007)

## Scenario 1 — Valid decisions route correctly (US1, SC-001)

Inject a fake client returning each valid decision.

- **Expected**: `local_llm` → routes to the local model node; `tool_execution` → routes to the tool node; `finish` → run terminates with `status="completed"`.

## Scenario 2 — Out-of-vocabulary / unparseable output is rejected (US1.4, SC-001)

Fake client returns `{"next_node": "banana"}` or non-JSON.

- **Expected**: not accepted as a decision → `invalid_output` failure → safe `finish`, `status="degraded"`, failure recorded.

## Scenario 3 — Each infrastructure failure degrades safely (US2, SC-002/003/005)

Fake client raising, in turn: missing key (`get_client()` → None), auth error (401), timeout, network error.

- **Expected**: each resolves to `finish`, `status="degraded"`, a recorded category (`missing_credential` / `auth` / `timeout` / `network`); the run terminates and returns a complete outcome — no crash, no hang.

## Scenario 4 — Bounded wait (SC-004)

Confirm the model call is configured with a request timeout so a slow model cannot hang the run (timeout maps to a `timeout` degraded finish).

## Scenario 5 — Usage capture (US3, SC-006)

Fake client reporting known `usage_metadata` token counts; and one reporting none.

- **Expected**: reported tokens are added to the run's usage totals; when none are reported, totals are unchanged and no error occurs.

## Scenario 6 — Greet-plan reproducibility with a scripted fake (regression)

Inject a fake scripted `["local_llm", "tool_execution", "finish"]`.

- **Expected**: `nodes_executed == ["local_llm", "tool_execution"]`, stub `total_tokens` accumulates as before (plus any fake model usage); trace reproduced deterministically.

## Scenario 7 — Gateway integration under degradation (SC-003)

Submit an accepted intent to `POST /intent` with no `GEMINI_API_KEY` set.

- **Expected**: HTTP 200 with an orchestration outcome whose `status="degraded"` (Supervisor could not route), `nodes_executed == []` — the caller still gets a well-formed response, never an unhandled error.

## Run the automated suite

```bash
./venv/bin/python -m pytest tests/ -v
```

- **Expected**: all suites green — `test_supervisor.py` (decisions, failures, usage), updated `test_orchestrator.py` (fake-client runs), updated `test_gateway_endpoints.py`, and the unchanged 001/002 tests. No network access occurs.

## Live smoke check (optional, requires a real key + spend)

```bash
GEMINI_API_KEY=... ./venv/bin/uvicorn main:app --port 8000 &
curl -s -X POST localhost:8000/intent -H 'content-type: application/json' \
  -d '{"intent":"greet","confidence":0.9,"raw_input":"hi","source":"cli"}' | python -m json.tool
```

- **Expected**: an `accepted` envelope; the Supervisor's real routing decisions drive the run; `orchestration.status` is `completed` (or `degraded` if the live call fails), and `usage` reflects real model tokens when reported.
