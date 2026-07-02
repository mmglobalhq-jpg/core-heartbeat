# Quickstart: Gateway Routing Interface

Validation guide proving the gateway (`main.py` + `router.py`) satisfies the spec. Run from the repository root.

## Prerequisites

- Python 3.14 venv (`./venv`), with FastAPI/Pydantic/uvicorn already installed.
- Dev deps: pytest (present) plus an httpx-compatible test client (installed by the setup task):

```bash
./venv/bin/pip install -r requirements-dev.txt
```

## Artifacts this validates

- Data model & response envelope: [data-model.md](./data-model.md)
- Endpoint contract: [contracts/openapi.yaml](./contracts/openapi.yaml)
- Requirements: [spec.md](./spec.md) (FR-001…FR-014, SC-001…SC-008)

## Config

The acceptance threshold is read from `HEARTBEAT_CONFIDENCE_THRESHOLD` at startup (default `0.5`, inclusive `>=`).

## Scenario 1 — Accept a confident intent (US1, SC-001)

Submit a valid `IntentPayload` with `confidence >= threshold` to `POST /intent`.

- **Expected**: HTTP 200; body `outcome="accepted"`, `accepted=true`, echoes `intent`, and includes the `usage` field (null/empty).

## Scenario 2 — Accept exactly at the threshold (US1 scenario 2)

Submit with `confidence` equal to the configured threshold.

- **Expected**: HTTP 200, `outcome="accepted"` (inclusive boundary).

## Scenario 3 — Reject below threshold (US2, SC-002)

Submit a valid intent with `confidence` below the threshold.

- **Expected**: HTTP 422; `outcome="threshold_rejected"`, body reports both the submitted `confidence` and the required `threshold`; never reported as accepted.

## Scenario 4 — Reject invalid submissions (US3, SC-003)

Submit payloads that each violate the contract: missing `intent`, `confidence` out of `[0,1]`, an unknown extra field, wrong type.

- **Expected**: HTTP 422; `outcome="validation_rejected"`, `errors` identifies the problem; no threshold comparison occurs (an out-of-range confidence is a validation error, not a threshold rejection).

## Scenario 5 — Outcomes are distinguishable (SC-004)

Compare the three responses above.

- **Expected**: each is identifiable from the response body alone via the `outcome` field; all three carry the shared envelope including `usage`.

## Scenario 6 — Threshold is env-driven (US2 scenario 3, SC-005)

Build the app under two different `HEARTBEAT_CONFIDENCE_THRESHOLD` values and submit the same mid-range intent.

- **Expected**: opposite decisions (accepted under the low threshold, threshold_rejected under the high one) with no code change.

## Scenario 7 — Config safety (edge cases)

Start with the variable unset/blank (→ default `0.5`), and with an out-of-range/unparseable value.

- **Expected**: unset/blank uses `0.5`; out-of-range or unparseable raises a clear configuration error at startup naming the variable and value.

## Scenario 8 — Liveness (US4, SC-008)

Call `GET /health` with no body.

- **Expected**: HTTP 200, `status="online"`, `service="core-heartbeat"`; no effect on intent processing.

## Run the automated suite

```bash
./venv/bin/python -m pytest tests/test_gateway_logic.py tests/test_gateway_endpoints.py -v
```

- **Expected**: all scenarios pass. Logic tests (threshold decision, config loader) run without the HTTP client; endpoint tests drive `/intent` and `/health` in-process.

## Manual smoke check (optional)

```bash
# Start the gateway
./venv/bin/uvicorn main:app --port 8000 &
# Accepted:
curl -s -X POST localhost:8000/intent -H 'content-type: application/json' \
  -d '{"intent":"ping","confidence":0.9,"raw_input":"ping","source":"cli"}'
# Health:
curl -s localhost:8000/health
```

- **Expected**: first prints an `accepted` envelope; second prints `{"status":"online","service":"core-heartbeat"}`.
