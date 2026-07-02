# Quickstart: Intent Payload

Validation guide proving `IntentPayload` (in `models.py`) satisfies the spec. Run from the repository root.

## Prerequisites

- Python 3.14 venv (already present at `./venv`)
- Pydantic 2.13.x, FastAPI 0.139.x (already installed)
- pytest (dev dependency — install if not present):

```bash
./venv/bin/pip install pytest
```

## Artifacts this validates

- Data model: [data-model.md](./data-model.md)
- Field/constraint contract: [contracts/intent_payload.schema.json](./contracts/intent_payload.schema.json)
- Requirements: [spec.md](./spec.md) (FR-001…FR-011, SC-001…SC-005)

## Scenario 1 — Construct a valid payload (US1, SC-001)

Create an `IntentPayload` with a recognized identity and parameters; confirm all fields are populated and `entities` are intact.

- **Expected**: instance constructs without error; `intent`, `confidence`, `entities`, `raw_input`, `timestamp`, `source` all readable; `timestamp` is timezone-aware UTC when not supplied.

## Scenario 2 — Missing identity is rejected (FR-002, Edge case)

Attempt to construct without `intent` (or with an empty string).

- **Expected**: `pydantic.ValidationError` raised at construction; no payload produced.

## Scenario 3 — Confidence bounds (US2, FR-004, SC-003)

Construct with `confidence` at `0.0`, `1.0`, `-0.1`, and `1.1`.

- **Expected**: `0.0` and `1.0` accepted; `-0.1` and `1.1` raise `ValidationError`.

## Scenario 4 — Empty vs populated entities (FR-005, Edge case)

Construct once omitting `entities`, once with a populated map.

- **Expected**: omitted → `entities == {}` (valid, not an error); populated → preserved key-for-key.

## Scenario 5 — Serialize/reconstruct round-trip (FR-010, SC-004)

Serialize with `model_dump(mode="json")`, then reconstruct with `model_validate(...)`.

- **Expected**: reconstructed instance equals the original (`==`); `timestamp` survives as ISO-8601; zero field loss.

## Scenario 6 — Traceability (US3, SC-005)

Inspect a constructed payload's `raw_input`, `timestamp`, and `source`.

- **Expected**: original unparsed input, a creation time, and an identifiable origin are all recoverable from the payload alone.

## Run the automated suite

```bash
./venv/bin/python -m pytest tests/test_intent_payload.py -v
```

- **Expected**: all scenarios above pass as parametrized unit tests. (Test authoring is part of the `/speckit-tasks` → `/speckit-implement` phase; this guide defines what those tests must prove.)

## Manual smoke check (optional)

```bash
./venv/bin/python -c "
from models import IntentPayload
p = IntentPayload(intent='ping', confidence=0.98, raw_input='ping', source='cli')
print(p.model_dump(mode='json'))
assert IntentPayload.model_validate(p.model_dump(mode='json')) == p
print('round-trip OK')
"
```

- **Expected**: prints the JSON dict then `round-trip OK`. (Requires `models.py` implemented — not yet done at plan time.)
