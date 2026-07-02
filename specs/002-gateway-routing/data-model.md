# Phase 1 Data Model: Gateway Routing Interface

New models added to `models.py` (alongside `IntentPayload` from feature 001). These are **response** models — unlike `IntentPayload` they are not frozen/strict, since the gateway constructs them internally.

## Enum: `Outcome`

`str, Enum` — the authoritative discriminator across all responses (SC-004).

| Member | Value |
|--------|-------|
| `ACCEPTED` | `"accepted"` |
| `THRESHOLD_REJECTED` | `"threshold_rejected"` |
| `VALIDATION_REJECTED` | `"validation_rejected"` |

## Base: `GatewayResponse`

The shared envelope carried by every gateway response (FR-013).

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `outcome` | `Outcome` | — (required) | Which of the three outcomes this response represents. |
| `usage` | `dict[str, Any] \| None` | `None` | Optional usage/metadata map. **Unpopulated in this MVP**; reserved for future token-count/cost pass-through (FR-013, SC-007). |

## `IntentAccepted(GatewayResponse)` — success (HTTP 200)

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `outcome` | `Outcome` | `Outcome.ACCEPTED` | Fixed for this variant. |
| `intent` | `str` | — | Echoed intent identity (FR-005). |
| `accepted` | `bool` | `True` | Explicit accepted status (FR-005). |
| `detail` | `str` | e.g. "Intent received and validated." | Human-readable confirmation. |

## `ThresholdRejected(GatewayResponse)` — policy rejection (HTTP 422)

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `outcome` | `Outcome` | `Outcome.THRESHOLD_REJECTED` | Fixed. |
| `intent` | `str` | — | Echoed intent identity. |
| `confidence` | `float` | — | The submitted confidence (FR-006). |
| `threshold` | `float` | — | The required threshold it failed to meet (FR-006). |
| `detail` | `str` | e.g. "Confidence below acceptance threshold." | Reason. |

## `ValidationRejected(GatewayResponse)` — contract rejection (HTTP 422)

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `outcome` | `Outcome` | `Outcome.VALIDATION_REJECTED` | Fixed. |
| `errors` | `list[dict[str, Any]]` | — | What was invalid, derived from the validation exception's `.errors()` (FR-007). |
| `detail` | `str` | e.g. "Submission failed intent contract validation." | Summary. |

## `HealthStatus` — liveness (HTTP 200)

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `status` | `str` | `"online"` | Liveness indicator (FR-014). |
| `service` | `str` | `"core-heartbeat"` | Identifies the responding service. |

## Config value: Acceptance Threshold (not a model)

Resolved at startup by a loader in `main.py`/`router.py`, stored on `app.state.confidence_threshold`.

| Rule | Behavior | Spec ref |
|------|----------|----------|
| env `HEARTBEAT_CONFIDENCE_THRESHOLD` unset/blank | default `0.5` | FR-009, FR-012, edge "missing/blank" |
| parseable float in `[0.0, 1.0]` | use it | FR-003/004 |
| parseable float outside `[0.0, 1.0]` | raise clear config error at startup | edge "out of range" |
| unparseable | raise clear config error at startup | FR-012 |

## Decision Logic (helpers in `router.py`)

- **`decide(confidence, threshold) -> bool`**: returns `confidence >= threshold` (inclusive, VR from spec US1 scenario 2 / edge "exactly at threshold"). Pure, unit-tested (SC-002/SC-005).
- **Flow for `POST /intent`**:
  1. FastAPI validates body as `IntentPayload`. On failure → `RequestValidationError` → exception handler → `ValidationRejected` (422). Threshold never evaluated (SC-003).
  2. On valid body → `decide(payload.confidence, threshold)`:
     - `True` → `IntentAccepted` (200).
     - `False` → `ThresholdRejected` (422).

## Relationships

- `IntentPayload` (001) is the **request** body for `POST /intent`; unchanged by this feature (FR-011).
- All response models share `GatewayResponse` as base (FR-013).
- No persistence, no handler entities (FR-010).

## State Transitions

None. Each request is evaluated independently and statelessly; `/health` reflects process reachability only.
