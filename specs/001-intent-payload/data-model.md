# Phase 1 Data Model: Intent Payload

## Entity: `IntentPayload`

The self-contained record of one parsed intent moving from the API boundary to the router. Modeled as a Pydantic v2 `BaseModel` in `models.py`. Validated at construction; any consumer receiving an instance may trust the contract without re-validating (FR-009).

### Fields

| Field | Type | Required | Default | Constraint | Spec ref |
|-------|------|----------|---------|------------|----------|
| `intent` | `str` | ✅ | — | non-empty; unique intent identity used by router to select a handler | FR-001, FR-002 |
| `confidence` | `float` | ✅ | — | `ge=0.0, le=1.0` (inclusive); rejected out of range at construction | FR-003, FR-004 |
| `entities` | `dict[str, Any]` | ✅ (may be empty) | `{}` via `default_factory=dict` | name→value map; empty is valid | FR-005 |
| `raw_input` | `str` | ✅ | — | verbatim original input that produced the intent; not re-parsed/sanitized | FR-006 |
| `timestamp` | `datetime` | ✅ | tz-aware UTC now, overridable via `default_factory` | timezone-aware; ISO-8601 on serialize | FR-007 |
| `source` | `str` | ✅ | — | non-empty; originating channel/component identifier | FR-008 |

### Validation Rules

- **VR-1 (identity present)**: constructing without `intent` (or with an empty string) raises `ValidationError`. → FR-002, Edge "Missing intent identity".
- **VR-2 (confidence bounds)**: `confidence < 0` or `> 1` raises `ValidationError`. Boundary values `0.0` and `1.0` are accepted. → FR-004, SC-003, US2 scenario 3.
- **VR-3 (entities optional-empty)**: omitting `entities` yields `{}`; a populated map is preserved key-for-key. → FR-005, Edge "No extracted entities".
- **VR-4 (raw input required)**: `raw_input` must be present. Content is stored verbatim; the model performs no truncation or sanitization. → FR-006, Edge "Oversized/malformed raw input" (handling delegated to boundary).
- **VR-5 (source required)**: `source` must be present and non-empty. → FR-008.
- **VR-6 (strictness, REQUIRED)**: model MUST reject unknown/extra fields via `model_config = ConfigDict(extra="forbid")` so malformed payloads fail closed rather than silently carrying junk. → FR-009.
- **VR-7 (immutability, REQUIRED)**: model MUST be frozen via `model_config = ConfigDict(frozen=True)`; mutating any field after construction raises an error. Gateway payloads are strict and immutable. → user decision.

### Serialization Contract

- **Serialize**: `model_dump(mode="json")` → plain dict with `timestamp` as ISO-8601 string, `entities` as a JSON object, scalars as-is.
- **Reconstruct**: `model_validate(<dict>)` → an `IntentPayload` equal to the original (`==` holds). → FR-010, SC-004 (lossless round-trip).

### Relationships

- **Extracted Entities** — the `entities` map is a contained value, not a separate persisted entity. No foreign relationships; `IntentPayload` is a standalone value object.
- **Producer**: FastAPI boundary in `main.py` (out of scope for this feature).
- **Consumer**: `router.py`, which reads `intent` + `entities` to dispatch and applies its own confidence-threshold policy (out of scope for this feature).

### State Transitions

None. `IntentPayload` is an immutable value object created once at the boundary and read downstream. Immutability is enforced via `model_config = ConfigDict(frozen=True)` (see VR-7) — a hard requirement, not optional.

### Out of Scope (per spec Assumptions)

- Intent→handler mapping (router).
- Confidence acceptance threshold / clarification policy (router).
- Per-entity or multi-hypothesis confidence; nested/typed per-intent parameter schemas; multi-intent payloads. (v1 deferrals.)
