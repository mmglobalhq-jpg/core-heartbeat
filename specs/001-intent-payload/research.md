# Phase 0 Research: Intent Payload

All Technical Context unknowns resolved below. No open NEEDS CLARIFICATION remain.

## R1 — Confidence field: constraint mechanism

- **Decision**: `confidence: float` with `Field(ge=0.0, le=1.0)`. Pydantic v2 enforces the bound at construction and raises `ValidationError` on out-of-range input.
- **Rationale**: Directly satisfies FR-003/FR-004 and SC-003 (out-of-range rejected before reaching the router) using the framework's native, declarative constraint — no hand-written validator needed. `ge/le` (inclusive) matches the spec's "0–1 inclusive."
- **Alternatives considered**:
  - `condecimal`/`Decimal` — unnecessary precision for a confidence score; adds friction to JSON round-trip.
  - Custom `@field_validator` — more code for what `Field(ge, le)` does natively; reserved for cross-field rules only.
  - `Annotated[float, Ge(0), Le(1)]` — equivalent; `Field(...)` chosen for readability alongside other field metadata.

## R2 — Parameters/entities field: type

- **Decision**: `entities: dict[str, Any] = Field(default_factory=dict)`.
- **Rationale**: FR-005 requires a name→value collection that MAY be empty. `default_factory=dict` gives a safe empty default (no mutable-default aliasing bug) and satisfies the "may be empty" edge case without making the field optional/nullable. `Any` values match the spec's Assumption that params are flat and not per-intent typed in v1.
- **Alternatives considered**:
  - `dict[str, str]` — too narrow; extracted entities can be numbers, lists, booleans.
  - Optional/`None` default — rejected: an absent param set should be `{}`, not `null`, keeping consumers branch-free (FR-011).
  - Per-intent typed sub-models — out of scope per spec Assumptions (deferred beyond v1).

## R3 — Testing framework

- **Decision**: Use **pytest**; add it as a dev dependency (currently not installed in the venv). Author `tests/test_intent_payload.py` covering: valid construction, missing-identity rejection, confidence bounds (0, 1, <0, >1), empty-vs-populated entities, and serialize→reconstruct round-trip equality (SC-004).
- **Rationale**: pytest is the de facto standard for Pydantic/FastAPI projects and expresses the spec's acceptance scenarios cleanly. No test framework is currently present, so this is an additive tooling decision, not a migration.
- **Alternatives considered**:
  - stdlib `unittest` — heavier boilerplate, no fixtures/parametrize ergonomics.
  - No tests — rejected; SC-001..SC-005 are explicitly verifiable and warrant automated coverage.

## R4 — Timestamp default and type

- **Decision**: `timestamp: datetime = Field(default_factory=...)` producing a timezone-aware UTC value at construction.
- **Rationale**: FR-007 requires a creation timestamp; auto-defaulting means the boundary cannot forget to set it while still allowing an explicit override (e.g. reconstruction from serialized form). Timezone-aware UTC avoids ambiguous naive datetimes and serializes cleanly to ISO-8601 for the round-trip in SC-004.
- **Implementation note**: The exact default-factory callable (e.g. `lambda: datetime.now(timezone.utc)`) is an implementation detail for `models.py`; the contract is only "tz-aware UTC creation time, overridable."
- **Alternatives considered**:
  - Naive `datetime` — rejected: ambiguous across environments, lossy on round-trip.
  - Unix epoch `float`/`int` — less self-describing than ISO-8601; rejected for readability/auditability (User Story 3).
  - Required (no default) — rejected: forces every producer to set it, easy to get wrong; default+override is safer.

## R5 — Source field: type

- **Decision**: `source: str` (required, non-empty). Represents the originating channel/component (e.g. `"http"`, `"cli"`, `"webhook:slack"`).
- **Rationale**: FR-008 needs an identifiable origin for tracing (User Story 3, SC-005). A free-form string is the least-constraining contract that still supports auditing; the set of valid sources is an application concern that can harden into an `Enum`/`Literal` later without breaking the field's meaning.
- **Alternatives considered**:
  - `Enum`/`Literal` now — premature; the source vocabulary isn't fixed in v1 and locking it risks churn. Deferred.
  - Optional source — rejected: FR-008 makes origin mandatory for traceability.

## R6 — Serialization / round-trip strategy

- **Decision**: Rely on Pydantic v2 native `model_dump(mode="json")` ↔ `model_validate(...)` for lossless round-trip; datetime serializes to ISO-8601, `entities` to a JSON object.
- **Rationale**: FR-010 and SC-004 require lossless serialize/reconstruct. Pydantic v2's JSON mode handles `datetime` and nested containers deterministically, and `model_validate` reconstructs an equal instance, giving round-trip equality without custom (de)serialization code.
- **Alternatives considered**:
  - `dataclasses.asdict` + manual JSON — reintroduces the validation/typing work Pydantic already provides.
  - `pickle` — not cross-boundary/interop-safe; rejected.

## Cross-cutting: what stays OUT of models.py

Confirmed against spec Assumptions §2–§3 — the following are **router.py** concerns and are not modeled here:
- Mapping from `intent` identity → handler.
- The confidence acceptance/rejection threshold and any clarification/fallback policy.

The payload guarantees only a *valid, bounded, complete* structure; decisions about what to *do* with it belong downstream.
