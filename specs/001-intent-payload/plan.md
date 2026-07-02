# Implementation Plan: Intent Payload

**Branch**: `001-intent-payload` | **Date**: 2026-07-01 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-intent-payload/spec.md`

## Summary

Define `IntentPayload`, the self-contained data structure that carries one parsed intent from the API boundary (`main.py`) to the router (`router.py`). It is modeled as a Pydantic v2 `BaseModel` in `models.py` with construction-time validation: a required intent identity, a confidence float constrained to `[0, 1]`, a name→value parameter map (may be empty), the verbatim raw input, a creation timestamp, and a source identifier. Intent→handler mapping and confidence-threshold policy are explicitly **out of scope** — they live in `router.py`. The payload's job is to guarantee a valid, serializable contract that any consumer can trust without re-validating.

## Technical Context

**Language/Version**: Python 3.14.4 (venv)

**Primary Dependencies**: Pydantic 2.13.4 (model + validation), FastAPI 0.139.0 (boundary that produces the payload; consumes the same model for request bodies)

**Storage**: N/A — the payload is an in-flight value object, not persisted by this feature

**Testing**: pytest (not yet installed; to be added to the dev dependencies — see research.md R3)

**Target Platform**: Linux server (WSL2 dev); FastAPI/uvicorn service

**Project Type**: Single project — small web service (`main.py` boundary, `router.py` dispatch, `models.py` schemas)

**Performance Goals**: Not a bottleneck; payload construct + validate is sub-millisecond. No explicit target beyond "not a measurable hotspot at expected request volume."

**Constraints**: Validation must happen at construction (fail-closed); round-trip serialize/reconstruct must be lossless (SC-004); required fields non-optional (SC-001).

**Scale/Scope**: One model class plus its validators and a small unit-test suite. ~1 file of production code (`models.py`), no schema migrations.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

The project constitution (`.specify/memory/constitution.md`) is the unpopulated Spec Kit template — it defines no ratified principles, so there are no enforceable gates. **Gate status: PASS (vacuously).**

Applied sensible defaults in lieu of ratified principles:
- **Simplicity / YAGNI**: single model class, no speculative fields, scope held to the spec's Assumptions (flat params, single scalar confidence, one intent per payload).
- **Test-first**: field contract and validation rules captured as testable acceptance scenarios; unit tests planned before/with implementation.
- **No scope creep**: routing policy and handler mapping deliberately excluded, matching spec Assumptions §2–§3.

*Post-Phase 1 re-check*: Design stayed within these defaults — single `BaseModel`, no new dependencies beyond the already-installed stack (pytest is dev-only tooling). **PASS.**

## Project Structure

### Documentation (this feature)

```text
specs/001-intent-payload/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── intent_payload.schema.json
├── checklists/
│   └── requirements.md  # from /speckit-specify
└── tasks.md             # Phase 2 output (/speckit-tasks — NOT created here)
```

### Source Code (repository root)

```text
models.py        # IntentPayload BaseModel + validators (this feature)
router.py        # Consumes IntentPayload; owns intent→handler map + confidence policy (out of scope here)
main.py          # FastAPI boundary; produces IntentPayload from incoming requests (out of scope here)

tests/
└── test_intent_payload.py   # Unit tests for construction, validation, serialization round-trip
```

**Structure Decision**: Keep the existing flat single-project layout (`models.py`, `router.py`, `main.py` at repo root) — it is already established and the feature adds exactly one model. No `src/` restructure is introduced; that would be unjustified churn for a one-model change. A `tests/` directory is added for the unit suite.

## Complexity Tracking

> No constitution violations to justify — section intentionally empty.
