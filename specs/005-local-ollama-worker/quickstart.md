# Quickstart: Local Ollama Worker Node

Validation guide for feature 005. Implementation details live in `tasks.md` /
the implementation phase; this file proves the feature works end-to-end.

## Prerequisites

- Project venv (`./venv`) with existing deps — **no new package required**
  (`httpx` is already a runtime dep).
- For the **test** path: nothing else. Tests never touch the network or a daemon.
- For the **live** path (optional): a running Ollama daemon with the model pulled:
  ```
  ollama serve            # if not already running
  ollama pull qwen2.5:7b  # already done per the feature request
  ```

## Validate via the test suite (no daemon, no network — primary gate)

```
./venv/bin/python -m pytest tests/ -q
```

Expected: all green, including the new `tests/test_local_worker.py` and the
updated orchestrator/gateway tests. Confirms:
- **SC-001** — a routed run carries the mocked model's real text (no stub string).
- **SC-002** — mocked token counts sum field-wise into the run total.
- **SC-003** — every failure mode (unreachable / timeout / non-2xx / bad body)
  yields a recorded, categorized failure and a terminating run, no exception.
- **SC-004** — the suite passes with **no** Ollama daemon and **no** network.

Failure modes are exercised by an `httpx.MockTransport` handler injected via
`monkeypatch orchestrator.build_ollama_client` (see `contracts/local_worker.md`).

## Validate the live path (optional, requires the daemon)

With the daemon up and `qwen2.5:7b` pulled, drive an accepted intent whose
Supervisor routes to the local worker and confirm the response's orchestration
messages contain genuine model text and non-zero usage:

```
GEMINI_API_KEY=<key> ./venv/bin/uvicorn main:app --port 8000
# then POST an accepted intent to /intent and inspect orchestration.messages + usage
```

Expected: a `local_llm` message with real generated text and
`usage.total_tokens > 0` reflecting `prompt_eval_count + eval_count`.

To confirm **graceful degradation** (SC-003, Principle IV) without stopping to
script a failure: stop the daemon and repeat — the accepted intent still returns
HTTP 200, the run still terminates, and the orchestration messages contain
`local inference failure: unreachable`.

## Configuration knobs (all optional; defaults are zero-config)

| Var | Default |
|-----|---------|
| `OLLAMA_URL` | `http://localhost:11434/api/generate` |
| `OLLAMA_MODEL` | `qwen2.5:7b` |
| `OLLAMA_TIMEOUT_MS` | `120000` |

## Done when

- `pytest tests/` is green with no daemon/network (primary gate).
- (Optional) live path returns real local-model text + usage, and degrades to a
  recorded `unreachable` failure when the daemon is down.
