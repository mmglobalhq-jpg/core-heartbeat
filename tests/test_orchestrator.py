"""Tests for the orchestration engine (features 003 + 004 + 005).

Routing is model-driven, so these inject a scripted fake supervisor client (no
network) via monkeypatching orchestrator.get_client. A scripted
["local_llm","tool_execution","finish"] reproduces the feature-003 greet-plan
trace deterministically. Feature 005 made local_llm a live async Ollama call, so
these also inject an httpx.MockTransport Ollama client (no daemon) and drive the
now-async run() via asyncio.run. Covers termination, cyclic accumulation, and
determinism (SC-001..SC-004, SC-007).
"""

import asyncio

import httpx

from models import IntentPayload, RoutingDecision
import orchestrator
from orchestrator import MAX_STEPS


# --- scripted fake supervisor client (no network) ---------------------------

class _Resp:
    def __init__(self, choice):
        self.parsed = RoutingDecision(next_node=choice)
        self.text = None
        self.usage_metadata = None


class _Models:
    def __init__(self, choices):
        self._it = iter(choices)

    def generate_content(self, model, contents, config):
        try:
            return _Resp(next(self._it))
        except StopIteration:
            return _Resp("finish")


class _Client:
    def __init__(self, choices):
        self.models = _Models(choices)


# Mocked Ollama counts chosen to preserve the historical stub totals
# (local 10/20/30 + tool 5/0/5 = 15/20/35), so usage assertions stay stable.
def _ollama_handler(request):
    return httpx.Response(
        200,
        json={"response": "[local] mocked inference", "prompt_eval_count": 10, "eval_count": 20},
    )


def install(monkeypatch, choices):
    """Install scripted supervisor + MockTransport Ollama clients for one run."""
    client = _Client(choices)
    monkeypatch.setattr(orchestrator, "get_client", lambda *a, **k: client)
    monkeypatch.setattr(
        orchestrator,
        "build_ollama_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(_ollama_handler)),
    )
    return client


def run(payload):
    """Drive the now-async orchestrator.run from sync tests (no pytest-asyncio)."""
    return asyncio.run(orchestrator.run(payload))


def intent(identity="greet", confidence=0.9):
    return IntentPayload(intent=identity, confidence=confidence, raw_input="x", source="cli")


GREET_PLAN = ["tool_execution", "local_llm", "finish"]


# --- US1: terminates with a structured outcome (SC-001) ---------------------

def test_run_terminates_with_outcome(monkeypatch):
    install(monkeypatch, GREET_PLAN)
    o = run(intent())
    assert o.status == "completed"
    assert o.nodes_executed
    assert o.messages
    assert isinstance(o.steps, int)


# --- US2: cyclic accumulation (SC-003, SC-004) ------------------------------

def test_cyclic_plan_visits_both_workers_in_order(monkeypatch):
    install(monkeypatch, GREET_PLAN)
    o = run(intent("greet"))
    assert o.nodes_executed == ["tool_execution", "local_llm"]


def test_message_history_ordered_with_provenance(monkeypatch):
    install(monkeypatch, GREET_PLAN)
    o = run(intent("greet"))
    sources = [m.source for m in o.messages]
    assert sources == ["supervisor", "tool_execution", "supervisor", "local_llm", "supervisor"]
    assert [s for s in sources if s != "supervisor"] == ["tool_execution", "local_llm"]


def test_usage_totals_equal_sum_of_increments(monkeypatch):
    install(monkeypatch, GREET_PLAN)
    o = run(intent("greet"))
    # local_llm (10/20/30) + tool_execution (5/0/5); scripted fake reports no model usage
    assert o.usage.input_tokens == 15
    assert o.usage.output_tokens == 20
    assert o.usage.total_tokens == 35


# --- US2: bounded termination + finish-first (SC-002) -----------------------

def test_run_is_step_bounded(monkeypatch):
    install(monkeypatch, GREET_PLAN)
    o = run(intent("greet"))
    assert o.steps <= MAX_STEPS


def test_finish_first_yields_no_workers(monkeypatch):
    # reframed former "noop" test: model returns finish immediately
    install(monkeypatch, ["finish"])
    o = run(intent("ping"))
    assert o.status == "completed"
    assert o.nodes_executed == []
    assert o.usage.total_tokens == 0


# --- US2: determinism with a fixed script (SC-007) --------------------------

def test_run_is_deterministic(monkeypatch):
    install(monkeypatch, GREET_PLAN)
    a = run(intent("greet"))
    install(monkeypatch, GREET_PLAN)
    b = run(intent("greet"))
    assert a.model_dump() == b.model_dump()


# --- chat-history seeding: IntentPayload.history -> initial run messages -----

from models import HistoryTurn  # noqa: E402
from orchestrator import (  # noqa: E402
    HISTORY_LIMIT,
    WORKER_NODES,
    _build_local_prompt,
    _build_prompt,
    _seed_messages,
)


def _payload_with_history(turns):
    return IntentPayload(
        intent="chat",
        confidence=0.95,
        raw_input="current question",
        source="cli",
        history=turns,
    )


def test_seed_messages_empty_when_no_history():
    # Omitted history preserves today's stateless behavior: an empty seed.
    assert _seed_messages(intent("greet")) == []
    assert _seed_messages(_payload_with_history([])) == []


def test_seed_messages_preserves_order_and_maps_roles():
    turns = [
        HistoryTurn(role="user", content="first ask"),
        HistoryTurn(role="assistant", content="first answer"),
        HistoryTurn(role="user", content="second ask"),
    ]
    seeded = _seed_messages(_payload_with_history(turns))
    # order preserved, ascending step, roles mapped onto graph source vocabulary
    assert [(m.source, m.content, m.step) for m in seeded] == [
        ("user", "first ask", 0),
        ("assistant", "first answer", 1),
        ("user", "second ask", 2),
    ]


def test_seeded_history_never_counts_as_a_worker_reply():
    # Regression: assistant turns must NOT be sourced as a WORKER_NODES value
    # ("local_llm"/"tool_execution"), or the Supervisor's completion policy treats
    # the CURRENT question as already answered and routes straight to finish
    # (streaming nothing). Seeded turns are context only.
    turns = [
        HistoryTurn(role="user", content="hi"),
        HistoryTurn(role="assistant", content="hello"),
        HistoryTurn(role="assistant", content="anything else?"),
    ]
    seeded = _seed_messages(_payload_with_history(turns))
    assert all(m.source not in WORKER_NODES for m in seeded)


def test_seed_messages_truncates_to_limit():
    turns = [
        HistoryTurn(role="user", content=f"turn {i}")
        for i in range(HISTORY_LIMIT + 5)
    ]
    seeded = _seed_messages(_payload_with_history(turns))
    assert len(seeded) == HISTORY_LIMIT
    # the OLDEST turns are dropped — the most recent HISTORY_LIMIT are kept
    assert seeded[0].content == f"turn {5}"
    assert seeded[-1].content == f"turn {HISTORY_LIMIT + 4}"


def test_prior_context_reaches_answer_prompt_not_supervisor():
    # The fix for the "finishes before answering" bug: prior turns live in
    # `prior_context`, fed to the ANSWERING prompt (so the model has context) but
    # NOT to the Supervisor's completion logic (so it doesn't think the current
    # question is already answered and route straight to finish).
    payload = _payload_with_history([
        HistoryTurn(role="user", content="My name is Heath."),
        HistoryTurn(role="assistant", content="Hello Heath."),
    ])
    state = {
        "intent": payload,
        "messages": [],
        "prior_context": _seed_messages(payload),
        "user_id": "sandbox-user",
    }

    # Answerer SEES the prior conversation.
    local_prompt = _build_local_prompt(state)
    assert "My name is Heath." in local_prompt
    assert "Hello Heath." in local_prompt

    # Supervisor does NOT see prior turns, and counts zero composed answers for
    # this run — so it will still dispatch the current question to a worker.
    sup_prompt = _build_prompt(state)
    assert "Answers composed so far: 0" in sup_prompt
    assert "My name is Heath." not in sup_prompt


# --- C-3/C-4: entrypoint parity + vault prep -------------------------------

def test_run_syncs_the_vault_like_astream_run(monkeypatch):
    # C-3: the non-streaming /intent path (run) must localize the vault just like
    # /intent/stream (astream_run) — previously only astream_run did.
    calls = []

    async def fake_sync(uid):
        calls.append(uid)

    monkeypatch.setattr(orchestrator, "sync_user_vault", fake_sync)
    install(monkeypatch, ["finish"])
    run(intent("greet"))
    assert calls == [orchestrator.SANDBOX_USER_ID]
