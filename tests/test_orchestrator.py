"""Tests for the orchestration engine (features 003 + 004).

Routing is now model-driven, so these inject a scripted fake client (no network)
via monkeypatching orchestrator.get_client. A scripted
["local_llm","tool_execution","finish"] reproduces the feature-003 greet-plan
trace deterministically. Covers termination, cyclic accumulation, and determinism
(SC-001..SC-004, SC-007).
"""

from models import IntentPayload, RoutingDecision
import orchestrator
from orchestrator import MAX_STEPS, run


# --- scripted fake client (no network) --------------------------------------

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


def install(monkeypatch, choices):
    """Install a single scripted client instance for one run (iterator persists)."""
    client = _Client(choices)
    monkeypatch.setattr(orchestrator, "get_client", lambda: client)
    return client


def intent(identity="greet", confidence=0.9):
    return IntentPayload(intent=identity, confidence=confidence, raw_input="x", source="cli")


GREET_PLAN = ["local_llm", "tool_execution", "finish"]


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
    assert o.nodes_executed == ["local_llm", "tool_execution"]


def test_message_history_ordered_with_provenance(monkeypatch):
    install(monkeypatch, GREET_PLAN)
    o = run(intent("greet"))
    sources = [m.source for m in o.messages]
    assert sources == ["supervisor", "local_llm", "supervisor", "tool_execution", "supervisor"]
    assert [s for s in sources if s != "supervisor"] == ["local_llm", "tool_execution"]


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
