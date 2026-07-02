"""Tests for the LangGraph orchestration engine (feature 003).

Covers termination, cyclic accumulation, bounded steps, immediate finish, and
determinism (SC-001..SC-004, SC-007).
"""

from models import IntentPayload
from orchestrator import MAX_STEPS, run


def intent(identity="greet", confidence=0.9):
    return IntentPayload(intent=identity, confidence=confidence, raw_input="x", source="cli")


# --- US1: terminates with a structured outcome (SC-001) ---------------------

def test_run_terminates_with_outcome():
    o = run(intent())
    assert o.status == "completed"          # terminal status
    assert o.nodes_executed                 # some nodes ran
    assert o.messages                       # history present
    assert isinstance(o.steps, int)         # returned (did not hang)


# --- US2: cyclic accumulation (SC-003, SC-004) ------------------------------

def test_cyclic_plan_visits_both_workers_in_order():
    o = run(intent("greet"))
    assert o.nodes_executed == ["local_llm", "tool_execution"]


def test_message_history_ordered_with_provenance():
    o = run(intent("greet"))
    sources = [m.source for m in o.messages]
    # supervisor routes, worker runs, back to supervisor, etc.
    assert sources == ["supervisor", "local_llm", "supervisor", "tool_execution", "supervisor"]
    # one worker entry per node execution, in order
    assert [s for s in sources if s != "supervisor"] == ["local_llm", "tool_execution"]


def test_usage_totals_equal_sum_of_increments():
    o = run(intent("greet"))
    # local_llm (10/20/30) + tool_execution (5/0/5)
    assert o.usage.input_tokens == 15
    assert o.usage.output_tokens == 20
    assert o.usage.total_tokens == 35


# --- US2: bounded termination + immediate finish (SC-002, edge) -------------

def test_run_is_step_bounded():
    o = run(intent("greet"))
    assert o.steps <= MAX_STEPS


def test_noop_intent_finishes_immediately():
    o = run(intent("ping"))
    assert o.status == "completed"
    assert o.nodes_executed == []
    assert o.usage.total_tokens == 0


def test_noop_variant():
    o = run(intent("noop"))
    assert o.nodes_executed == []


# --- US2: determinism (SC-007) ----------------------------------------------

def test_run_is_deterministic():
    a = run(intent("greet"))
    b = run(intent("greet"))
    assert a.model_dump() == b.model_dump()
