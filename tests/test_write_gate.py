"""The write-confirmation gate.

Native tool calling can turn one sentence into a dozen ``create_calendar_event``
calls, and there is no undo beyond deleting each event by hand. Before this gate
existed, "add my football schedule to my calendar" wrote twelve real events with no
opportunity to look at them first — the earlier "I can add these, want me to?"
behaviour only ever appeared because the router *couldn't* act, and enabling native
tool calling removed that accident.

So a batch of writes is proposed rather than run, and the user's next message
releases it. Reads are never gated: making someone confirm "what's on my calendar"
would be noise, and noise trains people to approve without reading.

The asymmetry to keep in mind while reading these tests: a missed confirmation
re-proposes, which is mildly annoying; a false confirmation performs writes nobody
authorized. Everything here is built to fail toward asking again.
"""

import pytest

import orchestrator
from models import IntentPayload, Message, RoutingDecision, TokenUsage, ToolArgs


@pytest.fixture(autouse=True)
def _no_leaked_plans():
    """Pending plans live in a module-level dict keyed by user, so without this a
    proposal from one test would release inside the next."""
    orchestrator._pending_plans.clear()
    yield
    orchestrator._pending_plans.clear()


def _calls(n, name="create_calendar_event"):
    return [{"name": name, "args": {"summary": f"Game {i}", "start": f"2026-09-0{i%9+1}T19:00:00"}}
            for i in range(n)]


def _route(calls, raw="add my schedule", prior=None):
    """Run a native-style decision through the real guard/gate path."""
    state = {
        "intent": IntentPayload(intent="calendar_add", confidence=0.9, source="t", raw_input=raw),
        "messages": [], "prior_context": prior or [], "user_id": "u1",
        "documents": "", "document_images": [], "visited": [], "step": 0,
    }
    first = calls[0] if calls else None
    decision = RoutingDecision(
        next_node="tool_execution" if calls else "local_llm",
        tool_name=first["name"] if first else None,
        tool_args=ToolArgs(**(first["args"] if first else {})),
    )
    return orchestrator._finish_routing(state, 0, decision, TokenUsage(), calls)


def _assistant_turn():
    return [Message(source="assistant", content="I can add these 12 games. Confirm?", step=0)]


# --- what gets gated --------------------------------------------------------


def test_large_write_batch_is_proposed_not_executed():
    out = _route(_calls(12))
    assert out["next"] == "local_llm"
    assert out["tool_calls"] is None, "writes must not be dispatched before approval"
    assert out["tool_request"] is None
    assert len(out["pending_plan"]) == 12


def test_small_write_batch_runs_directly():
    """One or two wrong events are trivial to delete; gating them is just friction."""
    out = _route(_calls(2))
    assert out["next"] == "tool_execution"
    assert len(out["tool_calls"]) == 2
    assert out["pending_plan"] is None


def test_threshold_is_the_documented_boundary():
    assert orchestrator.WRITE_CONFIRM_THRESHOLD == 3
    assert _route(_calls(2))["pending_plan"] is None
    assert _route(_calls(3))["pending_plan"] is not None


def test_any_delete_is_always_confirmed():
    """Deletes are unrecoverable, so count doesn't matter."""
    out = _route([{"name": "delete_calendar_event", "args": {"event_id": "abc"}}])
    assert out["next"] == "local_llm"
    assert len(out["pending_plan"]) == 1


def test_reads_are_never_gated():
    reads = [{"name": "list_calendar_events", "args": {}},
             {"name": "query_knowledge_base", "args": {"query": "x"}},
             {"name": "get_latest_reit_report", "args": {"reit_symbol": "ARR"}}]
    out = _route(reads)
    assert out["next"] == "tool_execution"
    assert len(out["tool_calls"]) == 3
    assert out["pending_plan"] is None


def test_reads_mixed_with_a_big_write_batch_are_held_too():
    """Splitting the batch would run half a plan the user never saw."""
    out = _route(_calls(5) + [{"name": "list_calendar_events", "args": {}}])
    assert out["next"] == "local_llm"
    assert len(out["pending_plan"]) == 6


# --- releasing the plan -----------------------------------------------------


def test_propose_then_confirm_actually_runs_the_batch():
    """The full two-turn flow, which is the only way confirmation works.

    This is the bug that shipped: `pending_plan` is per-run state, so on the
    confirmation turn the router saw a bare "yes" and would have had to rebuild
    twelve tool calls from its own earlier prose. It didn't — it asked "Shall I
    proceed?" again, and NOTHING was ever dispatched. Verified against the real
    calendar afterwards: zero events created.
    """
    proposal = _route(_calls(12), raw="add my schedule")
    assert proposal["next"] == "local_llm" and len(proposal["pending_plan"]) == 12

    confirm = _route([], raw="yes", prior=_assistant_turn())
    assert confirm["next"] == "tool_execution"
    assert len(confirm["tool_calls"]) == 12, "the approved plan must actually run"


def test_confirmation_replays_the_approved_calls_not_a_fresh_batch():
    """The user approved a specific list; that list is what must run, rather than
    whatever the model would regenerate on the confirmation turn."""
    original = _calls(4)
    original[0]["args"]["summary"] = "Approved game"
    _route(original, raw="add them")

    confirm = _route([{"name": "create_calendar_event", "args": {"summary": "Something else"}}],
                     raw="yes", prior=_assistant_turn())
    assert [c["args"]["summary"] for c in confirm["tool_calls"]][0] == "Approved game"
    assert len(confirm["tool_calls"]) == 4


def test_a_plan_is_single_use():
    """A second "yes" must not run the same writes twice."""
    _route(_calls(5), raw="add them")
    first = _route([], raw="yes", prior=_assistant_turn())
    assert len(first["tool_calls"]) == 5

    second = _route([], raw="yes", prior=_assistant_turn())
    assert second["tool_calls"] is None
    assert second["plan_note"], "a repeat yes must be answered honestly, not silently"


def test_confirmation_with_no_stored_plan_admits_it():
    """Answering "yes" with "proceeding to add them" while dispatching nothing is
    exactly the false-completion this gate exists to prevent."""
    out = _route([], raw="yes", prior=_assistant_turn())
    assert out["next"] == "local_llm"
    assert out["tool_calls"] is None
    assert "do not have it any more" in out["plan_note"]
    assert "NOT claim anything is being added" in out["plan_note"]


def test_moving_on_discards_the_proposal():
    """A stale plan must not fire against a later, unrelated "yes"."""
    _route(_calls(6), raw="add my schedule")
    _route([{"name": "list_calendar_events", "args": {}}], raw="what's on friday?")

    out = _route([], raw="yes", prior=_assistant_turn())
    assert out["tool_calls"] is None, "an abandoned plan must not run later"


def test_affirmation_without_any_prior_assistant_turn_does_not_release():
    """An opening "yes" in a fresh conversation must not authorize writes."""
    _route(_calls(12), raw="add my schedule")
    out = _route([], raw="yes", prior=[])
    assert out["next"] == "local_llm"
    assert out["tool_calls"] is None


def test_qualified_yes_is_a_revision_not_a_confirmation():
    """"yes, but move the first one to Friday" changes the plan — releasing the
    batch proposed BEFORE that edit would write the wrong dates."""
    out = _route(_calls(12), raw="yes, but move the first one to Friday",
                 prior=_assistant_turn())
    assert out["next"] == "local_llm", "a qualified yes must re-propose, not execute"
    assert out["tool_calls"] is None


def test_a_new_instruction_is_not_a_confirmation():
    out = _route(_calls(12), raw="actually add my basketball schedule instead",
                 prior=_assistant_turn())
    assert out["next"] == "local_llm"


def test_common_go_aheads_are_recognized():
    for phrase in ("yes", "yep", "go ahead", "do it", "please do", "confirm",
                   "add them", "sounds good", "ok", "proceed"):
        assert orchestrator._is_affirmation(phrase), phrase


def test_negations_and_questions_are_not_affirmations():
    for phrase in ("no", "not yet", "wait", "why?", "what would that do?",
                   "no, cancel that", "hold on"):
        assert not orchestrator._is_affirmation(phrase), phrase


# --- how the proposal is presented ------------------------------------------


def test_plan_block_lists_the_actions_with_dates():
    state = {"pending_plan": _calls(3)}
    block = orchestrator._pending_plan_block(state)
    assert "Game 0" in block and "2026-09-01T19:00:00" in block
    assert "1." in block and "3." in block


def test_plan_block_forbids_claiming_the_work_is_done():
    """The model saying "added!" about events that don't exist is worse than not
    offering — the user would stop checking."""
    block = orchestrator._pending_plan_block({"pending_plan": _calls(4)})
    assert "NOT been performed" in block
    assert "Do NOT say the actions are done" in block


def test_no_plan_means_no_block_at_all():
    """Ordinary turns must be byte-identical to before the gate existed."""
    assert orchestrator._pending_plan_block({"pending_plan": None}) == ""
    assert orchestrator._pending_plan_block({}) == ""


def test_plan_reaches_the_compose_prompt():
    state = {
        "intent": IntentPayload(intent="calendar_add", confidence=0.9, source="t",
                                raw_input="add them"),
        "messages": [], "prior_context": [], "user_id": "u1",
        "documents": "", "pending_plan": _calls(5),
    }
    prompt = orchestrator._build_local_prompt(state)
    assert "PROPOSED" in prompt and "Game 0" in prompt


def test_delete_is_described_legibly():
    line = orchestrator._describe_call(
        {"name": "delete_calendar_event", "args": {"event_id": "evt123"}}
    )
    assert "Delete" in line and "evt123" in line


# --- confirmations must be checkable ----------------------------------------


def test_update_shows_new_values_not_just_field_names():
    """"start" tells the user nothing about whether the change is correct."""
    line = orchestrator._describe_call(
        {"name": "update_calendar_event",
         "args": {"event_id": "e1", "start": "2026-09-02T10:00:00"}}
    )
    assert "2026-09-02T10:00:00" in line


def test_destructive_actions_are_not_approved_by_bare_id():
    """A raw Google event id is unverifiable. The listing that produced it is in the
    same turn's context, so the model is told to resolve it to a title and time —
    otherwise the user is approving a delete they cannot check."""
    block = orchestrator._pending_plan_block({"pending_plan": [
        {"name": "delete_calendar_event", "args": {"event_id": "7f3k9d2m"}},
    ]})
    assert "[id: 7f3k9d2m]" in block
    assert "name the event by its title, date and time" in block
    assert "NEVER ask someone to approve deleting or changing a bare id" in block
    assert "cannot identify that event" in block


def test_every_calendar_write_is_covered_by_the_gate():
    """Adding, editing and removing must all be gated; searching must not be."""
    from tools.catalog import WRITE_TOOLS
    for name in ("create_calendar_event", "update_calendar_event", "delete_calendar_event"):
        assert name in WRITE_TOOLS, f"{name} would bypass confirmation entirely"
    assert "list_calendar_events" not in WRITE_TOOLS


def test_a_second_supervisor_step_does_not_apologise_after_success():
    """The Supervisor runs once per STEP, not once per turn, and raw_input stays
    "yes" the whole turn.

    Observed in production: step 0 released the plan and wrote 9 calendar events;
    step 1 re-entered the confirmation branch, found the plan correctly consumed,
    and answered "I don't have the details of what you approved" — an apology
    emitted straight after the writes succeeded. The user saw only the apology,
    retried, and ended up with duplicate events.
    """
    _route(_calls(9), raw="add my schedule")
    released = _route([], raw="yes", prior=_assistant_turn())
    assert len(released["tool_calls"]) == 9

    # Same turn, next step: tool_execution has now run.
    state = {
        "intent": IntentPayload(intent="calendar_add", confidence=0.9, source="t",
                                raw_input="yes"),
        "messages": [], "prior_context": _assistant_turn(), "user_id": "u1",
        "documents": "", "document_images": [], "visited": ["tool_execution"], "step": 1,
    }
    decision = RoutingDecision(next_node="local_llm", tool_name=None, tool_args=ToolArgs())
    out = orchestrator._finish_routing(state, 1, decision, TokenUsage(), [])
    assert out["plan_note"] is None, "must not apologise for work that just succeeded"


def test_a_genuine_repeat_yes_still_reports_honestly():
    """The fix must not silence the real case: a NEW turn where nothing is held."""
    out = _route([], raw="yes", prior=_assistant_turn())
    assert out["plan_note"] is not None


# --- repeat-call guard ------------------------------------------------------


def _route_with_history(calls, executed, raw="what football games are on my calendar?"):
    state = {
        "intent": IntentPayload(intent="calendar", confidence=0.9, source="t", raw_input=raw),
        "messages": [], "prior_context": [], "user_id": "u1",
        "documents": "", "document_images": [], "visited": ["tool_execution"],
        "step": 1, "executed_calls": executed,
    }
    first = calls[0] if calls else None
    decision = RoutingDecision(
        next_node="tool_execution" if calls else "local_llm",
        tool_name=first["name"] if first else None,
        tool_args=ToolArgs(**(first["args"] if first else {})),
    )
    return orchestrator._finish_routing(state, 1, decision, TokenUsage(), calls)


def test_an_identical_repeat_call_is_dropped():
    """Observed: asking about football games on a calendar with none produced
    list_calendar_events four times, exhausted MAX_STEPS, and returned "No reply
    produced (status: halted_step_bound)". An empty result is the answer."""
    call = [{"name": "list_calendar_events", "args": {}}]
    sig = orchestrator._call_signature("list_calendar_events", {})
    out = _route_with_history(call, [sig])
    assert out["next"] == "local_llm", "must compose, not search again"
    assert out["tool_calls"] is None


def test_same_tool_with_different_args_gets_a_bounded_retry():
    """Narrowing a date range is legitimate; looping on it is not."""
    sig = orchestrator._call_signature("list_calendar_events", {})
    wider = [{"name": "list_calendar_events", "args": {"time_min": "2026-01-01T00:00:00"}}]
    out = _route_with_history(wider, [sig])
    assert out["next"] == "tool_execution", "a genuine retry with new args is allowed"

    out2 = _route_with_history(wider, [sig, sig])  # already at the cap
    assert out2["next"] == "local_llm", "but not without bound"


def test_a_fresh_call_is_unaffected():
    call = [{"name": "list_calendar_events", "args": {}}]
    out = _route_with_history(call, [])
    assert out["next"] == "tool_execution"
    assert len(out["tool_calls"]) == 1


def test_repeats_are_dropped_without_losing_the_rest_of_the_batch():
    sig = orchestrator._call_signature("list_calendar_events", {})
    mixed = [
        {"name": "list_calendar_events", "args": {}},
        {"name": "get_latest_reit_report", "args": {"reit_symbol": "ARR"}},
    ]
    out = _route_with_history(mixed, [sig])
    assert [c["name"] for c in out["tool_calls"]] == ["get_latest_reit_report"]
