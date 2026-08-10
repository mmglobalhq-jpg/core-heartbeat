"""The tool catalog, and executing several tool calls in one turn.

Two things are under test here.

**The catalog** (tools/catalog.py) is the single declaration of every tool. The
assertions that matter are that it covers exactly what is dispatchable, and that
``user_id`` never appears in a model-facing schema — the model must not be able to
name whose calendar or vault it is touching.

**Multi-call execution.** The structured-output router emits one tool call per
supervisor round-trip, so adding a football schedule cost one hop per game and ran
into ``MAX_STEPS = 8`` around the seventh. ``tool_execution`` now accepts a list, so
a whole schedule is one step. These tests pin the parts that are easy to get wrong:
that every call actually runs, that unknown names are dropped before reaching a
backend, that ``user_id`` comes from state rather than from the call, and that the
UI events still fire once per call and in call order.
"""

import time

import orchestrator
from models import Message
from tools.catalog import ALL_TOOLS, CATALOG_TOOL_NAMES, TOOLS_BY_NAME, WRITE_TOOLS


def _state(**over):
    base = {"step": 0, "user_id": "user-abc", "tool_request": None, "tool_calls": None}
    base.update(over)
    return base


# --- the catalog ------------------------------------------------------------


def test_catalog_covers_exactly_what_is_dispatchable():
    assert CATALOG_TOOL_NAMES == orchestrator.DISPATCHABLE_TOOLS


def test_no_tool_exposes_user_id_or_state_to_the_model():
    """The security property the whole per-user isolation rests on.

    ``user_id`` is injected from graph state; if it ever became a model-facing
    argument, an emitted call could name another user's data.
    """
    for t in ALL_TOOLS:
        args = set(t.args)
        assert "state" not in args, f"{t.name} exposes state to the model"
        assert "user_id" not in args, f"{t.name} exposes user_id to the model"


def test_every_tool_has_a_description():
    """Docstrings replaced the Supervisor's prose catalogue — they are now the
    model's only description of each tool."""
    for t in ALL_TOOLS:
        assert t.description and len(t.description.strip()) > 20, t.name


def test_write_tools_are_exactly_the_mutating_ones():
    # Deliberately an explicit literal, not derived. Adding a tool that changes
    # something must be a conscious edit here, because membership is what puts it
    # behind the confirmation gate — a mutating tool omitted from this set runs
    # without ever asking.
    assert WRITE_TOOLS == {
        "write_user_note",
        "create_calendar_event",
        "update_calendar_event",
        "delete_calendar_event",
        # Change what the briefing covers. Delivery time, timezone and email
        # address are deliberately NOT writable by chat.
        "add_briefing_topic",
        "remove_briefing_topic",
    }
    assert WRITE_TOOLS <= CATALOG_TOOL_NAMES


def test_calendar_tools_take_naive_datetimes_in_their_description():
    """A UTC offset here double-applies the timezone and lands events at the wrong
    hour, so the instruction has to survive in the description the model sees."""
    for name in ("create_calendar_event", "list_calendar_events", "update_calendar_event"):
        assert "NO timezone offset" in TOOLS_BY_NAME[name].description


# --- multi-call execution ---------------------------------------------------


def test_single_tool_request_still_works(monkeypatch):
    """The pre-existing shape must behave exactly as before."""
    monkeypatch.setattr(orchestrator, "run_vault_tool", lambda n, u, a: f"read {a['filename']}")
    out = orchestrator.tool_execution(
        _state(tool_request={"name": "read_user_note", "args": {"filename": "a.md"}})
    )
    assert len(out["messages"]) == 1
    assert out["messages"][0].content == "[tool:read_user_note] read a.md"
    assert out["visited"] == ["tool_execution"]
    assert out["step"] == 1


def test_no_request_keeps_the_stub_behaviour():
    out = orchestrator.tool_execution(_state())
    assert out["messages"][0].content == "[stub] tool executed"


def test_every_call_in_a_batch_runs(monkeypatch):
    """The bulk-calendar case: a whole schedule in ONE step."""
    seen = []

    def fake(name, user_id, args):
        seen.append(args["summary"])
        return f"created {args['summary']}"

    monkeypatch.setattr(orchestrator, "run_calendar_tool", fake)
    games = [f"Game {i}" for i in range(12)]
    out = orchestrator.tool_execution(
        _state(tool_calls=[
            {"name": "create_calendar_event",
             "args": {"summary": g, "start": "2026-08-21T19:00:00", "end": "2026-08-21T21:00:00"}}
            for g in games
        ])
    )
    assert sorted(seen) == sorted(games)
    assert len(out["messages"]) == 12
    # One step for the whole batch — this is what MAX_STEPS used to make impossible.
    assert out["step"] == 1


def test_batch_results_stay_in_call_order(monkeypatch):
    """Completion order must not reorder the transcript: the first call's result has
    to be the first message even when it finishes last."""
    def fake(name, user_id, args):
        time.sleep(0.05 if args.get("summary") == "slow" else 0.0)
        return f"done {args.get('summary')}"

    monkeypatch.setattr(orchestrator, "run_calendar_tool", fake)
    out = orchestrator.tool_execution(
        _state(tool_calls=[
            {"name": "create_calendar_event", "args": {"summary": "slow"}},
            {"name": "create_calendar_event", "args": {"summary": "fast"}},
        ])
    )
    assert [m.content for m in out["messages"]] == [
        "[tool:create_calendar_event] done slow",
        "[tool:create_calendar_event] done fast",
    ]


def test_batch_actually_runs_concurrently(monkeypatch):
    """Sequential execution of a 12-game schedule would be ~12x a single call.

    Four 0.2s calls take ~0.2s in parallel and ~0.8s serially; asserting well under
    the serial time keeps this from passing on a slow machine by accident.
    """
    monkeypatch.setattr(orchestrator, "run_calendar_tool",
                        lambda n, u, a: (time.sleep(0.2), "ok")[1])
    started = time.monotonic()
    out = orchestrator.tool_execution(
        _state(tool_calls=[
            {"name": "create_calendar_event", "args": {"summary": str(i)}} for i in range(4)
        ])
    )
    elapsed = time.monotonic() - started
    assert len(out["messages"]) == 4
    assert elapsed < 0.5, f"took {elapsed:.2f}s — looks sequential, not parallel"


def test_unknown_tool_names_never_reach_a_backend(monkeypatch):
    """A hallucinated or stale name is dropped at the gate, not dispatched."""
    called = []
    monkeypatch.setattr(orchestrator, "run_calendar_tool",
                        lambda n, u, a: called.append(n) or "ok")
    out = orchestrator.tool_execution(
        _state(tool_calls=[
            {"name": "send_email", "args": {"to": "boss@example.com"}},
            {"name": "create_calendar_event", "args": {"summary": "real"}},
        ])
    )
    assert called == ["create_calendar_event"]
    assert len(out["messages"]) == 1


def test_user_id_comes_from_state_not_from_the_call(monkeypatch):
    """A call carrying its own user_id must not redirect the operation."""
    seen = {}
    monkeypatch.setattr(orchestrator, "run_vault_tool",
                        lambda n, u, a: seen.update(uid=u, args=a) or "ok")
    orchestrator.tool_execution(
        _state(user_id="real-user",
               tool_calls=[{"name": "read_user_note",
                            "args": {"filename": "a.md", "user_id": "victim"}}])
    )
    assert seen["uid"] == "real-user"


def test_tool_calls_supersedes_tool_request(monkeypatch):
    monkeypatch.setattr(orchestrator, "run_vault_tool", lambda n, u, a: "from-list")
    out = orchestrator.tool_execution(
        _state(tool_request={"name": "read_user_note", "args": {"filename": "old.md"}},
               tool_calls=[{"name": "read_user_note", "args": {"filename": "new.md"}}])
    )
    assert len(out["messages"]) == 1
    assert "from-list" in out["messages"][0].content


def test_kb_sources_survive_a_mixed_batch(monkeypatch):
    """The citation channel must still be populated when a KB call is one of several."""
    monkeypatch.setattr(orchestrator, "run_graphrag_tool",
                        lambda n, u, a: ("ctx", ["Doc A"]))
    monkeypatch.setattr(orchestrator, "run_calendar_tool", lambda n, u, a: "ok")
    out = orchestrator.tool_execution(
        _state(tool_calls=[
            {"name": "create_calendar_event", "args": {"summary": "x"}},
            {"name": "query_knowledge_base", "args": {"query": "q"}},
        ])
    )
    assert out["kb_sources"] == ["Doc A"]


def test_one_ui_event_per_call_in_order(monkeypatch):
    """The SSE tool indicator must fire per call. Events are dispatched on the graph
    thread precisely because pool threads don't inherit the callback contextvar."""
    events = []
    monkeypatch.setattr(orchestrator, "run_calendar_tool", lambda n, u, a: f"ok-{a['summary']}")
    monkeypatch.setattr(orchestrator, "dispatch_custom_event",
                        lambda name, payload: events.append(payload))
    orchestrator.tool_execution(
        _state(tool_calls=[
            {"name": "create_calendar_event", "args": {"summary": "a"}},
            {"name": "create_calendar_event", "args": {"summary": "b"}},
        ])
    )
    assert [e["args"]["summary"] for e in events] == ["a", "b"]
    assert [e["result"] for e in events] == ["ok-a", "ok-b"]


def test_a_failing_tool_does_not_take_down_the_batch(monkeypatch):
    """run_*_tool converts failures to 'error: ...' strings; the batch still returns
    a message per call so the model can see which one failed."""
    def fake(name, user_id, args):
        if args.get("summary") == "bad":
            return "error: Calendar returned 403"
        return "created"

    monkeypatch.setattr(orchestrator, "run_calendar_tool", fake)
    out = orchestrator.tool_execution(
        _state(tool_calls=[
            {"name": "create_calendar_event", "args": {"summary": "good"}},
            {"name": "create_calendar_event", "args": {"summary": "bad"}},
        ])
    )
    assert len(out["messages"]) == 2
    assert "error: Calendar returned 403" in out["messages"][1].content


# --- native tool-calling router ---------------------------------------------


class _FakeResponse:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls
        self.usage_metadata = {"input_tokens": 10, "output_tokens": 20}


class _FakeBound:
    def __init__(self, tool_calls):
        self._tool_calls = tool_calls

    def invoke(self, _messages):
        return _FakeResponse(self._tool_calls)


def _native_state(raw="do the thing", **over):
    from models import IntentPayload
    base = {
        "intent": IntentPayload(intent="general", confidence=0.9, source="t", raw_input=raw),
        "messages": [], "prior_context": [], "user_id": "user-abc",
        "documents": "", "document_images": [], "visited": [], "step": 0,
    }
    base.update(over)
    return base


def test_native_tool_calling_is_off_by_default(monkeypatch):
    """The hot path must not change unless the flag is explicitly set."""
    monkeypatch.delenv("NATIVE_TOOL_CALLING", raising=False)
    import importlib
    assert importlib.reload(orchestrator).NATIVE_TOOL_CALLING is False


def test_native_prompt_does_not_instruct_the_routing_schema():
    """Measured failure: given the structured-output prompt, the model describes a
    route instead of calling tools and every turn yields zero tool calls."""
    prompt = orchestrator._build_native_prompt(_native_state())
    for token in ("next_node", "tool_name", "tool_args", "Route to local_llm"):
        assert token not in prompt, f"native prompt still instructs {token!r}"


def test_native_prompt_does_not_duplicate_the_tool_catalogue():
    """Descriptions come from bind_tools; repeating them here would recreate the
    duplication that let the calendar tools go stale."""
    prompt = orchestrator._build_native_prompt(_native_state())
    described = sum(1 for name in orchestrator.DISPATCHABLE_TOOLS if name in prompt)
    assert described == 0, "native prompt is re-listing tools that bind_tools declares"


def test_native_no_tool_calls_routes_to_compose(monkeypatch):
    monkeypatch.setattr(orchestrator, "_native_chat_model", lambda p: _FakeBound([]))
    nxt, calls, failure, usage = orchestrator._decide_native(_native_state(), "gemini-2.5-flash")
    assert (nxt, calls, failure) == ("local_llm", [], None)
    assert usage.input_tokens == 10 and usage.output_tokens == 20


def test_native_emits_every_call_it_receives(monkeypatch):
    batch = [
        {"name": "create_calendar_event", "args": {"summary": f"G{i}", "start": "x", "end": "y"}}
        for i in range(12)
    ]
    monkeypatch.setattr(orchestrator, "_native_chat_model", lambda p: _FakeBound(batch))
    nxt, calls, failure, _ = orchestrator._decide_native(_native_state(), "gemini-2.5-flash")
    assert nxt == "tool_execution"
    assert len(calls) == 12, "the whole batch must survive routing"
    assert failure is None


def test_native_drops_names_that_cannot_be_dispatched(monkeypatch):
    monkeypatch.setattr(orchestrator, "_native_chat_model", lambda p: _FakeBound([
        {"name": "send_email", "args": {"to": "x"}},
        {"name": "list_calendar_events", "args": {}},
    ]))
    _, calls, _, _ = orchestrator._decide_native(_native_state(), "gemini-2.5-flash")
    assert [c["name"] for c in calls] == ["list_calendar_events"]


def test_native_missing_client_reports_failure_not_a_crash(monkeypatch):
    monkeypatch.setattr(orchestrator, "_native_chat_model", lambda p: None)
    nxt, calls, failure, _ = orchestrator._decide_native(_native_state(), "gemini-2.5-flash")
    assert nxt is None and calls == []
    assert failure is not None and failure.category == "missing_credential"


def test_native_api_error_is_converted_to_a_failure(monkeypatch):
    class Boom:
        def invoke(self, _m):
            raise RuntimeError("upstream exploded")

    monkeypatch.setattr(orchestrator, "_native_chat_model", lambda p: Boom())
    nxt, calls, failure, _ = orchestrator._decide_native(_native_state(), "gemini-2.5-flash")
    assert nxt is None and calls == [] and failure is not None


def test_native_router_sees_prior_conversation_but_the_structured_one_does_not():
    """The asymmetry is deliberate, and both halves are load-bearing.

    NATIVE: prior_context used to be rendered only in the composer prompt, so this
    router saw "Conversation so far: (none)" every turn. "now add those to my
    calendar" arrived as a pronoun with no antecedent and it called nothing, while
    the composer — which did have the history — could only ask what "those" meant.

    STRUCTURED: this prompt can choose `finish`, and the native one cannot. Measured
    with prior turns visible here, "what did I just tell you my name was?" routes
    straight to finish and the user gets no answer. The explicit "Answers composed so
    far: 0" line does not prevent it.
    """
    from models import IntentPayload
    prior = [
        Message(source="user", content="list the start time for each game", step=0),
        Message(source="assistant", content="Monday, August 17: Christian Brothers at 5:30 PM", step=0),
    ]
    state = {
        "intent": IntentPayload(intent="calendar_add", confidence=0.9, source="t",
                                raw_input="now add those to my calendar"),
        "messages": [], "prior_context": prior, "user_id": "u1",
        "documents": "", "document_images": [], "visited": [], "step": 0,
    }
    assert "Christian Brothers" in orchestrator._build_native_prompt(state)
    assert "Christian Brothers" not in orchestrator._build_prompt(state)


def test_native_router_builds_langchain_image_parts_not_anthropic_ones():
    """The native router passes content to a LangChain chat model, which rejects
    Anthropic-shaped image blocks with "Unrecognized message part type: image."

    It was using the anthropic shape, so EVERY turn with an attachment raised,
    fell back to the structured router, and composed an offer instead of proposing
    a plan — leaving the user to confirm a plan that had never been stored. Text-only
    turns were unaffected, which is what made it look like context loss.
    """
    img = [{"media_type": "image/jpeg", "data": "QUJD", "filename": "s.jpg"}]
    parts = orchestrator._as_content_parts("hello", img, "langchain")
    assert parts[0] == {"type": "text", "text": "hello"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    # The shape that fails:
    assert all(p.get("type") != "image" for p in parts)


def test_no_images_still_returns_a_bare_string_for_every_provider():
    """The guard that keeps image support off the hot path: an attachment-free turn
    must produce exactly what it always did."""
    for provider in ("langchain", "anthropic", "gemini", "unknown"):
        assert orchestrator._as_content_parts("hi", [], provider) == "hi"


def test_routing_literal_matches_the_catalog_exactly():
    """The FOURTH place a tool name lives.

    Doc 29 says the catalog rework ended the four-places drift — registry,
    RoutingDecision Literal, ROUTING_JSON_SCHEMA enum, prompt prose — by making
    the catalog the single declaration. It removed three of the four. This
    Literal stayed hand-maintained, and on 2026-08-10 four briefing tools were
    added to the catalog and the dispatcher, the router happily selected
    get_briefing_preferences, and re-validating its own decision then raised
    ValidationError. Chat returned "No reply produced (status: error)" for every
    question that routed to one — a runtime failure, in production, from a
    mismatch that is knowable at import time.

    The existing coverage test compared the catalog to DISPATCHABLE_TOOLS only,
    so it passed throughout.
    """
    import typing

    from models import RoutingDecision
    from tools.catalog import CATALOG_TOOL_NAMES

    annotation = RoutingDecision.model_fields["tool_name"].annotation
    allowed = {
        value
        for arg in typing.get_args(annotation)
        for value in typing.get_args(arg)
        if isinstance(value, str)
    }
    assert allowed == set(CATALOG_TOOL_NAMES), (
        "RoutingDecision.tool_name and the catalog disagree; "
        f"only in Literal: {allowed - set(CATALOG_TOOL_NAMES)}, "
        f"only in catalog: {set(CATALOG_TOOL_NAMES) - allowed}"
    )


def test_prompt_tool_descriptions_come_from_the_catalog():
    """The prompt's per-tool text is GENERATED, not written twice.

    Each tool used to have two independent descriptions — its catalog docstring,
    which bind_tools sends on the native path, and hand-written prose in
    _build_prompt. That is the duplication doc 29 blames for the calendar tools
    going stale, and it recurred on 2026-08-10 when four briefing tools reached
    the catalog and never reached the prompt.
    """
    import orchestrator
    from tools.catalog import ALL_TOOLS

    block = orchestrator._tool_catalogue_block()
    for tool in ALL_TOOLS:
        assert f"* {tool.name}" in block, f"{tool.name} missing from the prompt"
        # A distinctive slice of the real docstring, so paraphrasing it back into
        # the prompt by hand would fail here rather than drift quietly.
        head = " ".join((tool.description or "").split())[:40]
        assert head in block, f"{tool.name}'s prompt text is not its docstring"


def test_write_tools_are_marked_in_the_prompt():
    """The model should see which options change something before it picks one."""
    import orchestrator
    from tools.catalog import WRITE_TOOLS

    block = orchestrator._tool_catalogue_block()
    for name in WRITE_TOOLS:
        line = next(l for l in block.splitlines() if l.strip().startswith(f"* {name}"))
        assert "[CHANGES DATA]" in line, f"{name} is not marked as mutating"


def test_native_prompt_tells_the_router_what_to_do_when_nothing_fits():
    """"can you add baseball cards?" routed to query_knowledge_base because no
    write tool existed and nothing told the router that calling nothing was the
    better move. A near-miss tool answers a question nobody asked."""
    import orchestrator
    from models import IntentPayload

    state = {
        "intent": IntentPayload(intent="general", raw_input="hello",
                                confidence=0.9, source="test"),
        "messages": [], "prior_context": [], "user_id": orchestrator.SANDBOX_USER_ID,
    }
    prompt = orchestrator._build_native_prompt(state)
    assert "no tool can do it, call NOTHING" in prompt
    assert "Never promise an action you did not call a tool for" in prompt
