"""Conformance tests binding the routing vocabulary to what is actually dispatchable.

The Supervisor's tool vocabulary was, historically, written out by hand in four
independent places: ``ROUTING_JSON_SCHEMA``, ``RoutingDecision.tool_name``,
``ToolArgs``, and the prose catalogue inside the Supervisor prompt. Nothing tied
them together, so they drifted — the four Google Calendar tools were added to the
Pydantic model and the prompt but never to ``ROUTING_JSON_SCHEMA``.

The consequence was provider-dependent and silent. The default Gemini path passes
``RoutingDecision`` as ``response_schema`` and was unaffected, so calendar routing
worked in production. The OpenAI path sends ``ROUTING_JSON_SCHEMA`` with
``strict: true``, which hard-enforces the enum: the model could not emit a calendar
tool name at all, and ``additionalProperties: False`` rejected every calendar
argument. Anthropic sends the same schema as ``input_schema``, steering the model
away from tools that do in fact exist.

``ROUTING_JSON_SCHEMA`` is now derived from the Pydantic models, so those two can no
longer disagree. What no derivation can catch is a tool added to a *registry* and
forgotten in the model — that is what these tests are for. They are deliberately
equality assertions, not subset ones: a name the model may emit but nothing can
execute is as much a defect as the reverse.
"""

import orchestrator
from models import RoutingDecision, ToolArgs
from tools.daily_briefing import BRIEFING_TOOL_REGISTRY
from tools.google_calendar import CALENDAR_TOOL_REGISTRY
from tools.graphrag import GRAPHRAG_TOOL_REGISTRY
from tools.reit_research import REIT_TOOL_REGISTRY
from tools.user_vault import USER_VAULT_TOOLS
from tools.web_tools import WEB_TOOL_REGISTRY


def dispatchable_tools() -> set[str]:
    """Every tool name ``tool_execution`` can actually run (orchestrator.py)."""
    return (
        {t.name for t in USER_VAULT_TOOLS}
        | set(GRAPHRAG_TOOL_REGISTRY)
        | set(CALENDAR_TOOL_REGISTRY)
        | set(REIT_TOOL_REGISTRY)
        | set(BRIEFING_TOOL_REGISTRY)
        | set(WEB_TOOL_REGISTRY)
    )


def test_this_files_mirror_matches_the_orchestrator():
    """This helper is itself a hand-maintained copy of orchestrator.py's union —
    a FIFTH place a tool name has to be written.

    That is not theoretical. On 2026-08-10 four briefing tools were added to the
    catalog, the dispatcher and the model, and both equality tests below kept
    passing because this helper had not been updated either: 14 == 14 on both
    sides while production returned "No reply produced (status: error)". A guard
    with its own private copy of the truth can go blind exactly when it matters.
    """
    assert dispatchable_tools() == set(orchestrator.DISPATCHABLE_TOOLS)


def schema_tool_names() -> set[str]:
    enum = orchestrator.ROUTING_JSON_SCHEMA["properties"]["tool_name"]["enum"]
    return {n for n in enum if n is not None}


def model_tool_names() -> set[str]:
    values = orchestrator._literal_values(RoutingDecision.model_fields["tool_name"].annotation)
    return {v for v in values if isinstance(v, str)}


def test_wire_schema_matches_dispatchable_tools():
    """The enum sent to OpenAI/Anthropic must be exactly what we can execute.

    This is the assertion that would have caught the calendar regression.
    """
    assert schema_tool_names() == dispatchable_tools()


def test_validator_matches_dispatchable_tools():
    """RoutingDecision must accept exactly the dispatchable tools.

    A name missing here is rejected as ``invalid_output`` after the model emits it;
    an extra name routes to a tool_execution branch that matches no registry.
    """
    assert model_tool_names() == dispatchable_tools()


def test_wire_schema_and_validator_agree():
    """Redundant while the schema is derived — it fails loudly if anyone
    re-hardcodes ``ROUTING_JSON_SCHEMA`` back to a literal."""
    assert schema_tool_names() == model_tool_names()


def test_calendar_tools_present():
    """Names the regression specifically dropped, asserted explicitly so the
    failure message points at calendar rather than a set diff."""
    for name in ("list_calendar_events", "create_calendar_event",
                 "update_calendar_event", "delete_calendar_event"):
        assert name in schema_tool_names(), f"{name} missing from ROUTING_JSON_SCHEMA"
        assert name in model_tool_names(), f"{name} missing from RoutingDecision"


def test_tool_args_cover_every_argument_the_schema_advertises():
    """``additionalProperties: False`` makes any ToolArgs field absent from the wire
    schema an unsendable argument — which is how calendar's start/end/summary were
    silently unreachable on the strict OpenAI path."""
    schema_args = set(orchestrator.ROUTING_JSON_SCHEMA["properties"]["tool_args"]["properties"])
    assert schema_args == set(ToolArgs.model_fields)


def test_calendar_arguments_are_sendable():
    schema_args = set(orchestrator.ROUTING_JSON_SCHEMA["properties"]["tool_args"]["properties"])
    for arg in ("summary", "start", "end", "event_id",
                "time_min", "time_max", "location", "description", "max_results"):
        assert arg in schema_args, f"calendar argument {arg} cannot be sent"


def test_supervisor_prompt_documents_every_dispatchable_tool():
    """The prose catalogue is the model's only description of each tool. A tool the
    schema allows but the prompt never mentions is one the model won't know to use.

    Uses the real prompt builder so this tracks the live text, not a copy.
    """
    state = {
        "intent": orchestrator.IntentPayload(
            intent="general", raw_input="hello", confidence=0.9, source="test"
        ),
        "messages": [],
        "prior_context": [],
        "user_id": orchestrator.SANDBOX_USER_ID,
    }
    prompt = orchestrator._build_prompt(state)
    missing = [t for t in sorted(dispatchable_tools()) if t not in prompt]
    assert not missing, f"tools absent from the Supervisor prompt: {missing}"


def test_next_node_enum_matches_the_graph():
    assert set(orchestrator.ROUTING_JSON_SCHEMA["properties"]["next_node"]["enum"]) == {
        "local_llm",
        "tool_execution",
        "finish",
    }
