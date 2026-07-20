"""Supervisor routing for the REIT tools + schema parity.

No network: a fake routing client returns a scripted RoutingDecision. Verifies that
clear REIT questions route to the dedicated tools, that the forced generic-KB
retrieval does NOT preempt a clear REIT question (while still firing for unrelated
questions), that a REIT tool result routes on to local_llm to compose, and that the
Pydantic model + provider JSON schema stay aligned.
"""
import orchestrator
from models import IntentPayload, Message, RoutingDecision, ToolArgs
from orchestrator import supervisor


class _Resp:
    def __init__(self, parsed):
        self.parsed = parsed
        self.text = None
        self.usage_metadata = None


class _Client:
    def __init__(self, decision):
        class _Models:
            def generate_content(self, model, contents, config):
                return _Resp(decision)

        self.models = _Models()


def _install(monkeypatch, decision):
    monkeypatch.setattr(orchestrator, "get_client", lambda *a, **k: _Client(decision))


def _state(raw="x", messages=None, visited=None):
    return {
        "intent": IntentPayload(intent="chat", confidence=0.9, raw_input=raw, source="cli"),
        "messages": messages or [],
        "usage": None,
        "visited": visited or [],
        "step": 0,
        "next": "",
        "status": "",
    }


# --- routing to the REIT tools ----------------------------------------------

def test_latest_wording_routes_to_get_latest_reit_report(monkeypatch):
    _install(monkeypatch, RoutingDecision(
        next_node="tool_execution", tool_name="get_latest_reit_report",
        tool_args=ToolArgs(reit_symbol="ARR"),
    ))
    update = supervisor(_state(raw="show me ARR's latest report"))
    assert update["next"] == "tool_execution"
    assert update["tool_request"]["name"] == "get_latest_reit_report"
    assert update["tool_request"]["args"] == {"reit_symbol": "ARR"}


def test_list_wording_routes_to_list_reit_reports(monkeypatch):
    _install(monkeypatch, RoutingDecision(
        next_node="tool_execution", tool_name="list_reit_reports",
        tool_args=ToolArgs(reit_symbol="ARR"),
    ))
    update = supervisor(_state(raw="which ARR reports are available?"))
    assert update["tool_request"]["name"] == "list_reit_reports"


# --- forced-KB retrieval must not preempt a clear REIT question --------------

def test_forced_kb_does_not_preempt_reit_question(monkeypatch):
    # KB configured + model routes straight to local_llm: normally the retrieve-first
    # guard would force a query_knowledge_base call. A clear REIT question must be
    # exempt so it reaches the REIT tools instead.
    monkeypatch.setenv("GRAPHRAG_SERVICE_URL", "http://kb")
    _install(monkeypatch, RoutingDecision(next_node="local_llm"))
    update = supervisor(_state(raw="What changed in ARR's latest report?"))
    assert update["next"] == "local_llm"
    assert update.get("tool_request") is None  # NOT forced into query_knowledge_base


def test_forced_kb_still_fires_for_unrelated_question(monkeypatch):
    # The exemption is narrow: a non-REIT substantive question still gets forced KB.
    monkeypatch.setenv("GRAPHRAG_SERVICE_URL", "http://kb")
    _install(monkeypatch, RoutingDecision(next_node="local_llm"))
    update = supervisor(_state(raw="how do I roast a chicken?"))
    assert update["next"] == "tool_execution"
    assert update["tool_request"]["name"] == "query_knowledge_base"


def test_reit_tool_result_routes_to_local_llm_not_finish(monkeypatch):
    # After a REIT tool result is in the history, composing the answer (local_llm) is
    # the next hop — the supervisor honors that rather than finishing on raw data.
    _install(monkeypatch, RoutingDecision(next_node="local_llm"))
    s = _state(
        raw="summarize ARR's latest report",
        messages=[Message(source="tool_execution", content="[tool:get_latest_reit_report] Issuer: ARMOUR Residential REIT (ARR)\nReport ID: rep-a", step=0)],
        visited=["tool_execution"],
    )
    update = supervisor(s)
    assert update["next"] == "local_llm"


# --- ORC (Orchid) routing ---------------------------------------------------

def test_orc_latest_wording_routes_to_get_latest(monkeypatch):
    _install(monkeypatch, RoutingDecision(
        next_node="tool_execution", tool_name="get_latest_reit_report",
        tool_args=ToolArgs(reit_symbol="ORC"),
    ))
    update = supervisor(_state(raw="What is the latest ORC report?"))
    assert update["tool_request"]["name"] == "get_latest_reit_report"
    assert update["tool_request"]["args"] == {"reit_symbol": "ORC"}


def test_orchid_wording_routes_to_list(monkeypatch):
    _install(monkeypatch, RoutingDecision(
        next_node="tool_execution", tool_name="list_reit_reports",
        tool_args=ToolArgs(reit_symbol="Orchid Island"),
    ))
    update = supervisor(_state(raw="Show me Orchid Island's reports."))
    assert update["tool_request"]["name"] == "list_reit_reports"


def test_orc_namespaced_id_routes_to_get_report(monkeypatch):
    rid = "orc:aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    _install(monkeypatch, RoutingDecision(
        next_node="tool_execution", tool_name="get_reit_report",
        tool_args=ToolArgs(report_id=rid),
    ))
    update = supervisor(_state(raw=f"Get report {rid}."))
    assert update["tool_request"]["name"] == "get_reit_report"
    assert update["tool_request"]["args"] == {"report_id": rid}


def test_orchid_question_exempt_from_forced_kb(monkeypatch):
    # An Orchid/ORC reference must be exempt from the forced generic-KB retrieval.
    monkeypatch.setenv("GRAPHRAG_SERVICE_URL", "http://kb")
    _install(monkeypatch, RoutingDecision(next_node="local_llm"))
    for q in ("What changed in Orchid Island Capital's RMBS portfolio?",
              "latest ORC report"):
        update = supervisor(_state(raw=q))
        assert update["next"] == "local_llm"
        assert update.get("tool_request") is None


def test_looks_like_reit_reference_matches_orc_terms():
    from tools.reit_research import looks_like_reit_reference

    for q in ("latest ORC report", "Orchid Island Capital", "what did ARMOUR do"):
        assert looks_like_reit_reference(q)
    assert not looks_like_reit_reference("how do I roast a chicken?")


# --- schema parity ----------------------------------------------------------

REIT_NAMES = {"list_reit_issuers", "list_reit_reports", "get_reit_report", "get_latest_reit_report"}


def test_registry_matches_tool_names():
    from tools.reit_research import REIT_TOOL_REGISTRY
    assert set(REIT_TOOL_REGISTRY) == REIT_NAMES


def test_pydantic_and_json_schema_aligned():
    import typing

    # Pydantic RoutingDecision.tool_name Literal includes every REIT name.
    ann = RoutingDecision.model_fields["tool_name"].annotation
    literal_values: set = set()
    for arg in typing.get_args(ann):
        if arg is type(None):
            continue
        literal_values |= set(typing.get_args(arg))
    assert REIT_NAMES <= literal_values

    # Provider JSON schema enum + tool_args properties include them too.
    enum = {x for x in orchestrator.ROUTING_JSON_SCHEMA["properties"]["tool_name"]["enum"] if x}
    assert REIT_NAMES <= enum
    props = orchestrator.ROUTING_JSON_SCHEMA["properties"]["tool_args"]["properties"]
    assert {"reit_symbol", "report_id", "limit"} <= set(props)

    # ToolArgs carries the REIT fields.
    from models import ToolArgs as TA
    assert {"reit_symbol", "report_id", "limit"} <= set(TA.model_fields)
