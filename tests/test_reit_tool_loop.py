"""End-to-end: the Supervisor tool loop runs a REIT tool and surfaces it.

A scripted supervisor client (no routing network) emits a real get_latest_reit_report
call; a MockTransport stands in for the ARR engine's PostgREST. We verify the graph
routes into tool_execution, runs the tool globally (user_id threaded but not leaked
into args), and surfaces the call on both the non-streaming outcome and the SSE
``tool_call`` event.
"""
import asyncio

import httpx

import orchestrator
import services.storage_sync as storage_sync
import tools.reit_research as reit
from models import IntentPayload, RoutingDecision, ToolArgs

REPORTS = [{"id": "rep-a", "issuer_code": "ARR", "portfolio_as_of_date": "2026-05-31", "current_version_id": "v-a", "status": "completed"}]
VERSIONS = [{"id": "v-a", "headline": "ARR adds $466mm in May", "version": 1, "source_document_id": "d-a", "status": "completed", "markdown": "# Exec summary\n\nGrew $466mm."}]
DOCS = [{"id": "d-a", "publication_date": "2026-06-12"}]


def _postgrest(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("/reit_arr_reports"):
        return httpx.Response(200, json=REPORTS)
    if path.endswith("/reit_arr_report_versions"):
        return httpx.Response(200, json=VERSIONS)
    if path.endswith("/reit_arr_source_documents"):
        return httpx.Response(200, json=DOCS)
    return httpx.Response(404, json=[])


class _Resp:
    def __init__(self, decision):
        self.parsed = decision
        self.text = None
        self.usage_metadata = None


class _ScriptedClient:
    def __init__(self, decisions):
        it = iter(decisions)

        class _Models:
            def generate_content(self, model, contents, config):
                try:
                    return _Resp(next(it))
                except StopIteration:
                    return _Resp(RoutingDecision(next_node="finish"))

        self.models = _Models()


def _setup(monkeypatch, tmp_path):
    # REIT service credentials + mock transport.
    monkeypatch.setenv("REITS_SUPABASE_URL", "http://reits.local")
    monkeypatch.setenv("REITS_SUPABASE_SERVICE_ROLE_KEY", "svc")
    monkeypatch.setattr(reit, "_transport", httpx.MockTransport(_postgrest))
    # Keep any vault sync off real directories.
    monkeypatch.setenv(storage_sync.VAULT_SYNC_ROOT_ENV, str(tmp_path / "vaults"))
    monkeypatch.setenv(storage_sync.VAULT_MOCK_ROOT_ENV, str(tmp_path / "mock"))
    client = _ScriptedClient([
        RoutingDecision(
            next_node="tool_execution", tool_name="get_latest_reit_report",
            tool_args=ToolArgs(reit_symbol="ARR"),
        ),
        RoutingDecision(next_node="finish"),
    ])
    monkeypatch.setattr(orchestrator, "get_client", lambda *a, **k: client)


def _payload():
    return IntentPayload(
        intent="chat", confidence=0.95, raw_input="show me ARR's latest report", source="cli",
    )


def test_reit_tool_runs_and_surfaces_on_outcome(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    outcome = asyncio.run(orchestrator.run(_payload()))
    assert outcome.status == "completed"
    assert "tool_execution" in outcome.nodes_executed
    tool_msgs = [m for m in outcome.messages if m.source == "tool_execution"]
    assert tool_msgs and tool_msgs[0].content.startswith("[tool:get_latest_reit_report]")
    assert "Report ID: rep-a" in tool_msgs[0].content
    assert "# Exec summary" in tool_msgs[0].content


def test_reit_tool_call_surfaces_as_stream_event(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)

    async def _run():
        return [ev async for ev in orchestrator.astream_run(_payload())]

    events = asyncio.run(_run())
    tool_calls = [e["tool_call"] for e in events if "tool_call" in e]
    assert len(tool_calls) == 1
    call = tool_calls[0]
    assert call["name"] == "get_latest_reit_report"
    assert call["args"] == {"reit_symbol": "ARR"}  # user_id never leaks into args
    assert "Report ID: rep-a" in call["result"]
