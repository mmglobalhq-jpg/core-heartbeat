"""End-to-end: the Supervisor tool loop runs a REIT tool and surfaces it.

A scripted supervisor client (no routing network) emits a real get_latest_reit_report
call; a MockTransport stands in for the ARR engine's reader-contract RPCs. We verify the
graph routes into tool_execution, runs the tool globally (user_id threaded but not
leaked into args), and surfaces the call on both the non-streaming outcome and the SSE
``tool_call`` event.
"""
import asyncio
import json

import httpx

import orchestrator
import services.storage_sync as storage_sync
import tools.reit_research as reit
from models import IntentPayload, RoutingDecision, ToolArgs

ARR_ID = "arr:aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
_SUMMARY = {"report_id": ARR_ID, "issuer_code": "ARR", "issuer_name": "ARMOUR Residential REIT",
            "portfolio_as_of_date": "2026-05-31", "publication_date": "2026-06-12",
            "title": "ARR adds $466mm in May", "version": 1, "status": "completed"}
_DETAIL = {**_SUMMARY, "markdown": "# Exec summary\n\nGrew $466mm."}


def _rpc(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    body = json.loads(request.content or b"{}")
    if path.endswith("/rpc/reit_research_list_reports_v1"):
        rows = [_SUMMARY] if (body.get("p_issuer_code") or "").upper() == "ARR" else []
        return httpx.Response(200, json=rows)
    if path.endswith("/rpc/reit_research_get_report_v1"):
        rows = [_DETAIL] if body.get("p_report_id") == ARR_ID else []
        return httpx.Response(200, json=rows)
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
    monkeypatch.setattr(reit, "_transport", httpx.MockTransport(_rpc))
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
    assert f"Report ID: {ARR_ID}" in tool_msgs[0].content
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
    assert f"Report ID: {ARR_ID}" in call["result"]
