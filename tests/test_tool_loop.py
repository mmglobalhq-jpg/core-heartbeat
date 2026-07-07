"""Integration tests for the Supervisor tool-calling loop (feature 007).

A scripted supervisor client (no network) emits a real vault tool call, and we
verify the graph routes into tool_execution, runs the tool against the run's
isolated user_id, and surfaces the call both on the non-streaming outcome and as
a structured ``tool_call`` event on the SSE stream. The canonical prompt is
"Search my vault for openclaw".
"""

import asyncio

import pytest

import orchestrator
import services.storage_sync as storage_sync
from models import IntentPayload, RoutingDecision, ToolArgs


# --- scripted supervisor client (no network) --------------------------------

class _Resp:
    def __init__(self, decision):
        self.parsed = decision
        self.text = None
        self.usage_metadata = None


class _ScriptedClient:
    """Returns the given RoutingDecisions in order, then finish forever."""

    def __init__(self, decisions):
        it = iter(decisions)

        class _Models:
            def generate_content(self, model, contents, config):
                try:
                    return _Resp(next(it))
                except StopIteration:
                    return _Resp(RoutingDecision(next_node="finish"))

        self.models = _Models()


def _search_then_finish(query="openclaw"):
    return [
        RoutingDecision(
            next_node="tool_execution",
            tool_name="search_user_vault",
            tool_args=ToolArgs(query=query),
        ),
        RoutingDecision(next_node="finish"),
    ]


def _install_supervisor(monkeypatch, decisions):
    # ONE memoized client so the scripted iterator advances across supervisor
    # turns (a fresh client per call would restart at the tool call and loop to
    # the step bound).
    client = _ScriptedClient(decisions)
    monkeypatch.setattr(orchestrator, "get_client", lambda *a, **k: client)


def _setup_vault(monkeypatch, tmp_path, body):
    """Seed the sandbox user's vault with a note containing `body`.

    Writes to BOTH the sync root (used by run(), which does not sync) and the mock
    root (astream_run first syncs, clearing then repopulating the sync dir from the
    mock). Redirects both roots into tmp_path so nothing touches real dirs.
    """
    sync_root = tmp_path / "vaults"
    mock_root = tmp_path / "mock"
    monkeypatch.setenv(storage_sync.VAULT_SYNC_ROOT_ENV, str(sync_root))
    monkeypatch.setenv(storage_sync.VAULT_MOCK_ROOT_ENV, str(mock_root))
    user = orchestrator.SANDBOX_USER_ID
    for root in (sync_root, mock_root):
        note = root / user / "notes.md"
        note.parent.mkdir(parents=True, exist_ok=True)
        note.write_text(body, encoding="utf-8")


def _payload():
    return IntentPayload(
        intent="search_vault",
        confidence=0.95,
        raw_input="Search my vault for openclaw",
        source="cli",
    )


# --- streaming: the tool call surfaces as a structured event ----------------

def test_search_prompt_triggers_tool_turn_streaming(monkeypatch, tmp_path):
    _setup_vault(monkeypatch, tmp_path, "Daily log\nI love openclaw the game")
    _install_supervisor(monkeypatch, _search_then_finish())

    async def _run():
        return [ev async for ev in orchestrator.astream_run(_payload())]

    events = asyncio.run(_run())

    tool_calls = [e["tool_call"] for e in events if "tool_call" in e]
    assert len(tool_calls) == 1, f"expected exactly one tool turn, got {events}"
    call = tool_calls[0]
    assert call["name"] == "search_user_vault"
    assert call["args"] == {"query": "openclaw"}       # user_id never leaks into args
    assert "openclaw" in call["result"]                # the vault was actually searched
    assert events[-1] == {"status": "completed"}       # loop closed cleanly


def test_streaming_tool_result_is_not_an_assistant_token(monkeypatch, tmp_path):
    # The raw "[tool:...]" internal result must NOT be streamed as a chat token;
    # it rides the structured tool_call event instead.
    _setup_vault(monkeypatch, tmp_path, "note with openclaw")
    _install_supervisor(monkeypatch, _search_then_finish())

    async def _run():
        return [ev async for ev in orchestrator.astream_run(_payload())]

    events = asyncio.run(_run())
    assert not any("token" in e for e in events)


# --- non-streaming run(): the tool actually executed ------------------------

def test_search_prompt_triggers_tool_turn_run(monkeypatch, tmp_path):
    _setup_vault(monkeypatch, tmp_path, "shopping list\nopenclaw beta notes")
    _install_supervisor(monkeypatch, _search_then_finish())

    outcome = asyncio.run(orchestrator.run(_payload()))

    assert outcome.status == "completed"
    assert "tool_execution" in outcome.nodes_executed
    tool_msgs = [m for m in outcome.messages if m.source == "tool_execution"]
    assert tool_msgs, "expected a tool_execution message in the history"
    assert tool_msgs[0].content.startswith("[tool:search_user_vault]")
    assert "openclaw" in tool_msgs[0].content


def test_tool_turn_scoped_to_run_user_id(monkeypatch, tmp_path):
    # A different user's note with the same term must never surface for the
    # sandbox run — the loop searches only the run's own vault.
    _setup_vault(monkeypatch, tmp_path, "sandbox openclaw note")
    other = tmp_path / "vaults" / "someone-else" / "secret.md"
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_text("someone-else openclaw secret", encoding="utf-8")
    _install_supervisor(monkeypatch, _search_then_finish())

    outcome = asyncio.run(orchestrator.run(_payload()))
    tool_msg = next(m for m in outcome.messages if m.source == "tool_execution")
    assert "sandbox openclaw note" in tool_msg.content
    assert "secret" not in tool_msg.content


# --- the loop returns to the model after the tool (multi-turn) --------------

def test_tool_result_feeds_back_to_model_then_finishes(monkeypatch, tmp_path):
    _setup_vault(monkeypatch, tmp_path, "openclaw appears here")
    _install_supervisor(monkeypatch, _search_then_finish())

    outcome = asyncio.run(orchestrator.run(_payload()))
    sources = [m.source for m in outcome.messages]
    # supervisor -> tool_execution -> supervisor(finish): the model saw the tool
    # result and then closed the run.
    assert sources == ["supervisor", "tool_execution", "supervisor"]
    assert outcome.steps <= orchestrator.MAX_STEPS
