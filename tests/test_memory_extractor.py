"""Tests for the silent memory_extractor node (feature 008).

Covers structured extraction parsing, the path-safe upsert into the user's vault
profile (isolation preserved), and the node's silent best-effort behavior
(writes a high-confidence durable preference; skips none/low-confidence/no-reply;
never emits messages/usage or breaks the run). No network: a fake provider client
is injected at the get_client seam, and the vault root is redirected to tmp_path.
"""

import asyncio

import pytest

import orchestrator
import services.storage_sync as storage_sync
from models import IntentPayload, MemoryExtraction, Message, RoutingDecision
from orchestrator import (
    PREFERENCES_FILE,
    _latest_local_reply,
    _record_preference,
    extract_user_preference,
    memory_extractor,
)
from tools.user_vault import read_note


# --- fakes (no network) -----------------------------------------------------

class _MemResp:
    def __init__(self, parsed=None, text=None):
        self.parsed = parsed
        self.text = text
        self.usage_metadata = None


class _MemClient:
    """Gemini-shaped fake whose generate_content returns a fixed response."""

    def __init__(self, parsed=None, text=None):
        p, t = parsed, text

        class _Models:
            def generate_content(self, model, contents, config):
                return _MemResp(p, t)

        self.models = _Models()


@pytest.fixture(autouse=True)
def _vault_root(monkeypatch, tmp_path):
    monkeypatch.setenv(storage_sync.VAULT_SYNC_ROOT_ENV, str(tmp_path / "vaults"))
    return tmp_path / "vaults"


def _pref(ptype="favorite", key="favorite_language", value="Python", conf=0.9):
    return MemoryExtraction(
        preference_type=ptype, key_insight=key, value=value, confidence_score=conf
    )


def _state(reply="I'll remember you love Python.", user_id="user-abc", raw="I love Python"):
    return {
        "intent": IntentPayload(intent="chat", confidence=0.95, raw_input=raw, source="cli"),
        "user_id": user_id,
        "messages": [Message(source="local_llm", content=reply, step=1)] if reply else [],
        "usage": None,
        "visited": [],
        "step": 1,
        "next": "",
        "status": "",
        "tool_request": None,
    }


# --- extraction parsing -----------------------------------------------------

def test_extract_parses_gemini_structured_output():
    client = _MemClient(parsed=_pref())
    out = extract_user_preference("I love Python", "noted", client, "gemini-2.5-flash")
    assert isinstance(out, MemoryExtraction)
    assert out.value == "Python" and out.preference_type == "favorite"


def test_extract_parses_from_json_text():
    client = _MemClient(text='{"preference_type":"tool_setting","key_insight":"editor","value":"vim","confidence_score":0.8}')
    out = extract_user_preference("use vim", "ok", client, "gemini-2.5-flash")
    assert out.value == "vim" and out.preference_type == "tool_setting"


def test_extract_returns_none_on_wrong_shape():
    # A routing-decision-shaped response (as the supervisor fakes return) is not a
    # MemoryExtraction -> graceful None, never raises.
    client = _MemClient(parsed=RoutingDecision(next_node="finish"))
    assert extract_user_preference("hi", "hello", client, "gemini-2.5-flash") is None


# --- path-safe upsert into the vault profile --------------------------------

def test_record_writes_profile_in_user_dir(_vault_root):
    _record_preference("user-abc", _pref())
    content = read_note("user-abc", PREFERENCES_FILE)
    assert content.startswith("# User Preferences")
    assert "**favorite** | favorite_language: Python (confidence 0.90)" in content
    assert (_vault_root / "user-abc" / PREFERENCES_FILE).is_file()


def test_record_upserts_same_key(_vault_root):
    _record_preference("user-abc", _pref(value="Python"))
    _record_preference("user-abc", _pref(value="Rust"))  # same type+key -> replace
    content = read_note("user-abc", PREFERENCES_FILE)
    assert "Rust" in content
    assert "Python" not in content
    assert content.count("favorite_language") == 1  # not duplicated


def test_record_appends_distinct_keys(_vault_root):
    _record_preference("user-abc", _pref(key="favorite_language", value="Python"))
    _record_preference("user-abc", _pref(ptype="tool_setting", key="editor", value="vim"))
    content = read_note("user-abc", PREFERENCES_FILE)
    assert "favorite_language" in content and "editor" in content


def test_record_is_user_isolated(_vault_root):
    _record_preference("user-abc", _pref(value="Python"))
    # A different user's profile is untouched / independent.
    assert not (_vault_root / "user-xyz" / PREFERENCES_FILE).exists()
    _record_preference("user-xyz", _pref(value="Go"))
    assert "Go" in read_note("user-xyz", PREFERENCES_FILE)
    assert "Go" not in read_note("user-abc", PREFERENCES_FILE)


# --- node behavior ----------------------------------------------------------

def test_node_writes_high_confidence_preference(monkeypatch, _vault_root):
    monkeypatch.setattr(orchestrator, "get_client", lambda *a, **k: _MemClient(parsed=_pref(conf=0.9)))
    update = asyncio.run(memory_extractor(_state(user_id="user-abc")))
    assert update == {}  # silent: no messages/usage/visited
    assert "Python" in read_note("user-abc", PREFERENCES_FILE)


def test_node_skips_preference_type_none(monkeypatch, _vault_root):
    monkeypatch.setattr(
        orchestrator, "get_client",
        lambda *a, **k: _MemClient(parsed=_pref(ptype="none", key="n", value="n", conf=0.0)),
    )
    asyncio.run(memory_extractor(_state(user_id="user-abc")))
    assert not (_vault_root / "user-abc" / PREFERENCES_FILE).exists()


def test_node_skips_low_confidence(monkeypatch, _vault_root):
    monkeypatch.setattr(orchestrator, "get_client", lambda *a, **k: _MemClient(parsed=_pref(conf=0.3)))
    asyncio.run(memory_extractor(_state(user_id="user-abc")))
    assert not (_vault_root / "user-abc" / PREFERENCES_FILE).exists()


def test_node_skips_when_no_local_reply(monkeypatch, _vault_root):
    called = {"n": 0}

    def _get_client(*a, **k):
        called["n"] += 1
        return _MemClient(parsed=_pref())

    monkeypatch.setattr(orchestrator, "get_client", _get_client)
    asyncio.run(memory_extractor(_state(reply="", user_id="user-abc")))  # no assistant reply
    assert called["n"] == 0  # short-circuits before any model call
    assert not (_vault_root / "user-abc" / PREFERENCES_FILE).exists()


def test_node_skips_local_failure_notice(monkeypatch, _vault_root):
    monkeypatch.setattr(orchestrator, "get_client", lambda *a, **k: _MemClient(parsed=_pref()))
    state = _state(reply="local inference failure: unreachable", user_id="user-abc")
    asyncio.run(memory_extractor(state))
    assert not (_vault_root / "user-abc" / PREFERENCES_FILE).exists()


def test_node_no_client_is_silent_noop(monkeypatch, _vault_root):
    monkeypatch.setattr(orchestrator, "get_client", lambda *a, **k: None)
    update = asyncio.run(memory_extractor(_state(user_id="user-abc")))
    assert update == {}
    assert not (_vault_root / "user-abc" / PREFERENCES_FILE).exists()


def test_node_never_raises_on_bad_extraction(monkeypatch, _vault_root):
    # Wrong-shaped model output -> extraction None -> silent no-op, no exception.
    monkeypatch.setattr(orchestrator, "get_client", lambda *a, **k: _MemClient(parsed=RoutingDecision(next_node="finish")))
    assert asyncio.run(memory_extractor(_state(user_id="user-abc"))) == {}


# --- helper + graph wiring --------------------------------------------------

def test_latest_local_reply_picks_last_local_llm():
    msgs = [
        Message(source="local_llm", content="first", step=1),
        Message(source="tool_execution", content="[stub] tool executed", step=2),
        Message(source="local_llm", content="second", step=3),
    ]
    assert _latest_local_reply(msgs) == "second"


def test_graph_routes_finish_through_memory_extractor():
    g = orchestrator.graph.get_graph()
    assert "memory_extractor" in g.nodes
    # memory_extractor terminates the graph.
    edges = [(e.source, e.target) for e in g.edges]
    assert ("memory_extractor", "__end__") in edges
