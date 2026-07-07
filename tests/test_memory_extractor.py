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
    _build_local_prompt,
    _build_prompt,
    _latest_local_reply,
    _load_user_profile,
    _record_preference,
    _user_profile_block,
    extract_and_record_preference,
    extract_user_preference,
    schedule_memory_extraction,
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


@pytest.fixture(autouse=True)
def _stub_s3_writeback(monkeypatch):
    """Neutralize the real S3 write-back in every memory test (no network) and
    record the (user_id, filename) calls so tests can assert on them."""
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(orchestrator, "upload_user_file", lambda uid, fn: calls.append((uid, fn)))
    return calls


def _pref(ptype="favorite", key="favorite_language", value="Python", conf=0.9):
    return MemoryExtraction(
        preference_type=ptype, key_insight=key, value=value, confidence_score=conf
    )


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


# --- detached extract+record coroutine --------------------------------------

def _run_extract(user_id="user-abc", user_message="I love Python", reply="noted"):
    return asyncio.run(
        extract_and_record_preference(user_id, "gemini-2.5-flash", user_message, reply)
    )


def test_extract_and_record_writes_high_confidence(monkeypatch, _vault_root):
    monkeypatch.setattr(orchestrator, "get_client", lambda *a, **k: _MemClient(parsed=_pref(conf=0.9)))
    line = _run_extract()
    assert "Python" in line
    assert "Python" in read_note("user-abc", PREFERENCES_FILE)


def test_extract_and_record_skips_none(monkeypatch, _vault_root):
    monkeypatch.setattr(
        orchestrator, "get_client",
        lambda *a, **k: _MemClient(parsed=_pref(ptype="none", key="n", value="n", conf=0.0)),
    )
    assert _run_extract() is None
    assert not (_vault_root / "user-abc" / PREFERENCES_FILE).exists()


def test_extract_and_record_skips_low_confidence(monkeypatch, _vault_root):
    monkeypatch.setattr(orchestrator, "get_client", lambda *a, **k: _MemClient(parsed=_pref(conf=0.3)))
    assert _run_extract() is None
    assert not (_vault_root / "user-abc" / PREFERENCES_FILE).exists()


def test_extract_and_record_skips_empty_reply(monkeypatch, _vault_root):
    called = {"n": 0}

    def _gc(*a, **k):
        called["n"] += 1
        return _MemClient(parsed=_pref())

    monkeypatch.setattr(orchestrator, "get_client", _gc)
    out = asyncio.run(
        extract_and_record_preference("user-abc", "gemini-2.5-flash", "hi", "")
    )
    assert out is None
    assert called["n"] == 0  # short-circuits before any client/model call


def test_extract_and_record_no_client(monkeypatch, _vault_root):
    monkeypatch.setattr(orchestrator, "get_client", lambda *a, **k: None)
    assert _run_extract() is None
    assert not (_vault_root / "user-abc" / PREFERENCES_FILE).exists()


def test_extract_and_record_never_raises_on_bad_shape(monkeypatch, _vault_root):
    monkeypatch.setattr(
        orchestrator, "get_client", lambda *a, **k: _MemClient(parsed=RoutingDecision(next_node="finish"))
    )
    assert _run_extract() is None  # graceful None, no exception


def test_extract_and_record_writes_back_to_s3(monkeypatch, _vault_root, _stub_s3_writeback):
    # After the local write, the profile is mirrored to S3 (durability).
    monkeypatch.setattr(orchestrator, "get_client", lambda *a, **k: _MemClient(parsed=_pref(conf=0.9)))
    line = _run_extract(user_id="user-abc")
    assert "Python" in line
    assert _stub_s3_writeback == [("user-abc", PREFERENCES_FILE)]  # exactly one write-back


def test_write_back_failure_keeps_local_write(monkeypatch, _vault_root):
    # An S3 write-back failure must NOT undo the successful local record.
    monkeypatch.setattr(orchestrator, "get_client", lambda *a, **k: _MemClient(parsed=_pref(conf=0.9)))

    def _boom(uid, fn):
        raise RuntimeError("s3 unreachable")

    monkeypatch.setattr(orchestrator, "upload_user_file", _boom)  # overrides the autouse stub
    line = _run_extract(user_id="user-abc")
    assert "Python" in line
    assert "Python" in read_note("user-abc", PREFERENCES_FILE)  # local copy intact


def test_no_write_back_when_nothing_recorded(monkeypatch, _vault_root, _stub_s3_writeback):
    # A "none"/low-confidence extraction records nothing, so no write-back fires.
    monkeypatch.setattr(
        orchestrator, "get_client",
        lambda *a, **k: _MemClient(parsed=_pref(ptype="none", key="n", value="n", conf=0.0)),
    )
    assert _run_extract(user_id="user-abc") is None
    assert _stub_s3_writeback == []


# --- schedule_memory_extraction (detached task) -----------------------------

def _intent(raw="I love Python"):
    return IntentPayload(intent="chat", confidence=0.95, raw_input=raw, source="cli")


def test_schedule_returns_task_and_writes_when_awaited(monkeypatch, _vault_root):
    monkeypatch.setattr(orchestrator, "get_client", lambda *a, **k: _MemClient(parsed=_pref(conf=0.9)))

    async def _drive():
        task = schedule_memory_extraction("user-abc", _intent(), "you love Python")
        assert task is not None
        return await task

    line = asyncio.run(_drive())
    assert "Python" in line
    assert "Python" in read_note("user-abc", PREFERENCES_FILE)


def test_schedule_returns_none_without_reply(_vault_root):
    # No assistant reply -> nothing scheduled (no task, no write).
    assert schedule_memory_extraction("user-abc", _intent(), "") is None
    assert not (_vault_root / "user-abc" / PREFERENCES_FILE).exists()


# --- helper + graph wiring --------------------------------------------------

def test_latest_local_reply_picks_last_local_llm():
    msgs = [
        Message(source="local_llm", content="first", step=1),
        Message(source="tool_execution", content="[stub] tool executed", step=2),
        Message(source="local_llm", content="second", step=3),
    ]
    assert _latest_local_reply(msgs) == "second"


def test_graph_finish_routes_directly_to_end():
    # memory_extractor is no longer a graph node — it's a detached task.
    g = orchestrator.graph.get_graph()
    assert "memory_extractor" not in g.nodes
    edges = [(e.source, e.target) for e in g.edges]
    assert ("supervisor", "__end__") in edges  # finish terminates immediately


# --- profile injection into the prompts (read side) -------------------------

def _sup_state(user_id, raw="hello"):
    return {
        "intent": IntentPayload(intent="chat", confidence=0.9, raw_input=raw, source="cli"),
        "user_id": user_id, "messages": [], "usage": None, "visited": [],
        "step": 1, "next": "", "status": "", "tool_request": None,
    }


def test_profile_block_empty_when_no_profile(_vault_root):
    assert _user_profile_block("nobody-home") == ""


def test_profile_block_present_when_profile_exists(_vault_root):
    _record_preference("user-abc", _pref(value="Red"))
    block = _user_profile_block("user-abc")
    assert "### USER PROFILE & LONG-TERM PREFERENCES" in block
    assert "Red" in block


def test_supervisor_prompt_injects_profile(_vault_root):
    _record_preference("user-abc", _pref(value="Red"))
    prompt = _build_prompt(_sup_state("user-abc"))
    assert "### USER PROFILE & LONG-TERM PREFERENCES" in prompt
    assert "Red" in prompt
    assert "do NOT call a tool to fetch it" in prompt


def test_supervisor_prompt_omits_block_without_profile(_vault_root):
    assert "USER PROFILE" not in _build_prompt(_sup_state("nobody-home"))


def test_local_prompt_injects_profile(_vault_root):
    _record_preference("user-abc", _pref(value="Red"))
    prompt = _build_local_prompt(_sup_state("user-abc"))
    assert "### USER PROFILE & LONG-TERM PREFERENCES" in prompt
    assert "Red" in prompt


def test_load_user_profile_is_user_isolated(_vault_root):
    _record_preference("user-abc", _pref(value="Red"))
    assert "Red" in _load_user_profile("user-abc")
    assert _load_user_profile("user-xyz") == ""  # path-safe read is user-scoped
