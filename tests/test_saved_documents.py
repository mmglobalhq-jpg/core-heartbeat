"""Saved-document guard: the main chat points to Knowledge chat instead of improvising."""
import asyncio

import orchestrator
from models import IntentPayload, RoutingDecision, ToolArgs
from tools.saved_documents import match_title, pointer_reply, saved_document_reference

LIBRARY = [
    {"title": "North America Securitized Products Research - September 11, 2026"},
    {"title": "North America Securitized Products Research - August 28, 2026"},
    {"title": "Data Center ABS: Finding Value Amidst A Credit Market Sell-off"},
]
lib = lambda _uid: LIBRARY  # noqa: E731


def test_matches_a_library_title_and_picks_the_right_issue_by_date():
    assert saved_document_reference("summarize the August 28 securitized products report", "u", fetch=lib) \
        == "North America Securitized Products Research - August 28, 2026"


def test_publisher_or_explicit_reference_without_a_title_match():
    assert saved_document_reference("what did the Morgan Stanley note recommend?", "u", fetch=lambda _: []) == ""
    assert saved_document_reference("what's in my saved research about CLOs?", "u", fetch=lambda _: []) == ""


def test_leaves_ordinary_turns_and_reit_reports_alone():
    calls = []
    fetch = lambda uid: calls.append(uid) or LIBRARY  # noqa: E731
    for text in ("how do I roast a chicken?", "what is 15% of 2,340?", "What is the latest ARR report about?",
                 "summarize Orchid Island's latest report", "hi"):
        assert saved_document_reference(text, "u", fetch=fetch) is None, text
    # the library is only fetched for report-shaped messages
    assert calls == []


def test_one_shared_word_is_not_a_match():
    assert match_title("give me a data summary", LIBRARY) is None


def _state(raw: str) -> dict:
    return {
        "intent": IntentPayload(intent="chat", confidence=0.9, raw_input=raw, source="t"),
        "user_id": "u", "messages": [], "prior_context": [], "visited": [], "step": 0,
        "documents": "", "document_images": [],
    }


def test_supervisor_sets_the_pointer_only_when_no_tool_was_chosen(monkeypatch):
    monkeypatch.setattr(orchestrator, "saved_document_reference", lambda text, uid: "Doc X")
    out = orchestrator._finish_routing(_state("summarize doc x report"), 0, RoutingDecision(next_node="local_llm"), orchestrator.TokenUsage(), [])
    assert out["canned_reply"] == pointer_reply("Doc X")

    out = orchestrator._finish_routing(
        _state("search the web for the Doc X report"), 0,
        RoutingDecision(next_node="tool_execution", tool_name="search_web", tool_args=ToolArgs(query="q")),
        orchestrator.TokenUsage(), [],
    )
    assert out["canned_reply"] is None


def test_local_llm_emits_the_pointer_without_a_model_call(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no model call expected")
    monkeypatch.setattr(orchestrator, "generate_cloud", boom)
    monkeypatch.setattr(orchestrator, "generate_local", boom)
    out = asyncio.run(orchestrator.local_llm({**_state("x"), "canned_reply": "go to Knowledge"}))
    assert out["messages"][0].content == "go to Knowledge"
