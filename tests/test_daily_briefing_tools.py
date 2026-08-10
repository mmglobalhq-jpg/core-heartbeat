"""Chat access to the daily briefing.

The briefing shipped as a pipeline, an email and a reader page; chat was never
the fourth. Asked "what topics are on my daily brief?", the assistant correctly
said it had no way to know.

These tools are per-user, and unlike the REIT tools that is the security
boundary rather than a uniform signature: the service-role key bypasses RLS, so
the user_id filter IS the isolation.
"""

from __future__ import annotations

import json

import httpx
import pytest

from tools import daily_briefing as db

UID = "712d8946-ced7-4c85-b968-6cdd1e3887b8"
OTHER = "00000000-0000-0000-0000-000000000001"


@pytest.fixture(autouse=True)
def _creds(monkeypatch):
    monkeypatch.setenv(db.SUPABASE_URL_ENV, "https://example.supabase.co")
    monkeypatch.setenv(db.SERVICE_ROLE_ENV, "test-key")
    monkeypatch.delenv(db.SERVICE_ROLE_FILE_ENV, raising=False)
    yield
    db._transport = None


def _mock(handler):
    db._transport = httpx.MockTransport(handler)


def test_preferences_render_topics(monkeypatch):
    def handler(request):
        assert request.url.params["user_id"] == f"eq.{UID}"
        return httpx.Response(200, json=[{
            "topics": ["Financial Markets", "UGA football"], "deliver_at": "06:30:00",
            "timezone": "America/Chicago", "deliver_email": True,
            "email_to": "a@b.example", "enabled": True}])
    _mock(handler)
    out = db.run_briefing_tool("get_briefing_preferences", UID)
    assert "Financial Markets" in out and "UGA football" in out
    assert "06:30" in out and "America/Chicago" in out


def test_every_query_is_scoped_to_the_caller():
    """The filter is the isolation. It must be present on every request."""
    seen = []

    def handler(request):
        seen.append(dict(request.url.params))
        return httpx.Response(200, json=[])

    _mock(handler)
    for name in ("get_briefing_preferences", "list_briefing_sources", "get_latest_briefing"):
        db.run_briefing_tool(name, UID)
    assert seen, "no requests made"
    for params in seen:
        assert params.get("user_id") == f"eq.{UID}"


def test_a_blank_or_malformed_user_id_fails_closed():
    """A missing id must not become an unfiltered query."""
    calls = []
    _mock(lambda r: calls.append(r) or httpx.Response(200, json=[]))
    for bad in ("", "   ", "not-a-uuid", "*", "eq.anything"):
        out = db.run_briefing_tool("get_briefing_preferences", bad)
        assert out.startswith("error:"), f"{bad!r} was not rejected"
    assert not calls, "a request was issued despite an invalid user id"


def test_search_is_scoped_through_the_users_own_briefings():
    """PostgREST cannot filter a child on a parent column; resolving parents
    first is what keeps this from scanning every user's sections."""
    seen = []

    def handler(request):
        seen.append(request.url)
        if request.url.path.endswith("/briefings"):
            return httpx.Response(200, json=[{"id": "b1", "briefing_date": "2026-08-09"}])
        return httpx.Response(200, json=[{"briefing_id": "b1", "kind": "top", "rank": 1,
                                          "headline": "Fed holds rates", "source_name": "FT",
                                          "url": "https://x.example/1"}])
    _mock(handler)
    out = db.run_briefing_tool("search_briefings", UID, {"query": "Fed"})
    assert "Fed holds rates" in out
    assert seen[0].params["user_id"] == f"eq.{UID}"          # parents scoped
    assert seen[1].params["briefing_id"] == "in.(b1)"          # children limited to them


def test_missing_credentials_degrade_to_an_error_string(monkeypatch):
    monkeypatch.delenv(db.SERVICE_ROLE_ENV, raising=False)
    out = db.run_briefing_tool("get_briefing_preferences", UID)
    assert out.startswith("error:")


def test_http_failure_never_raises():
    _mock(lambda r: httpx.Response(500, json={}))
    out = db.run_briefing_tool("get_latest_briefing", UID)
    assert out.startswith("error:")


def test_unknown_tool_is_reported_not_raised():
    assert db.run_briefing_tool("drop_everything", UID).startswith("error:")


def test_no_briefing_yet_is_a_sentence_not_an_error():
    _mock(lambda r: httpx.Response(200, json=[]))
    assert "no briefings" in db.run_briefing_tool("get_latest_briefing", UID).lower()


def test_bad_date_is_rejected():
    _mock(lambda r: httpx.Response(200, json=[]))
    out = db.run_briefing_tool("get_latest_briefing", UID, {"briefing_date": "yesterday"})
    assert out.startswith("error:")


def test_service_role_file_is_preferred_over_the_blanked_variable(tmp_path, monkeypatch):
    """Compose blanks the plain variable and mounts the value as a file; reading
    the variable first finds an empty string and looks like a missing credential."""
    f = tmp_path / "key"
    f.write_text("file-key\n")
    monkeypatch.setenv(db.SERVICE_ROLE_ENV, "")
    monkeypatch.setenv(db.SERVICE_ROLE_FILE_ENV, str(f))
    assert db._key() == "file-key"


def test_readers_are_free_and_writers_are_gated():
    """Membership in WRITE_TOOLS is what makes "yes" work. A writer missing from
    it runs without ever asking; a reader wrongly in it demands approval to
    answer a question."""
    from tools.catalog import WRITE_TOOLS

    readers = db.BRIEFING_TOOL_REGISTRY - db.BRIEFING_WRITE_TOOLS
    assert not (readers & WRITE_TOOLS), "a read-only briefing tool is gated"
    assert db.BRIEFING_WRITE_TOOLS <= WRITE_TOOLS, "a briefing writer is not gated"
    assert db.BRIEFING_WRITE_TOOLS == {"add_briefing_topic", "remove_briefing_topic"}


def test_add_topic_is_idempotent_and_case_insensitive():
    calls = []

    def handler(request):
        calls.append(request.method)
        if request.method == "GET":
            return httpx.Response(200, json=[{"topics": ["Baseball", "AI"]}])
        return httpx.Response(204)

    _mock(handler)
    out = db.run_briefing_tool("add_briefing_topic", UID, {"topic": "baseball"})
    assert "already" in out.lower()
    assert "PATCH" not in calls, "a duplicate topic still issued a write"


def test_add_topic_appends_and_preserves_order():
    seen = {}

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json=[{"topics": ["AI", "Memphis"]}])
        seen["body"] = json.loads(request.content)
        return httpx.Response(204)

    _mock(handler)
    out = db.run_briefing_tool("add_briefing_topic", UID, {"topic": "Baseball cards"})
    assert seen["body"]["topics"] == ["AI", "Memphis", "Baseball cards"]
    assert "Baseball cards" in out


def test_remove_topic_matches_case_insensitively():
    seen = {}

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json=[{"topics": ["AI", "Baseball cards"]}])
        seen["body"] = json.loads(request.content)
        return httpx.Response(204)

    _mock(handler)
    db.run_briefing_tool("remove_briefing_topic", UID, {"topic": "BASEBALL CARDS"})
    assert seen["body"]["topics"] == ["AI"]


def test_removing_something_absent_changes_nothing():
    calls = []

    def handler(request):
        calls.append(request.method)
        return httpx.Response(200, json=[{"topics": ["AI"]}])

    _mock(handler)
    out = db.run_briefing_tool("remove_briefing_topic", UID, {"topic": "Cricket"})
    assert "not on your briefing" in out
    assert "PATCH" not in calls


def test_writers_reject_a_bad_user_id_before_writing():
    calls = []
    _mock(lambda r: calls.append(r.method) or httpx.Response(200, json=[]))
    for name in ("add_briefing_topic", "remove_briefing_topic"):
        assert db.run_briefing_tool(name, "", {"topic": "X"}).startswith("error:")
    assert not calls


def test_topic_ceiling_is_enforced():
    _mock(lambda r: httpx.Response(200, json=[{"topics": [f"t{i}" for i in range(db.MAX_TOPICS)]}]))
    out = db.run_briefing_tool("add_briefing_topic", UID, {"topic": "one more"})
    assert "maximum" in out.lower()


def test_catalog_and_dispatch_agree():
    import orchestrator
    from tools.catalog import CATALOG_TOOL_NAMES
    assert db.BRIEFING_TOOL_REGISTRY <= CATALOG_TOOL_NAMES
    assert db.BRIEFING_TOOL_REGISTRY <= orchestrator.DISPATCHABLE_TOOLS
