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

from pathlib import Path

from tools import daily_briefing as db

UID = "712d8946-ced7-4c85-b968-6cdd1e3887b8"
OTHER = "00000000-0000-0000-0000-000000000001"


@pytest.fixture(autouse=True)
def _creds(monkeypatch):
    monkeypatch.setenv(db.SUPABASE_URL_ENV, "https://example.supabase.co")
    monkeypatch.setenv(db.SERVICE_ROLE_ENV, "test-key")
    monkeypatch.delenv(db.SERVICE_ROLE_FILE_ENV, raising=False)
    # The repointed reader (2026-08-25) talks to briefing-agent over HTTP and FAILS CLOSED
    # without both of these. Set here so the tests below exercise the behaviour under test
    # rather than the unconfigured path — two of them were passing on "error:" either way.
    monkeypatch.setenv(db.BRIEF_API_BASE_ENV, "http://brief-web.test:3100")
    monkeypatch.setenv(db.BRIEF_API_TOKEN_ENV, "t" * 48)
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


def test_every_postgrest_query_is_scoped_to_the_caller():
    """The filter is the isolation. It must be present on every request.

    Applies to the tools still reading ``briefing_*`` directly: the service-role key
    bypasses RLS, so ``user_id=eq.<uid>`` is the ONLY thing standing between two users.
    """
    seen = []

    def handler(request):
        seen.append(dict(request.url.params))
        return httpx.Response(200, json=[])

    _mock(handler)
    for name in ("get_briefing_preferences", "list_briefing_sources"):
        db.run_briefing_tool(name, UID)
    assert seen, "no requests made"
    for params in seen:
        assert params.get("user_id") == f"eq.{UID}"


def test_the_repointed_reader_sends_a_token_and_NO_user_id():
    """The new surface isolates differently, and this pins the difference.

    ``get_latest_briefing`` was repointed at briefing-agent on 2026-08-25. That service is
    single-owner: it resolves the reader from its own ``BRIEF_USER_ID`` and never accepts one
    from a caller. Passing a user id would be strictly WORSE — it would turn a service that
    can only answer for its owner into one that answers for whoever is named.

    So isolation here is (a) the token, and (b) the service having exactly one owner. The
    caller's id is still validated first, so a blank session cannot reach the network at all.
    """
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={})

    _mock(handler)
    db.run_briefing_tool("get_latest_briefing", UID)
    assert seen, "no request made"
    request = seen[0]
    assert request.headers.get("authorization", "").startswith("Bearer ")
    assert "user_id" not in dict(request.url.params)
    assert UID not in str(request.url)


def test_a_blank_or_malformed_user_id_fails_closed():
    """A missing id must not become an unfiltered query."""
    calls = []
    _mock(lambda r: calls.append(r) or httpx.Response(200, json=[]))
    for bad in ("", "   ", "not-a-uuid", "*", "eq.anything"):
        out = db.run_briefing_tool("get_briefing_preferences", bad)
        assert out.startswith("error:"), f"{bad!r} was not rejected"
    assert not calls, "a request was issued despite an invalid user id"


def test_search_is_scoped_by_the_SERVICE_not_by_this_caller():
    """The scoping moved, and this pins where it moved to.

    The old reader resolved the user's briefing ids and then bounded the child query to
    them, because PostgREST cannot filter a child on a parent column and the alternative was
    scanning every user's sections. Repointing at briefing-agent (2026-08-25) moved that
    work server-side, where it is asserted by that repo's own tests.

    What must hold HERE is that this tool cannot ask for anyone else's brief: it sends the
    query, a bounded limit and a bearer token, and never a user id. The caller's id is still
    validated first, so a blank session cannot reach the network at all.
    """
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"ok": True, "query": "Fed", "count": 1, "matches": [
            {"briefDate": "2026-08-09", "section": "Business",
             "text": "Fed holds rates", "url": "https://x.example/1"}]})

    _mock(handler)
    out = db.run_briefing_tool("search_briefings", UID, {"query": "Fed"})
    assert "Fed holds rates" in out
    assert "2026-08-09" in out

    request = seen[0]
    assert request.headers.get("authorization", "").startswith("Bearer ")
    assert request.url.params["q"] == "Fed"
    assert "user_id" not in dict(request.url.params)
    assert UID not in str(request.url)


def test_search_limit_is_bounded_before_it_leaves():
    """A caller-supplied limit is clamped here as well as at the surface.

    Belt and braces on purpose: the model picks this number, and an unbounded one would ask
    the brief service for everything and then paste it into a chat reply.
    """
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"ok": True, "count": 0, "matches": []})

    _mock(handler)
    db.run_briefing_tool("search_briefings", UID, {"query": "Fed", "limit": 9999})
    assert int(seen[0].url.params["limit"]) <= 25
    _mock(handler)
    db.run_briefing_tool("search_briefings", UID, {"query": "Fed", "limit": "nonsense"})
    assert int(seen[1].url.params["limit"]) == 10


def test_missing_credentials_degrade_to_an_error_string(monkeypatch):
    monkeypatch.delenv(db.SERVICE_ROLE_ENV, raising=False)
    out = db.run_briefing_tool("get_briefing_preferences", UID)
    assert out.startswith("error:")


def test_http_failure_never_raises():
    # 500 from the brief service, not from PostgREST — same requirement either way.
    _mock(lambda r: httpx.Response(500, json={}))
    out = db.run_briefing_tool("get_latest_briefing", UID)
    assert out.startswith("error:")


def test_unknown_tool_is_reported_not_raised():
    assert db.run_briefing_tool("drop_everything", UID).startswith("error:")


def test_no_briefing_yet_is_a_sentence_not_an_error():
    # The CLAIM is that absence reads as a sentence rather than an error string, which is
    # what the assistant needs in order to say "there isn't one yet" instead of apologising
    # for a failure. The old wording was "you have no briefings yet"; repointing at
    # briefing-agent (2026-08-25) made the noun singular. Asserted on the property, not the
    # phrasing, so a future rewording cannot fail this for the wrong reason.
    _mock(lambda r: httpx.Response(200, json={}))
    out = db.run_briefing_tool("get_latest_briefing", UID)
    assert not out.startswith("error:")
    assert "no brief" in out.lower()


def test_bad_date_is_rejected():
    _mock(lambda r: httpx.Response(200, json={}))
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


def test_the_writers_are_RETIRED_and_the_module_cannot_write():
    """add/remove_briefing_topic are gone by owner decision (2026-08-25).

    They wrote to ``briefing_prefs.topics``, a settings column on the pipeline being sunset.
    They were retired rather than repointed because the new system has nothing to repoint AT:
    an interest there is a ``stated`` event on an append-only log, and spec §2 reserves
    removal to the user alone. Giving chat a write path is a decision about write authority,
    not a port of these two.

    Asserted three ways, because a retired tool that is still reachable is worse than one that
    was never removed: gone from dispatch, gone from the catalog, and the module has no HTTP
    write verb left at all.
    """
    for name in ("add_briefing_topic", "remove_briefing_topic"):
        assert name not in db.BRIEFING_TOOL_REGISTRY
        assert db.run_briefing_tool(name, UID).startswith("error:")

    # Kept as an EMPTY set rather than deleted: the confirmation machinery that consumes it is
    # unchanged and ready for a future write family, and removing the name would make its
    # absence look like a bug at the call site.
    assert db.BRIEFING_WRITE_TOOLS == frozenset()

    source = Path(db.__file__).read_text(encoding="utf-8")
    for verb in (".patch(", ".post(", ".put(", ".delete("):
        assert verb not in source, f"the briefing tools are read-only; found {verb}"


def test_catalog_and_dispatch_agree():
    import orchestrator
    from tools.catalog import CATALOG_TOOL_NAMES
    assert db.BRIEFING_TOOL_REGISTRY <= CATALOG_TOOL_NAMES
    assert db.BRIEFING_TOOL_REGISTRY <= orchestrator.DISPATCHABLE_TOOLS


class TestBriefingTurnsSkipTheForcedKnowledgeBaseRetrieval:
    """When the router calls no tool on a substantive turn, the supervisor forces
    a query_knowledge_base with the raw user text. The briefing lives in its own
    tables, so that retrieval returns whatever is nearest in vector space and the
    composer answers from it.

    Measured on the deployed build: "change my briefing delivery time to 5am"
    produced "I can change your briefing delivery time if it's an event on your
    Google Calendar." The router had correctly declined; the backstop overrode it.
    """

    def test_recognises_a_briefing_turn(self):
        for text in ("what topics are on my daily brief?",
                     "change my briefing delivery time to 5am",
                     "add baseball cards to my daily briefing",
                     "what was in the briefing today"):
            assert db.looks_like_briefing_reference(text), text

    def test_is_narrow_enough_not_to_swallow_ordinary_turns(self):
        for text in ("keep it brief", "brief me on the roadmap",
                     "summarise this in brief", "debrief the team",
                     "what's on my calendar?"):
            assert not db.looks_like_briefing_reference(text), text

    def test_the_supervisor_actually_consults_it(self):
        import orchestrator
        assert orchestrator.looks_like_briefing_reference is db.looks_like_briefing_reference
