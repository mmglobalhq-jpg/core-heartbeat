"""Unit tests for the Google Calendar tool (tools/google_calendar.py) — no network.

A single httpx.MockTransport routes the three hosts the tool talks to: Supabase
(token table), Google's token endpoint (refresh), and the Calendar API.
"""
import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

import tools.google_calendar as gc

UID = "u-123"


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _creds(expiry: datetime) -> dict:
    return {
        "user_id": UID, "email": "u@gmail.com",
        "access_token": "at-current", "refresh_token": "rt-1",
        "scope": "calendar", "expiry": _iso(expiry),
    }


def _handler(creds_row=None, captured=None):
    def h(request: httpx.Request) -> httpx.Response:
        url, method = str(request.url), request.method
        if captured is not None:
            captured.append((method, url.split("?")[0]))
        base = url.split("?")[0]
        if "/rest/v1/google_credentials" in url:
            if method == "GET":
                return httpx.Response(200, json=[creds_row] if creds_row else [])
            return httpx.Response(204)  # PATCH
        if base == gc.GOOGLE_TOKEN_URL:
            return httpx.Response(200, json={"access_token": "at-refreshed", "expires_in": 3600})
        if base.endswith("/calendars/primary"):
            return httpx.Response(200, json={"timeZone": "America/Chicago"})
        if base.endswith("/calendars/primary/events"):
            if method == "GET":
                return httpx.Response(200, json={"items": [
                    {"id": "ev1", "summary": "Dentist",
                     "start": {"dateTime": "2026-07-14T15:00:00-05:00"},
                     "end": {"dateTime": "2026-07-14T16:00:00-05:00"},
                     "location": "Clinic"},
                ]})
            body = json.loads(request.content)  # POST create
            return httpx.Response(200, json={"id": "new1", "summary": body.get("summary"),
                                             "start": body.get("start"), "end": body.get("end")})
        if "/calendars/primary/events/" in base:
            if method == "PATCH":
                body = json.loads(request.content)
                return httpx.Response(200, json={"id": base.rsplit("/", 1)[-1],
                                                 "summary": body.get("summary", "(no title)")})
            return httpx.Response(204)  # DELETE
        return httpx.Response(404, json={"error": f"unhandled {method} {base}"})
    return h


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "http://sb")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "svc")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "cid")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "secret")


def _install(handler):
    gc._transport = httpx.MockTransport(handler)


def _reset():
    gc._transport = None


def test_not_connected_prompts_to_connect(env):
    _install(_handler(creds_row=None))
    try:
        out = gc.run_calendar_tool("list_calendar_events", UID, {})
    finally:
        _reset()
    assert "isn't connected" in out.lower() or "connect it" in out.lower()


def test_list_events_formats_with_ids(env):
    _install(_handler(_creds(datetime.now(timezone.utc) + timedelta(hours=1))))
    try:
        out = gc.run_calendar_tool("list_calendar_events", UID, {})
    finally:
        _reset()
    assert "Dentist" in out
    assert "[id: ev1]" in out
    assert "@ Clinic" in out


def test_create_event(env):
    _install(_handler(_creds(datetime.now(timezone.utc) + timedelta(hours=1))))
    try:
        out = gc.run_calendar_tool("create_calendar_event", UID, {
            "summary": "Lunch", "start": "2026-07-14T12:00:00", "end": "2026-07-14T13:00:00"})
    finally:
        _reset()
    assert "Created event" in out and "Lunch" in out


def test_create_event_missing_fields(env):
    _install(_handler(_creds(datetime.now(timezone.utc) + timedelta(hours=1))))
    try:
        out = gc.run_calendar_tool("create_calendar_event", UID, {"summary": "Lunch"})
    finally:
        _reset()
    assert out.startswith("error:") and "start" in out


def test_delete_event(env):
    _install(_handler(_creds(datetime.now(timezone.utc) + timedelta(hours=1))))
    try:
        out = gc.run_calendar_tool("delete_calendar_event", UID, {"event_id": "ev1"})
    finally:
        _reset()
    assert out == "Deleted event [id: ev1]."


def test_delete_needs_event_id(env):
    _install(_handler(_creds(datetime.now(timezone.utc) + timedelta(hours=1))))
    try:
        out = gc.run_calendar_tool("delete_calendar_event", UID, {})
    finally:
        _reset()
    assert out.startswith("error:") and "event_id" in out


def test_expired_token_is_refreshed(env):
    captured: list = []
    _install(_handler(_creds(datetime.now(timezone.utc) - timedelta(minutes=5)), captured))
    try:
        out = gc.run_calendar_tool("list_calendar_events", UID, {})
    finally:
        _reset()
    assert "Dentist" in out
    # the token endpoint was hit (refresh) and the new token persisted (PATCH)
    assert ("POST", gc.GOOGLE_TOKEN_URL) in captured
    assert any(m == "PATCH" and p.endswith("/google_credentials") for m, p in captured)


def test_unknown_tool(env):
    out = gc.run_calendar_tool("nope", UID, {})
    assert "unknown tool" in out


def test_to_rfc3339_normalizes_naive_and_passes_through():
    from tools.google_calendar import _to_rfc3339
    out = _to_rfc3339("2026-07-20T00:00:00", "d", lambda: "America/Chicago")
    assert out.startswith("2026-07-20T00:00:00-0")   # got an offset (CDT -05 / CST -06)
    assert _to_rfc3339("2026-07-20T00:00:00Z", "d", lambda: "UTC") == "2026-07-20T00:00:00Z"
    assert _to_rfc3339("2026-07-20T00:00:00-05:00", "d", lambda: "UTC") == "2026-07-20T00:00:00-05:00"
    assert _to_rfc3339("", "DEFAULT", lambda: "UTC") == "DEFAULT"


# --- error explanation ------------------------------------------------------
#
# A 403 previously became "error: Google Calendar returned 403" — the same text
# whether the connection lacks calendar scope, the user isn't the event's
# organiser, or the API is throttling. Those need three different actions, and the
# user hit exactly this: a 403 they could only resolve by guessing (disconnect and
# reconnect, which happened to be right).

import httpx as _httpx

from tools.google_calendar import _explain_status, _google_error


def _resp(status, body):
    return _httpx.Response(status, json=body,
                           request=_httpx.Request("GET", "https://example.test"))


def _google_403(reason, message="Insufficient Permission"):
    return _resp(403, {"error": {"code": 403, "message": message,
                                 "errors": [{"reason": reason, "message": message}]}})


def test_missing_scope_tells_the_user_to_reconnect():
    """The case actually hit in production. "Try again" never fixes it."""
    text = _explain_status(_google_403("insufficientPermissions"))
    assert "Settings" in text and "connect it again" in text
    assert "403" not in text


def test_non_organiser_is_distinguished_from_a_scope_problem():
    text = _explain_status(_google_403("forbiddenForNonOrganizer"))
    assert "organiser" in text
    assert "Settings" not in text, "reconnecting would not help here"


def test_rate_limiting_says_to_wait_rather_than_reconnect():
    for reason in ("rateLimitExceeded", "userRateLimitExceeded"):
        text = _explain_status(_google_403(reason))
        assert "rate-limiting" in text and "shortly" in text
        assert "Settings" not in text


def test_quota_exhaustion_is_its_own_message():
    assert "midnight Pacific" in _explain_status(_google_403("dailyLimitExceeded"))


def test_an_unknown_403_reason_still_surfaces_googles_message():
    """Better to relay Google's own words than to drop them."""
    text = _explain_status(_google_403("somethingNew", "Calendar usage limits exceeded."))
    assert "Calendar usage limits exceeded." in text


def test_a_missing_event_explains_itself():
    """update/delete need an id from a listing, which goes stale."""
    for code in (404, 410):
        text = _explain_status(_resp(code, {"error": {"code": code, "message": "Not Found"}}))
        assert "no longer exists" in text and "listing your events again" in text


def test_server_errors_are_not_blamed_on_the_user():
    assert "Google Calendar is having trouble" in _explain_status(
        _resp(503, {"error": {"code": 503, "message": "Backend Error"}})
    )


def test_an_unparsable_body_does_not_crash_the_explanation():
    r = _httpx.Response(403, content=b"<html>nope</html>",
                        request=_httpx.Request("GET", "https://example.test"))
    assert _google_error(r) == ("", "")
    assert "denied the request" in _explain_status(r)


def test_oauth_style_string_error_is_handled():
    r = _resp(400, {"error": "invalid_grant"})
    assert _google_error(r) == ("invalid_grant", "")
