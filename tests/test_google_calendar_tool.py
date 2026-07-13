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
