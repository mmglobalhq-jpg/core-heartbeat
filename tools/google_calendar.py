"""Google Calendar tools for the orchestrator (view / create / update / delete).

Per-user: the caller's Google OAuth tokens live in the ``google_credentials`` table
(written by core-chat's connect flow). This module loads them with the Supabase
service-role key, refreshes the access token when expired (persisting the new one),
and calls the Google Calendar API v3 as that user. ``user_id`` is threaded from graph
state (like the KB/vault tools), never a model-supplied arg, so a request can't be
redirected to another user's calendar.

Mirrors tools/graphrag.py: an injectable ``_transport`` test seam, a name->callable
dispatch, and a ``run_calendar_tool`` entrypoint that degrades to a friendly string
(never raises) so a bad tool call never crashes the graph.
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx
from services.secrets import secret as read_secret

logger = logging.getLogger(__name__)

SUPABASE_URL_ENV = "SUPABASE_URL"
SERVICE_ROLE_ENV = "SUPABASE_SERVICE_ROLE_KEY"
GOOGLE_CLIENT_ID_ENV = "GOOGLE_CLIENT_ID"
GOOGLE_CLIENT_SECRET_ENV = "GOOGLE_CLIENT_SECRET"

GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
CAL_BASE = "https://www.googleapis.com/calendar/v3"
REQUEST_TIMEOUT_S = 30.0
EXPIRY_BUFFER_S = 60          # refresh a little before actual expiry
MAX_RESULTS_CAP = 50
DEFAULT_WINDOW_DAYS = 7      # "what's coming up" (no search term)
SEARCH_WINDOW_DAYS = 180     # widen when searching by name/keyword

# Test seam: unit tests set this to an httpx.MockTransport. None -> real network.
_transport: httpx.BaseTransport | None = None


class CalendarError(Exception):
    """KB/config/service error surfaced to the model as ``error: ...``."""


class NotConnected(Exception):
    """The user has not connected Google Calendar (or the refresh token expired)."""


def _http() -> httpx.Client:
    return httpx.Client(timeout=REQUEST_TIMEOUT_S, transport=_transport)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rfc3339(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


# --- token storage (Supabase, service role) ---------------------------------

def _sb_url() -> str:
    url = os.environ.get(SUPABASE_URL_ENV)
    if not url:
        raise CalendarError(f"{SUPABASE_URL_ENV} is not set")
    return url.rstrip("/")


def _sb_headers() -> dict[str, str]:
    key = read_secret(SERVICE_ROLE_ENV)
    if not key:
        raise CalendarError(f"{SERVICE_ROLE_ENV} is not set")
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _load_credentials(user_id: str) -> dict | None:
    with _http() as c:
        r = c.get(
            f"{_sb_url()}/rest/v1/google_credentials",
            params={"user_id": f"eq.{user_id}", "select": "*"},
            headers=_sb_headers(),
        )
    r.raise_for_status()
    rows = r.json()
    return rows[0] if rows else None


def _save_access_token(user_id: str, access_token: str, expiry: datetime) -> None:
    """Persist a freshly refreshed access token.

    Deliberately does NOT raise: the refresh already succeeded, so failing here
    would break a call that was about to work. But it must not be silent either —
    an unpersisted token means every subsequent request refreshes again, burning
    Google's refresh quota until it starts failing, with nothing in the logs to say
    why. The response was previously not checked at all.
    """
    with _http() as c:
        r = c.patch(
            f"{_sb_url()}/rest/v1/google_credentials",
            params={"user_id": f"eq.{user_id}"},
            headers={**_sb_headers(), "Prefer": "return=minimal"},
            json={"access_token": access_token, "expiry": _rfc3339(expiry),
                  "updated_at": _rfc3339(_now())},
        )
    if r.status_code >= 400:
        logger.warning(
            "google access token NOT persisted for user %s: HTTP %d %s — every "
            "later request will refresh again",
            user_id[:8], r.status_code, (r.text or "")[:200],
        )


def _refresh(user_id: str, creds: dict) -> str:
    cid = os.environ.get(GOOGLE_CLIENT_ID_ENV)
    secret = os.environ.get(GOOGLE_CLIENT_SECRET_ENV)
    if not cid or not secret:
        raise CalendarError("Google OAuth is not configured (missing client id/secret)")
    with _http() as c:
        r = c.post(GOOGLE_TOKEN_URL, data={
            "client_id": cid, "client_secret": secret,
            "refresh_token": creds["refresh_token"], "grant_type": "refresh_token",
        })
    # invalid_grant -> refresh token revoked/expired (e.g. the 7-day test-mode limit)
    if r.status_code in (400, 401):
        raise NotConnected()
    r.raise_for_status()
    body = r.json()
    token = body["access_token"]
    expiry = _now() + timedelta(seconds=int(body.get("expires_in", 3600)))
    _save_access_token(user_id, token, expiry)
    return token


def _access_token(user_id: str) -> str:
    creds = _load_credentials(user_id)
    if not creds:
        raise NotConnected()
    try:
        expiry = datetime.fromisoformat(creds["expiry"])
    except (KeyError, ValueError):
        expiry = _now()  # unknown -> force refresh
    if expiry <= _now() + timedelta(seconds=EXPIRY_BUFFER_S):
        return _refresh(user_id, creds)
    return creds["access_token"]


# --- Calendar API v3 --------------------------------------------------------

def _google_error(resp: httpx.Response) -> tuple[str, str]:
    """Pull ``(reason, message)`` out of a Google API error body.

    Google explains itself in the response body — a machine-readable ``reason`` and
    a human ``message``. This client used to discard both and report only
    "Google Calendar returned 403", which is the same text whether the connection
    lacks calendar scope, the user isn't the event's organiser, or the API is
    rate-limiting. Those need three different actions from the user.
    """
    try:
        err = (resp.json() or {}).get("error") or {}
    except Exception:
        return "", ""
    if isinstance(err, str):          # OAuth-style {"error": "invalid_grant"}
        return err, ""
    errors = err.get("errors") or []
    reason = (errors[0].get("reason") if errors and isinstance(errors[0], dict) else "") or ""
    return reason, (err.get("message") or "").strip()


# Actionable text per 403 reason. The point is to say what the USER can do; a bare
# status code leaves them with nothing but "try again", which for a scope problem
# never works.
_FORBIDDEN_GUIDANCE = {
    "insufficientPermissions": (
        "Google Calendar refused the request because the connection doesn't grant "
        "calendar access. Open Settings → Integrations, disconnect Google Calendar, "
        "then connect it again and accept the calendar permission."
    ),
    "forbiddenForNonOrganizer": (
        "Google Calendar won't allow that change because you aren't the event's "
        "organiser. Ask the organiser to change it, or add your own copy instead."
    ),
    "requiredAccessLevel": (
        "That calendar is shared with you read-only, so it can't be changed here."
    ),
    "rateLimitExceeded": "Google Calendar is rate-limiting requests. Try again shortly.",
    "userRateLimitExceeded": "Google Calendar is rate-limiting requests. Try again shortly.",
    "dailyLimitExceeded": (
        "The Google Calendar daily quota for this app is used up. It resets at "
        "midnight Pacific time."
    ),
    "quotaExceeded": (
        "The Google Calendar quota for this app is used up. Try again later."
    ),
}


def _explain_status(resp: httpx.Response) -> str:
    """A message worth showing the user for a failed Calendar call."""
    reason, message = _google_error(resp)
    code = resp.status_code
    if code == 403:
        guidance = _FORBIDDEN_GUIDANCE.get(reason)
        if guidance:
            return guidance
        detail = f" ({message})" if message else ""
        return f"Google Calendar denied the request{detail}."
    if code in (404, 410):
        return ("That calendar event no longer exists — it may already have been "
                "deleted or changed. Try listing your events again to get the "
                "current ones.")
    if code == 409:
        return "That calendar event conflicts with an existing one."
    if code >= 500:
        return "Google Calendar is having trouble right now. Try again shortly."
    detail = f": {message}" if message else ""
    return f"Google Calendar returned {code}{detail}"


def _cal(method: str, path: str, token: str, *, params=None, json_body=None) -> httpx.Response:
    with _http() as c:
        r = c.request(method, f"{CAL_BASE}{path}",
                      params=params, json=json_body,
                      headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
    if r.status_code == 401:
        raise NotConnected()  # token rejected -> prompt reconnect
    if r.status_code >= 400:
        # Log the machine-readable reason for triage, return the human one to the
        # user. Previously the body was dropped entirely, so a scope problem and a
        # rate limit were indistinguishable in both places.
        reason, message = _google_error(r)
        logger.warning(
            "calendar %s %s -> %d reason=%s message=%s",
            method, path, r.status_code, reason or "-", message or "-",
        )
        raise CalendarError(_explain_status(r))
    return r


def _calendar_tz(token: str) -> str:
    try:
        r = _cal("GET", "/calendars/primary", token)
        return r.json().get("timeZone") or "UTC"
    except Exception:
        return "UTC"


def _time_field(value: str, tz: str) -> dict:
    """Build a Calendar start/end object. A bare YYYY-MM-DD is an all-day date;
    anything else is treated as a dateTime interpreted in the calendar's timezone."""
    v = (value or "").strip()
    if len(v) == 10 and v[4] == "-" and v[7] == "-":
        return {"date": v}
    return {"dateTime": v, "timeZone": tz}


def _fmt_event(e: dict) -> str:
    summary = e.get("summary") or "(no title)"
    start = (e.get("start") or {}).get("dateTime") or (e.get("start") or {}).get("date") or "?"
    end = (e.get("end") or {}).get("dateTime") or (e.get("end") or {}).get("date") or "?"
    loc = e.get("location")
    tail = f" @ {loc}" if loc else ""
    return f"• {summary} — {start} to {end}{tail}  [id: {e.get('id')}]"


_HAS_TZ = re.compile(r"(Z|[+-]\d{2}:?\d{2})$")


def _to_rfc3339(value: str | None, default: str, tz_getter) -> str:
    """Google's list timeMin/timeMax require an offset. The model emits NAIVE local
    datetimes (consistent with create/update), so attach the calendar's timezone to
    any naive value; pass through anything already offset-aware or the default."""
    if not value:
        return default
    v = value.strip()
    if _HAS_TZ.search(v):
        return v
    try:
        return datetime.fromisoformat(v).replace(tzinfo=ZoneInfo(tz_getter())).isoformat()
    except Exception:
        return v  # malformed -> let Google reject it rather than guess


def list_events(user_id: str, args: dict) -> str:
    token = _access_token(user_id)
    _tz_cache: list[str] = []
    def _tz() -> str:
        if not _tz_cache:
            _tz_cache.append(_calendar_tz(token))
        return _tz_cache[0]
    # A name search ("when is my dentist appointment?") shouldn't be capped to the
    # next week — widen the default window when a query is given so it finds events
    # further out. An explicit time_min/time_max still wins.
    window_days = SEARCH_WINDOW_DAYS if args.get("query") else DEFAULT_WINDOW_DAYS
    time_min = _to_rfc3339(args.get("time_min"), _rfc3339(_now()), _tz)
    time_max = _to_rfc3339(args.get("time_max"), _rfc3339(_now() + timedelta(days=window_days)), _tz)
    # A season of games is 10-15 events, so a default of 10 silently truncated the
    # answer: asking "what football games are on my calendar?" returned a partial
    # list that read as complete. The window is already bounded above, so the cost
    # of a larger default is a slightly longer prompt, not a runaway result set.
    try:
        max_results = int(args.get("max_results") or 25)
    except (TypeError, ValueError):
        max_results = 25
    params = {
        "timeMin": time_min, "timeMax": time_max,
        "singleEvents": "true", "orderBy": "startTime",
        "maxResults": str(max(1, min(max_results, MAX_RESULTS_CAP))),
    }
    if args.get("query"):
        params["q"] = args["query"]
    r = _cal("GET", "/calendars/primary/events", token, params=params)
    items = r.json().get("items", [])
    if not items:
        return "No events found in that window."
    return "\n".join(_fmt_event(e) for e in items)


def create_event(user_id: str, args: dict) -> str:
    token = _access_token(user_id)
    summary, start, end = args.get("summary"), args.get("start"), args.get("end")
    if not summary or not start or not end:
        return "error: create_calendar_event needs summary, start, and end (ISO 8601)."
    tz = _calendar_tz(token)
    body: dict = {"summary": summary, "start": _time_field(start, tz), "end": _time_field(end, tz)}
    if args.get("description"):
        body["description"] = args["description"]
    if args.get("location"):
        body["location"] = args["location"]
    e = _cal("POST", "/calendars/primary/events", token, json_body=body).json()
    return f"Created event: {_fmt_event(e)}"


def update_event(user_id: str, args: dict) -> str:
    token = _access_token(user_id)
    event_id = args.get("event_id")
    if not event_id:
        return "error: update_calendar_event needs event_id (find it with list_calendar_events first)."
    body: dict = {}
    if args.get("summary"):
        body["summary"] = args["summary"]
    if args.get("description"):
        body["description"] = args["description"]
    if args.get("location"):
        body["location"] = args["location"]
    if args.get("start") or args.get("end"):
        tz = _calendar_tz(token)
        if args.get("start"):
            body["start"] = _time_field(args["start"], tz)
        if args.get("end"):
            body["end"] = _time_field(args["end"], tz)
    if not body:
        return "error: nothing to update — provide a field to change (summary/start/end/location/description)."
    e = _cal("PATCH", f"/calendars/primary/events/{event_id}", token, json_body=body).json()
    return f"Updated event: {_fmt_event(e)}"


def delete_event(user_id: str, args: dict) -> str:
    token = _access_token(user_id)
    event_id = args.get("event_id")
    if not event_id:
        return "error: delete_calendar_event needs event_id (find it with list_calendar_events first)."
    _cal("DELETE", f"/calendars/primary/events/{event_id}", token)
    return f"Deleted event [id: {event_id}]."


# --- dispatch (name -> callable(user_id, args) -> str) ----------------------

_DISPATCH = {
    "list_calendar_events": list_events,
    "create_calendar_event": create_event,
    "update_calendar_event": update_event,
    "delete_calendar_event": delete_event,
}

CALENDAR_TOOL_REGISTRY = frozenset(_DISPATCH)


def run_calendar_tool(name: str, user_id: str, args: dict | None = None) -> str:
    """Execute a registered calendar tool for a user; never raises.

    A missing connection yields a friendly "connect it" message; any other failure
    yields ``error: ...`` so the graph keeps running.
    """
    fn = _DISPATCH.get(name)
    if fn is None:
        return f"error: unknown tool {name!r}"
    try:
        return fn(user_id, args or {})
    except NotConnected:
        return ("Google Calendar isn't connected (or access expired). Open "
                "Settings → Integrations and connect it, then try again.")
    except CalendarError as exc:
        return f"error: {exc}"
    except httpx.HTTPStatusError as exc:
        return f"error: Google Calendar returned {exc.response.status_code}"
    except Exception as exc:  # never crash the graph on a bad tool call
        return f"error: {type(exc).__name__}: {exc}"
