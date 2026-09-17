"""Read-only daily-briefing tools for the orchestrator.

WHY THIS EXISTS
The briefing shipped as three things — a pipeline, an email, and the ``/briefing``
reader page — and chat was never the fourth. Asked "what topics are on my daily
brief?", the assistant answered that it had no way to know, which was true: no
tool touched ``briefing_prefs``, ``briefings`` or ``briefing_user_sources``, and
those tables are not in the vault or the knowledge base either. This closes that
gap and nothing else; the pipeline is untouched.

READS ARE FREE; THE WRITERS ARE IN ``WRITE_TOOLS`` — WHICH IS NOT THE SAME AS
"ALWAYS CONFIRMED", AND THE DIFFERENCE WAS MEASURED, NOT ASSUMED
Membership gates a write through ``_needs_confirmation``, which proposes a batch
only at ``WRITE_CONFIRM_THRESHOLD`` (3) or more writes, or for a tool in
``ALWAYS_CONFIRM_TOOLS``. So a SINGLE "add baseball cards" runs immediately and
answers in one turn — verified on the deployed build, where the trace read
``gate.passthrough`` rather than ``gate.PROPOSE``. An earlier version of this
docstring claimed nothing happens until the user approves; that was wrong for the
one-topic case, which is the common one.

That policy is right here rather than merely inherited: a topic add is undone by
``remove_briefing_topic`` in one sentence, so proposing it would add a round trip
to protect against something trivially reversible. Deleting a calendar event is
in ``ALWAYS_CONFIRM_TOOLS`` because it is not. If topic writes ever become
destructive — dropping every topic at once, say — they belong in that set too.

WHAT THE WRITERS FIXED. The first
version of this module was read-only, and when a user asked to add a topic the
assistant offered to do it, no plan was ever proposed because no write tool
existed, and the approval then found nothing to confirm: "I don't have a record
of your previous request." The gate was right to refuse; the missing tool was the
defect. With a writer present the router now calls it on the first turn and the
question never arises.

Still deliberately absent: changing the delivery time, the timezone, the email
address, or triggering a run. Those change WHEN and WHETHER a real person is
emailed, rather than what the briefing is about, and belong in settings where
they are seen rather than in a sentence.

BRIEFINGS ARE PER USER, UNLIKE REIT REPORTS
``tools/reit_research.py`` accepts ``user_id`` for a uniform dispatch signature
and ignores it, because reports are global. Here it is the security boundary:
every query filters on the ``user_id`` taken from LangGraph state, never from a
model argument. The API layer does not rely on RLS — the service-role key
bypasses it — so that filter IS the isolation, the same rule doc 30 §4 records
for the briefing HTTP routes. A missing or malformed ``user_id`` returns an error
rather than an unfiltered query.

Mirrors the other tool modules: an injectable ``_transport`` seam, a
name->callable dispatch, and a ``run_briefing_tool`` entrypoint that degrades to
a concise ``error: ...`` string rather than raising into the graph.
"""

from __future__ import annotations

import datetime as dt
import os
import re

import httpx

# THE SERVICE-ROLE CREDENTIAL IS GONE FROM THIS MODULE (2026-08-25).
#
# Every remaining tool reads briefing-agent over HTTP with a bearer token, so nothing here
# needs a key that bypasses RLS on the Core project any more. That is the point worth keeping:
# retiring the four old-table tools did not just remove code, it removed this module from the
# set of things that hold a credential able to read every user's rows.

_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                   r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")

_transport: httpx.BaseTransport | None = None
"""Test seam. Set to an httpx.MockTransport to run without a network."""


class _BriefingError(RuntimeError):
    """Internal; converted to an ``error: ...`` string at the boundary."""




def _uid(user_id: str) -> str:
    """Validate the caller's id before it reaches a query.

    Not decoration: this value is the entire isolation boundary, so a blank or
    malformed one must fail closed rather than produce a filter that matches
    everything.
    """
    value = (user_id or "").strip()
    if not _UUID.match(value):
        raise _BriefingError("no valid user id in session")
    return value


# --- the NEW brief (briefing-agent), reached over its integration surface ------
#
# READS GO THROUGH HTTP, NOT THE DATABASE, and that is the boundary rather than a
# preference. Two systems reading one schema means every migration over there becomes a
# coordinated deploy over here, and the one that forgets is the one that breaks at 06:00.
# briefing-agent owns brief_*; this asks it questions.
#
# The token is required. brief-web sits on heartbeat-net so cloudflared can serve it, which
# means this call never passes Cloudflare Access — the surface fails closed without it.

BRIEF_API_BASE_ENV = "BRIEF_API_BASE_URL"
BRIEF_API_TOKEN_ENV = "BRIEF_API_TOKEN"


def _brief_api(path: str, params: dict | None = None) -> dict:
    base = (os.environ.get(BRIEF_API_BASE_ENV) or "").strip().rstrip("/")
    token = (os.environ.get(BRIEF_API_TOKEN_ENV) or "").strip()
    if not base or not token:
        raise _BriefingError("the daily brief integration is not configured")
    with httpx.Client(timeout=20.0, transport=_transport) as client:
        response = client.get(
            f"{base}{path}",
            params=params or {},
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        )
    if response.status_code == 404:
        return {}
    if response.status_code >= 400:
        # Never echo the token, and never the body — it may carry the brief itself.
        raise _BriefingError(f"the daily brief service returned HTTP {response.status_code}")
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


# --- tools -------------------------------------------------------------------


# --- readers on the OLD tables: RETIRED 2026-08-25 --------------------------
#
# ``get_briefing_preferences`` and ``list_briefing_sources`` read
# ``briefing_prefs`` and ``briefing_user_sources`` — settings tables belonging to
# the pipeline being sunset (briefing-agent/docs/SUNSET.md). Retired rather than
# repointed because the replacement has nothing equivalent to read: it is
# config-driven, its schedule lives in the worker's environment, and its feed
# list is a static declaration in the repo rather than per-user rows.
#
# WHAT THIS COSTS, STATED PLAINLY. Doc 29 §32 records the founding case for this
# whole tool family: asked "what topics are on my daily brief?", the assistant
# answered that it had no way to know. After this it cannot answer that again.
# The difference is that there is now no per-user answer to give — a config file
# is the truth — rather than an answer it was merely blind to.

def get_latest_briefing(user_id: str, args: dict) -> str:
    """Today's daily brief, or one specific date (YYYY-MM-DD)."""
    _uid(user_id)  # the isolation boundary still has to hold before any call goes out
    wanted = (args or {}).get("briefing_date")
    params: dict = {}
    if wanted:
        try:
            dt.date.fromisoformat(str(wanted))
        except ValueError:
            return "error: briefing_date must be YYYY-MM-DD"
        params["date"] = str(wanted)

    # REPOINTED 2026-08-25 at briefing-agent (ROADMAP Phase 12). The old reader queried
    # briefing_* directly; that schema is being retired and this one is not ours to read.
    payload = _brief_api("/api/brief/today", params)
    if not payload or not payload.get("ok"):
        return ("No brief found for that date." if wanted
                else "There is no brief for today yet.")

    out = [f"Brief for {payload.get('displayDate') or payload.get('date')}", ""]
    if payload.get("summary"):
        out += [str(payload["summary"]), ""]

    # The opening: markets on a weekday, yesterday's scores at the weekend, and a STATED
    # absence when there is neither (spec §2 — silence is a confident false statement).
    for row in payload.get("dataRows") or []:
        change = f"  {row.get('change')}" if row.get("change") else ""
        out.append(f"{row.get('label')}: {row.get('value')}{change}")
    for group in payload.get("scoreGroups") or []:
        out.append(str(group.get("league")))
        out += [f"  {line}" for line in group.get("lines") or []]
    if payload.get("openingAbsence"):
        out.append(str(payload["openingAbsence"]))
    if payload.get("dataRows") or payload.get("scoreGroups") or payload.get("openingAbsence"):
        out.append("")

    for section in payload.get("sections") or []:
        out.append(f"{section.get('title')}:")
        if section.get("emptyReason"):
            out.append(f"  {section['emptyReason']}")
        for bullet in section.get("bullets") or []:
            out.append(f"  - {bullet.get('text')}")
            if bullet.get("url"):
                out.append(f"    {bullet['url']}")
        out.append("")

    notes = payload.get("degradationNotes") or []
    if payload.get("degradationTier") and payload["degradationTier"] != "full":
        out.append(f"[This brief ran {payload['degradationTier']}.]")
        out += [f"  - {n}" for n in notes]
    return "\n".join(out).rstrip()


def search_briefings(user_id: str, args: dict) -> str:
    """Find past brief items whose text matches a query."""
    _uid(user_id)  # the isolation boundary still has to hold before any call goes out
    query = ((args or {}).get("query") or "").strip()
    if not query:
        return "error: query is required"
    limit = (args or {}).get("limit") or 10
    try:
        limit = max(1, min(int(limit), 25))
    except (TypeError, ValueError):
        limit = 10

    # REPOINTED 2026-08-25 at briefing-agent (ROADMAP Phase 12). The old reader resolved the
    # user's briefing ids and then scanned their sections; the new surface does that scoping
    # itself, against its own owner, so nothing here passes a user id.
    payload = _brief_api("/api/brief/search", {"q": query, "limit": str(limit)})
    if not payload or not payload.get("ok"):
        return "The daily brief archive is unavailable."
    matches = payload.get("matches") or []
    if not matches:
        return f"No brief items matched {query!r}."

    out = [f"{len(matches)} brief item(s) matching {query!r}:"]
    for m in matches:
        section = f" [{m.get('section')}]" if m.get("section") else ""
        out.append(f"  {m.get('briefDate')} — {m.get('text')}{section}")
        if m.get("url"):
            out.append(f"      {m['url']}")
    return "\n".join(out)


# --- writers: RETIRED 2026-08-25 -------------------------------------------
#
# ``add_briefing_topic`` and ``remove_briefing_topic`` wrote to
# ``briefing_prefs.topics``, which belongs to the pipeline being sunset
# (briefing-agent/docs/SUNSET.md). Retired by owner decision rather than
# repointed, because the new system has no equivalent to repoint AT:
#
#   * topics there are not a settings column. An interest is a ``stated`` event
#     on an append-only user-model log, and spec §2 reserves removal to the user
#     alone — inference may add, only the user may retire.
#   * the integration surface is read-only BY CONSTRUCTION. Giving chat a write
#     path is a decision about write authority, not a port of these two.
#
# So this is a deliberate capability removal, not an oversight, and the reader
# now changes interests where the model actually lives rather than through a
# column the retiring pipeline reads. ``BRIEFING_WRITE_TOOLS`` stays as an empty
# frozenset: the confirmation machinery that consumes it is unchanged and ready
# for a future write family, and deleting the name would make its absence look
# like a bug at the call site.

_DISPATCH = {
    "get_latest_briefing": get_latest_briefing,
    "search_briefings": search_briefings,
}

BRIEFING_WRITE_TOOLS: frozenset[str] = frozenset()
"""Declared here, next to the implementations, and imported by tools/catalog.py.
A write tool that is not in WRITE_TOOLS runs without confirmation."""

BRIEFING_TOOL_REGISTRY = frozenset(_DISPATCH)


def run_briefing_tool(name: str, user_id: str, args: dict | None = None) -> str:
    """Execute a registered briefing tool; never raises.

    Any failure — missing credentials, a bad user id, an HTTP error — degrades to
    a concise ``error: ...`` string so the orchestration graph keeps running.
    """
    fn = _DISPATCH.get(name)
    if fn is None:
        return f"error: unknown briefing tool {name!r}"
    try:
        return fn(user_id, args or {})
    except _BriefingError as exc:
        return f"error: {exc}"
    except httpx.HTTPError as exc:
        return f"error: could not reach the briefing store ({type(exc).__name__})"
    except Exception as exc:  # noqa: BLE001 — a tool must never crash the graph
        return f"error: {type(exc).__name__}"
