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

SUPABASE_URL_ENV = "SUPABASE_URL"
SERVICE_ROLE_ENV = "SUPABASE_SERVICE_ROLE_KEY"
SERVICE_ROLE_FILE_ENV = "SUPABASE_SERVICE_ROLE_KEY_FILE"

_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                   r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")

_transport: httpx.BaseTransport | None = None
"""Test seam. Set to an httpx.MockTransport to run without a network."""


class _BriefingError(RuntimeError):
    """Internal; converted to an ``error: ...`` string at the boundary."""


_BRIEFING_REFERENCE_RE = re.compile(
    r"\b(?:daily\s+brief(?:ing)?s?|my\s+brief(?:ing)?s?|the\s+brief(?:ing)?)\b",
    re.I,
)


def looks_like_briefing_reference(text: str) -> bool:
    """True if the text is clearly about the user's own daily briefing.

    Used by the supervisor to keep briefing turns out of the FORCED generic
    knowledge-base retrieval, exactly as `looks_like_reit_reference` does for
    REIT questions. Deliberately narrow: it wants "my daily brief", not the word
    "brief" in "keep it brief".

    WHY THIS IS NEEDED AND NOT MERELY TIDY. When the router calls no tool on a
    substantive turn, the supervisor forces `query_knowledge_base` with the raw
    user text. The briefing lives in its own tables — the KB does not contain it
    — so that retrieval returns whatever is nearest in vector space and the
    composer then answers from it. Measured on the deployed build: "change my
    briefing delivery time to 5am" produced "I can change your briefing delivery
    time if it's an event on your Google Calendar." The router had correctly
    declined; the backstop overrode the decision and the composer confabulated
    from an unrelated result.

    So a decline can only mean "say we cannot do this" if nothing downstream
    reinterprets it as "go searching".
    """
    return bool(_BRIEFING_REFERENCE_RE.search(text or ""))


def _key() -> str:
    """Service-role key, file-mounted first.

    The platform moved service keys from environment values to mounted files on
    2026-08-06; the plain variable is deliberately blanked in compose, so reading
    it first would find an empty string and look like a missing credential.
    """
    path = os.environ.get(SERVICE_ROLE_FILE_ENV)
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            value = handle.read().strip()
            if value:
                return value
    value = (os.environ.get(SERVICE_ROLE_ENV) or "").strip()
    if not value:
        raise _BriefingError("no Supabase service-role credential configured")
    return value


def _client() -> httpx.Client:
    url = (os.environ.get(SUPABASE_URL_ENV) or "").strip().rstrip("/")
    if not url:
        raise _BriefingError(f"{SUPABASE_URL_ENV} is not set")
    key = _key()
    return httpx.Client(
        base_url=f"{url}/rest/v1",
        headers={"apikey": key, "Authorization": f"Bearer {key}",
                 "Accept": "application/json"},
        timeout=20.0,
        transport=_transport,
    )


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


def _get(path: str, params: dict) -> list[dict]:
    with _client() as client:
        response = client.get(path, params=params)
    if response.status_code >= 400:
        raise _BriefingError(f"briefing store returned HTTP {response.status_code}")
    payload = response.json()
    return payload if isinstance(payload, list) else []



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


def get_briefing_preferences(user_id: str, args: dict) -> str:
    """Topics, delivery time, timezone and email setting."""
    uid = _uid(user_id)
    rows = _get("/briefing_prefs", {
        "user_id": f"eq.{uid}",
        "select": "topics,deliver_at,timezone,deliver_email,email_to,enabled",
    })
    if not rows:
        return "No daily briefing preferences are set up for you yet."
    p = rows[0]
    topics = p.get("topics") or []
    lines = [
        f"Daily briefing: {'enabled' if p.get('enabled') else 'DISABLED'}",
        f"Delivery time: {str(p.get('deliver_at') or '')[:5]} {p.get('timezone') or ''}".strip(),
        ("Email: on to " + p["email_to"]) if p.get("deliver_email") and p.get("email_to")
        else "Email: off",
        "",
        f"Topics ({len(topics)}):" if topics else "Topics: none set",
    ]
    lines += [f"  - {t}" for t in topics]
    return "\n".join(lines)


def list_briefing_sources(user_id: str, args: dict) -> str:
    """The feeds this user added themselves."""
    uid = _uid(user_id)
    rows = _get("/briefing_user_sources", {
        "user_id": f"eq.{uid}",
        "select": "name,kind,url,topic,is_active,last_error",
        "order": "created_at",
    })
    if not rows:
        return ("You have not added any custom feeds. Your briefing uses the "
                "platform's default feeds only.")
    lines = [f"Your custom feeds ({len(rows)}):"]
    for r in rows:
        state = "" if r.get("is_active") else "  [inactive]"
        err = f"  [last error: {r['last_error']}]" if r.get("last_error") else ""
        lines.append(f"  - {r.get('name')} ({r.get('topic')}){state}{err}")
        lines.append(f"      {r.get('url')}")
    lines.append("\nThese are in addition to the platform's default feeds.")
    return "\n".join(lines)


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
    "get_briefing_preferences": get_briefing_preferences,
    "list_briefing_sources": list_briefing_sources,
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
