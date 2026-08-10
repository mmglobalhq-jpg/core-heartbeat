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


def _topics(uid: str) -> list[str]:
    rows = _get("/briefing_prefs", {"user_id": f"eq.{uid}", "select": "topics"})
    if not rows:
        raise _BriefingError("you have no daily briefing preferences yet")
    return list(rows[0].get("topics") or [])


def _write_topics(uid: str, topics: list[str]) -> None:
    with _client() as client:
        response = client.patch(
            "/briefing_prefs", params={"user_id": f"eq.{uid}"},
            json={"topics": topics},
            headers={"Content-Type": "application/json", "Prefer": "return=minimal"},
        )
    if response.status_code >= 400:
        raise _BriefingError(f"could not save topics (HTTP {response.status_code})")


def _get(path: str, params: dict) -> list[dict]:
    with _client() as client:
        response = client.get(path, params=params)
    if response.status_code >= 400:
        raise _BriefingError(f"briefing store returned HTTP {response.status_code}")
    payload = response.json()
    return payload if isinstance(payload, list) else []


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
    """The most recent briefing, or one specific date (YYYY-MM-DD)."""
    uid = _uid(user_id)
    params = {
        "user_id": f"eq.{uid}",
        "select": "id,briefing_date,status,briefing_sections(kind,rank,headline,body,url,source_name)",
        "order": "briefing_date.desc",
        "limit": "1",
    }
    wanted = (args or {}).get("briefing_date")
    if wanted:
        try:
            dt.date.fromisoformat(str(wanted))
        except ValueError:
            return "error: briefing_date must be YYYY-MM-DD"
        params["briefing_date"] = f"eq.{wanted}"
    rows = _get("/briefings", params)
    if not rows:
        return ("No briefing found for that date." if wanted
                else "You have no briefings yet.")
    b = rows[0]
    sections = sorted(
        b.get("briefing_sections") or [],
        key=lambda s: (0 if s.get("kind") == "top" else 1, s.get("rank") or 0),
    )
    out = [f"Briefing for {b.get('briefing_date')} (status: {b.get('status')})", ""]
    for s in sections:
        label = f"Top {s.get('rank')}" if s.get("kind") == "top" else "Deep Dive"
        out.append(f"{label}: {s.get('headline')}  [{s.get('source_name')}]")
        if s.get("body"):
            out.append(f"  {s['body']}")
        if s.get("url"):
            out.append(f"  {s['url']}")
        out.append("")
    return "\n".join(out).rstrip()


def search_briefings(user_id: str, args: dict) -> str:
    """Find past briefing items whose headline matches a query."""
    uid = _uid(user_id)
    query = ((args or {}).get("query") or "").strip()
    if not query:
        return "error: query is required"
    limit = (args or {}).get("limit") or 10
    try:
        limit = max(1, min(int(limit), 25))
    except (TypeError, ValueError):
        limit = 10
    # Scoped through the user's own briefings. PostgREST cannot filter a child on
    # a parent column, so the parent ids are resolved first — the alternative is
    # an unscoped scan of every user's sections, which is exactly the isolation
    # this module exists to keep.
    parents = _get("/briefings", {
        "user_id": f"eq.{uid}", "select": "id,briefing_date",
        "order": "briefing_date.desc", "limit": "120",
    })
    if not parents:
        return "You have no briefings yet."
    dates = {p["id"]: p["briefing_date"] for p in parents}
    ids = ",".join(dates)
    escaped = query.replace("*", "").replace(",", " ")
    rows = _get("/briefing_sections", {
        "briefing_id": f"in.({ids})",
        "headline": f"ilike.*{escaped}*",
        "select": "briefing_id,kind,rank,headline,source_name,url",
        "limit": str(limit),
    })
    if not rows:
        return f"No briefing items matched {query!r}."
    out = [f"{len(rows)} briefing item(s) matching {query!r}:"]
    for r in rows:
        out.append(f"  {dates.get(r.get('briefing_id'), '?')} — {r.get('headline')} "
                   f"[{r.get('source_name')}]")
        if r.get("url"):
            out.append(f"      {r['url']}")
    return "\n".join(out)


# --- writers (confirmation-gated) -------------------------------------------

MAX_TOPICS = 40
"""A ceiling, not a preference. Every topic costs candidate selection work and
enlarges the judge prompt; at 11 topics the model already began omitting keys
from its reply. Refusing at a limit is better than degrading silently."""


def add_briefing_topic(user_id: str, args: dict) -> str:
    """Add one topic. Idempotent, case-insensitive, order preserved."""
    uid = _uid(user_id)
    topic = ((args or {}).get("topic") or "").strip()
    if not topic:
        return "error: topic is required"
    if len(topic) > 60:
        return "error: that topic is too long (60 characters max)"
    current = _topics(uid)
    if any(t.strip().lower() == topic.lower() for t in current):
        return f"{topic!r} is already on your briefing. Nothing changed."
    if len(current) >= MAX_TOPICS:
        return (f"You already follow {len(current)} topics, which is the maximum. "
                "Remove one first.")
    _write_topics(uid, current + [topic])
    return (f"Added {topic!r} to your daily briefing. You now follow "
            f"{len(current) + 1} topics; it will be used from the next briefing.")


def remove_briefing_topic(user_id: str, args: dict) -> str:
    """Remove one topic, matched case-insensitively."""
    uid = _uid(user_id)
    topic = ((args or {}).get("topic") or "").strip()
    if not topic:
        return "error: topic is required"
    current = _topics(uid)
    kept = [t for t in current if t.strip().lower() != topic.lower()]
    if len(kept) == len(current):
        return (f"{topic!r} is not on your briefing, so nothing changed. "
                f"You follow: {', '.join(current) or '(none)'}.")
    _write_topics(uid, kept)
    return (f"Removed {topic!r} from your daily briefing. You now follow "
            f"{len(kept)} topics.")


# --- dispatch (name -> callable(user_id, args) -> str) ----------------------

_DISPATCH = {
    "get_briefing_preferences": get_briefing_preferences,
    "list_briefing_sources": list_briefing_sources,
    "get_latest_briefing": get_latest_briefing,
    "search_briefings": search_briefings,
    "add_briefing_topic": add_briefing_topic,
    "remove_briefing_topic": remove_briefing_topic,
}

BRIEFING_WRITE_TOOLS = frozenset({"add_briefing_topic", "remove_briefing_topic"})
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
