"""The single declaration of every tool the Supervisor may call.

Historically a tool existed in four hand-maintained places at once: a registry, a
``Literal`` in ``RoutingDecision``, an enum in ``ROUTING_JSON_SCHEMA``, and a prose
paragraph in the Supervisor's prompt. They drifted — the calendar tools were added
to three of the four and silently omitted from the schema.

Here each tool is declared **once**, as a LangChain ``@tool``. The name comes from
the function, the description from the docstring, and the argument schema from the
type hints — so a provider-shaped tool schema can be generated rather than written
out, and there is nothing to keep in sync.

Two deliberate choices:

**Execution still goes through the existing ``run_*_tool`` dispatchers.** Those are
already tested and already convert every failure into an ``error: ...`` string
rather than raising into the graph. These wrappers delegate to them instead of
reimplementing them, so adopting native tool-calling changed how tools are
*described*, not how they are *run*.

**``user_id`` arrives via ``InjectedState``, never as a model argument.** LangChain
excludes ``InjectedState`` parameters from the schema sent to the model (asserted in
tests/test_tool_catalog.py), so a model cannot name another user's calendar or vault
no matter what it emits. This mirrors the rule the raw dispatchers already enforce.

The docstrings are load-bearing: they are the model's *only* description of each
tool, replacing the prose catalogue that used to live in the Supervisor prompt.
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from tools.daily_briefing import BRIEFING_WRITE_TOOLS, run_briefing_tool
from tools.google_calendar import run_calendar_tool
from tools.graphrag import run_graphrag_tool
from tools.reit_research import run_reit_tool
from tools.user_vault import run_vault_tool
from tools.web_tools import run_web_tool

SANDBOX_FALLBACK = "00000000-0000-0000-0000-000000000000"


def _uid(state: dict) -> str:
    """Resolve the caller from graph state. Never from a tool argument."""
    return (state or {}).get("user_id") or SANDBOX_FALLBACK


# --- personal vault ---------------------------------------------------------


@tool
def read_user_note(filename: str, state: Annotated[dict, InjectedState]) -> str:
    """Read one note from the user's private Markdown vault.

    Give a vault-relative path such as "projects/roadmap.md". Never include a user
    id, an absolute path, or "..".
    """
    return run_vault_tool("read_user_note", _uid(state), {"filename": filename})


@tool
def search_user_vault(query: str, state: Annotated[dict, InjectedState]) -> str:
    """Case-insensitive text/regex search across all notes in the user's vault."""
    return run_vault_tool("search_user_vault", _uid(state), {"query": query})


@tool
def write_user_note(
    filename: str, content: str, state: Annotated[dict, InjectedState]
) -> str:
    """Create or overwrite a note in the user's private Markdown vault.

    Give a vault-relative path such as "notes/meeting.md". Never include a user id,
    an absolute path, or "..".
    """
    return run_vault_tool(
        "write_user_note", _uid(state), {"filename": filename, "content": content}
    )


# --- knowledge base ---------------------------------------------------------


@tool
def query_knowledge_base(query: str, state: Annotated[dict, InjectedState]) -> str:
    """Semantic search over the user's saved documents plus shared/global docs.

    This is the user's curated "core knowledge", which persists across chats — they
    saved it because they consider it important, so consult it on a substantive or
    topical turn even when you could answer from general knowledge. On a follow-up
    ("other options?", "tell me more"), build the query from the follow-up PLUS the
    topic established earlier in the conversation.

    Call this AT MOST ONCE per turn. Once a result has come back — including "no
    relevant information found" — answer from it rather than querying again with
    reworded terms. Skip it entirely for greetings and small talk.

    Do NOT use this for questions about REIT research reports; those have dedicated
    tools.
    """
    return run_graphrag_tool("query_knowledge_base", _uid(state), {"query": query})[0]


# --- google calendar --------------------------------------------------------
#
# Datetimes are NAIVE LOCAL — "2026-08-21T19:00:00", never a Z or a UTC offset. The
# calendar applies the user's own timezone and DST; sending an offset double-applies
# it and lands the event at the wrong hour.


@tool
def list_calendar_events(
    state: Annotated[dict, InjectedState],
    time_min: str | None = None,
    time_max: str | None = None,
    query: str | None = None,
    max_results: int | None = None,
) -> str:
    """View the user's calendar events. Defaults to the next 7 days.

    time_min/time_max are naive local ISO 8601 datetimes with NO timezone offset
    (e.g. "2026-08-21T19:00:00"). Each returned event ends with [id: <event_id>] —
    that id is what update_calendar_event and delete_calendar_event need, so list
    first when changing or removing something.

    `query` is a LITERAL text match against the title, description and location —
    not a concept search. "game" matches only events with the word "game" in them
    and silently misses "MUS JV/9th Football vs ECS". For a question like "do I have
    any football games?", either omit `query` and pass an explicit wide time_min /
    time_max, or search on a distinctive word that appears in EVERY event you want
    (here, "football"). A too-specific term returns a short list that looks complete
    and is not.

    Passing any `query` widens the default window to ~6 months. If you need a
    different span, set time_min/time_max explicitly rather than relying on defaults.
    """
    args = {
        k: v
        for k, v in {
            "time_min": time_min,
            "time_max": time_max,
            "query": query,
            "max_results": max_results,
        }.items()
        if v is not None
    }
    return run_calendar_tool("list_calendar_events", _uid(state), args)


@tool
def create_calendar_event(
    summary: str,
    start: str,
    end: str,
    state: Annotated[dict, InjectedState],
    description: str | None = None,
    location: str | None = None,
) -> str:
    """Add one event to the user's calendar.

    start/end are naive local ISO 8601 datetimes with NO timezone offset (e.g.
    "2026-08-21T19:00:00"), or "YYYY-MM-DD" for an all-day event. Resolve relative
    dates ("next Friday") against the current date given in the prompt; if a date or
    time is genuinely ambiguous, ask rather than guessing.

    To add several events, emit one call per event in the same response — they run
    together. Confirm with the user before creating a large batch.
    """
    args = {"summary": summary, "start": start, "end": end}
    if description is not None:
        args["description"] = description
    if location is not None:
        args["location"] = location
    return run_calendar_tool("create_calendar_event", _uid(state), args)


@tool
def update_calendar_event(
    event_id: str,
    state: Annotated[dict, InjectedState],
    summary: str | None = None,
    start: str | None = None,
    end: str | None = None,
    location: str | None = None,
    description: str | None = None,
) -> str:
    """Change an existing calendar event. Set only the fields being changed.

    Needs the event_id from list_calendar_events. start/end are naive local ISO 8601
    datetimes with NO timezone offset.
    """
    args = {"event_id": event_id}
    for key, value in (
        ("summary", summary), ("start", start), ("end", end),
        ("location", location), ("description", description),
    ):
        if value is not None:
            args[key] = value
    return run_calendar_tool("update_calendar_event", _uid(state), args)


@tool
def delete_calendar_event(event_id: str, state: Annotated[dict, InjectedState]) -> str:
    """Remove an event from the user's calendar.

    Needs the event_id from list_calendar_events. Always confirm with the user
    before deleting.
    """
    return run_calendar_tool("delete_calendar_event", _uid(state), {"event_id": event_id})


# --- REIT research reports (read-only, global) ------------------------------
#
# "ARR", "ARMOUR" and "ARMOUR Residential REIT" are the same issuer (ARR);
# "ORC", "Orchid", "Orchid Island" and "Orchid Island Capital" are the same (ORC).


@tool
def list_reit_issuers(state: Annotated[dict, InjectedState]) -> str:
    """List which REITs have research report coverage. Takes no arguments."""
    return run_reit_tool("list_reit_issuers", _uid(state), {})


@tool
def list_reit_reports(
    reit_symbol: str,
    state: Annotated[dict, InjectedState],
    limit: int | None = None,
) -> str:
    """List the research reports that exist for a REIT (metadata only, newest first).

    Use for "what reports are available" or when the period asked about is
    ambiguous. Pass the issuer name or symbol as reit_symbol (e.g. "ARR", "ARMOUR",
    "ORC", "Orchid"). Each line begins with the report's [id].
    """
    args: dict = {"reit_symbol": reit_symbol}
    if limit is not None:
        args["limit"] = limit
    return run_reit_tool("list_reit_reports", _uid(state), args)


@tool
def get_reit_report(report_id: str, state: Annotated[dict, InjectedState]) -> str:
    """Fetch one specific REIT research report in full, by its report id.

    Ids may be namespaced, e.g. "arr:<uuid>" or "orc:<uuid>". Get ids from
    list_reit_reports.
    """
    return run_reit_tool("get_reit_report", _uid(state), {"report_id": report_id})


@tool
def get_latest_reit_report(
    reit_symbol: str, state: Annotated[dict, InjectedState]
) -> str:
    """Fetch the newest research report for a REIT, in full.

    Use for "latest", "current", "most recent", or "summarize ARR's report". Pass
    the issuer name or symbol as reit_symbol (e.g. "ARR", "ARMOUR", "ORC",
    "Orchid").
    """
    return run_reit_tool("get_latest_reit_report", _uid(state), {"reit_symbol": reit_symbol})


# --- web --------------------------------------------------------------------


@tool
def search_web(query: str, state: Annotated[dict, InjectedState]) -> str:
    """Search the live internet and get an answer grounded in current results.

    Use this whenever the answer depends on information that is current, local,
    niche, or simply not in the knowledge base — sports rosters, prices, news,
    opening hours, "who won", anything after your training cutoff. If a
    query_knowledge_base result above says nothing relevant was found and the
    question is about the outside world, search rather than answering from memory
    or telling the user you don't know.

    Pass a natural-language question. Returns an answer plus its source URLs; cite
    them in your reply.
    """
    return run_web_tool("search_web", _uid(state), {"query": query})


@tool
def fetch_url(url: str, state: Annotated[dict, InjectedState]) -> str:
    """Read one specific web page and return its text.

    Use when the user gives a URL, or when a search result needs reading in full.
    For open questions use search_web instead — this fetches exactly one page and
    does not find pages.

    Only public http/https pages work. Private, local and internal addresses are
    refused by design.
    """
    return run_web_tool("fetch_url", _uid(state), {"url": url})


# --- the catalog ------------------------------------------------------------

@tool
def get_briefing_preferences(state: Annotated[dict, InjectedState]) -> str:
    """The user's daily briefing settings: which topics they follow, what time it
    is delivered, their timezone, and whether it is emailed. Takes no arguments.

    Use for "what topics am I following", "what's on my daily brief", "when does
    my briefing arrive". Read-only — it cannot change any setting.
    """
    return run_briefing_tool("get_briefing_preferences", _uid(state), {})


@tool
def list_briefing_sources(state: Annotated[dict, InjectedState]) -> str:
    """List the news feeds the user added to their own daily briefing, on top of
    the platform's default feeds. Takes no arguments.

    Use for "what feeds do I have", "where does my briefing get news from".
    """
    return run_briefing_tool("list_briefing_sources", _uid(state), {})


@tool
def get_latest_briefing(
    state: Annotated[dict, InjectedState],
    briefing_date: str | None = None,
) -> str:
    """Read the user's most recent daily briefing in full — the Top stories and
    the Deep Dive, with sources.

    Pass briefing_date as YYYY-MM-DD for a specific day; omit it for the latest.
    Use for "what was in my briefing", "what did my brief say today".
    """
    args = {"briefing_date": briefing_date} if briefing_date else {}
    return run_briefing_tool("get_latest_briefing", _uid(state), args)


@tool
def search_briefings(
    query: str,
    state: Annotated[dict, InjectedState],
    limit: int | None = None,
) -> str:
    """Search headlines across the user's past daily briefings.

    Use for "did my briefing cover the Fed", "have I seen anything about Georgia
    football lately". Matches on headline text only.
    """
    args: dict = {"query": query}
    if limit is not None:
        args["limit"] = limit
    return run_briefing_tool("search_briefings", _uid(state), args)


@tool
def add_briefing_topic(topic: str, state: Annotated[dict, InjectedState]) -> str:
    """Add a topic to the user's daily briefing so future briefings cover it.

    Use whenever the user asks to add, follow, track or start covering something
    in their brief — "add baseball cards", "follow the Fed", "I want more about
    Georgia football". One topic per call; call it repeatedly for several.
    This CHANGES a setting, so it is proposed for confirmation before it runs.
    """
    return run_briefing_tool("add_briefing_topic", _uid(state), {"topic": topic})


@tool
def remove_briefing_topic(topic: str, state: Annotated[dict, InjectedState]) -> str:
    """Remove a topic from the user's daily briefing.

    Use for "stop covering X", "drop X from my brief", "I don't care about X any
    more". This CHANGES a setting, so it is proposed for confirmation first.
    """
    return run_briefing_tool("remove_briefing_topic", _uid(state), {"topic": topic})


ALL_TOOLS = [
    read_user_note,
    search_user_vault,
    write_user_note,
    query_knowledge_base,
    list_calendar_events,
    create_calendar_event,
    update_calendar_event,
    delete_calendar_event,
    list_reit_issuers,
    list_reit_reports,
    get_reit_report,
    get_latest_reit_report,
    get_briefing_preferences,
    list_briefing_sources,
    get_latest_briefing,
    search_briefings,
    add_briefing_topic,
    remove_briefing_topic,
    search_web,
    fetch_url,
]

TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
CATALOG_TOOL_NAMES = frozenset(TOOLS_BY_NAME)

# Tools that CHANGE something. Used to gate confirmation/batching policy — a plan
# that only reads can run freely; one that writes should be shown to the user first.
WRITE_TOOLS = frozenset({
    *BRIEFING_WRITE_TOOLS,
    "write_user_note",
    "create_calendar_event",
    "update_calendar_event",
    "delete_calendar_event",
})
