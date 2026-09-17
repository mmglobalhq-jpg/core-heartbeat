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

from tools.attachments import run_attachment_tool
from tools.daily_briefing import BRIEFING_WRITE_TOOLS, run_briefing_tool
from tools.google_calendar import run_calendar_tool
from tools.reit_research import run_reit_tool
from tools.user_vault import run_vault_tool
from tools.flights import run_flight_tool
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

    All-day date ranges are INCLUSIVE of both ends. A holiday running Monday to
    Friday 3-7 August is start="2026-08-03", end="2026-08-07" — one call, not five.
    A single all-day event repeats the date: start=end="2026-08-03". Never collapse
    a multi-day range to its first day, and never split one into separate events
    per day.

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
    niche, or otherwise outside your training — sports rosters, prices, news,
    opening hours, "who won", anything after your training cutoff. Search rather
    than answering from memory or telling the user you don't know.

    Pass a natural-language question. Returns an answer plus its source URLs; cite
    them in your reply.
    """
    return run_web_tool("search_web", _uid(state), {"query": query})


@tool
def find_sources(
    query: str,
    state: Annotated[dict, InjectedState],
    max_results: int | None = None,
) -> str:
    """Get a list of web pages to READ, instead of a summary of them.

    Use this when the answer is a SPECIFIC detail that a summary tends to round
    off — a schedule, a table, opening hours, a fare, a roster, a figure, an
    address. Then call fetch_url on the one or two most promising results to read
    the actual page.

    search_web is the right tool when a summary IS the answer ("who won", "what is
    X"). This one is right when you need to look at the source. If a search_web
    result came back vague, said the specific information "was not available", or
    told the user to go and check a website — that is the signal to use this and
    read the page yourself rather than passing the hedge along.

    Returns results with their site names. The numbering is NOT a ranking — pick by
    which site is authoritative for the question (the organisation's own site over
    an aggregator or a social media page). If a fetched page has little usable
    text, read the next source. The links expire, so fetch on the same turn.
    """
    return run_web_tool(
        "find_sources", _uid(state), {"query": query, "max_results": max_results}
    )


@tool
def fetch_url(url: str, state: Annotated[dict, InjectedState]) -> str:
    """Read one specific web page and return its text.

    Use when the user gives a URL, or to read a result from find_sources — that
    pairing is how you get a specific detail off a page rather than a summary of
    it. For open questions use search_web instead; this fetches exactly one page
    and does not find pages.

    Only public http/https pages work. Private, local and internal addresses are
    refused by design.
    """
    return run_web_tool("fetch_url", _uid(state), {"url": url})


# --- travel -----------------------------------------------------------------


@tool
def search_flights(
    origins: str,
    destination: str,
    departure_date: str,
    state: Annotated[dict, InjectedState],
    earliest_departure_time: str | None = None,
    adults: int = 1,
) -> str:
    """Find REAL bookable flights for a date — airlines, departure and arrival
    times, stops, duration and fares.

    Use this for ANY question about catching a flight. search_web cannot answer
    these: it returns a prose summary and will say schedules "are not available",
    which is not an answer to "what time can I fly".

    origins: one or more IATA airport codes, space or comma separated ("SAV CHS
    HHH"). Pass EVERY airport within a reasonable drive, not just the closest one —
    the nearest field is often tiny and the airport an hour away is the one with
    service. Ranking is done on the itineraries that come back.
    destination: one IATA code ("MEM").
    departure_date: YYYY-MM-DD.
    earliest_departure_time: optional local clock time ("13:30" or "1:30 PM") —
    nothing departing before it is returned. Use it to respect a commitment the
    user has to finish first, allowing for drive time and check-in.

    Times returned are LOCAL to each airport. If it returns no flights or an
    error, say so plainly; never fall back to describing which airlines "generally"
    serve a route.
    """
    return run_flight_tool(
        "search_flights",
        _uid(state),
        {
            "origins": origins,
            "destination": destination,
            "departure_date": departure_date,
            "earliest_departure_time": earliest_departure_time,
            "adults": adults,
        },
    )


# --- the catalog ------------------------------------------------------------

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
def reread_attachment(
    doc_id: str,
    question: str,
    state: Annotated[dict, InjectedState],
) -> str:
    """Look again at an image the user attached EARLIER in this conversation.

    You are shown an attached image only on the turn it is sent. On any later turn
    you cannot see it. If the user refers back to an image — "check the image
    again", "what about X on the schedule", "you got that date wrong" — you MUST
    call this rather than answering from memory or from the extracted text. Recalling
    what an image said is how wrong dates and figures get invented.

    doc_id comes from the attachment list given in the prompt. Ask one specific
    question: "what dates are listed for Fall Break?" beats "what does this say".

    If it reports that the image cannot be viewed or read, tell the user that
    plainly. Never fill the gap with a plausible answer.
    """
    return run_attachment_tool(
        "reread_attachment", _uid(state), {"doc_id": doc_id, "question": question}
    )


ALL_TOOLS = [
    reread_attachment,
    read_user_note,
    search_user_vault,
    write_user_note,
    list_calendar_events,
    create_calendar_event,
    update_calendar_event,
    delete_calendar_event,
    list_reit_issuers,
    list_reit_reports,
    get_reit_report,
    get_latest_reit_report,
    get_latest_briefing,
    search_briefings,
    search_web,
    find_sources,
    fetch_url,
    search_flights,
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
