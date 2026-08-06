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

from tools.google_calendar import run_calendar_tool
from tools.graphrag import run_graphrag_tool
from tools.reit_research import run_reit_tool
from tools.user_vault import run_vault_tool

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


# --- the catalog ------------------------------------------------------------

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
]

TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
CATALOG_TOOL_NAMES = frozenset(TOOLS_BY_NAME)

# Tools that CHANGE something. Used to gate confirmation/batching policy — a plan
# that only reads can run freely; one that writes should be shown to the user first.
WRITE_TOOLS = frozenset({
    "write_user_note",
    "create_calendar_event",
    "update_calendar_event",
    "delete_calendar_event",
})
