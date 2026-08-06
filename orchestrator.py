"""LangGraph orchestration engine for core-heartbeat.

Feature 003 introduced a cyclic graph (supervisor -> local_llm | tool_execution
| END). Feature 004 makes the Supervisor node **model-driven**: it asks Gemini
2.5 Flash (via the Google GenAI SDK) for the routing decision, enforcing a strict
output schema. All model-call failures degrade to a safe `finish` so the graph
always terminates. local_llm and tool_execution remain deterministic stubs.

Termination is still guaranteed three ways: the Supervisor's finish/degrade
decision, the MAX_STEPS bound (checked before any model call), and LangGraph's
recursion_limit. See specs/003-* and specs/004-api-supervisor/.
"""

import asyncio
import json
import logging
import operator
import os
import threading
import time
import re
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Literal, get_args, get_origin

from typing_extensions import TypedDict

from google import genai
from google.genai import errors, types
import httpx

from langchain_core.callbacks import adispatch_custom_event
from langchain_core.callbacks.manager import dispatch_custom_event
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph

from auth import SANDBOX_USER_ID
from services import pending_plans
from services.storage_sync import sync_user_vault, upload_user_file
from tools.user_vault import USER_VAULT_TOOLS, read_note, run_vault_tool, write_note
from tools.web_tools import WEB_TOOL_REGISTRY, run_web_tool
from tools.graphrag import GRAPHRAG_TOOL_REGISTRY, kb_configured, run_graphrag_tool
from tools.google_calendar import CALENDAR_TOOL_REGISTRY, run_calendar_tool
from tools.catalog import ALL_TOOLS, WRITE_TOOLS
from tools.reit_research import (
    REIT_TOOL_REGISTRY,
    looks_like_reit_reference,
    run_reit_tool,
)
from models import (
    HistoryTurn,
    IntentPayload,
    MemoryExtraction,
    Message,
    OrchestrationOutcome,
    RoutingDecision,
    RoutingFailure,
    TokenUsage,
    ToolArgs,
    WorkerFailure,
)

logger = logging.getLogger(__name__)

# --- constants --------------------------------------------------------------

MAX_STEPS = 8          # graceful step bound (Supervisor finishes at/after this)
# Concurrency for a multi-call turn. The structured-output router emits exactly one
# tool call per round-trip, so adding a schedule meant one supervisor hop per event
# and MAX_STEPS ran out around the 7th game. Native tool calling lets the model emit
# every call in one response; this bounds how many actually run at once, because
# "add my season" against Google Calendar is a burst of writes to one API.
MAX_PARALLEL_TOOL_CALLS = int(os.environ.get("MAX_PARALLEL_TOOL_CALLS", "4"))
# How many times one tool may run in a single turn. Enough for a genuine retry with
# different arguments (a wider date range, different search terms); far below
# MAX_STEPS, so a loop is cut short by this rather than by the step bound, which
# ends the turn with no reply at all.
MAX_SAME_TOOL_CALLS = int(os.environ.get("MAX_SAME_TOOL_CALLS", "2"))
# Native tool calling: schemas generated from tools/catalog.py and several calls
# per response. Off by default — this is the hot path for every chat turn, so the
# switch is an env var and rollback needs no redeploy.
NATIVE_TOOL_CALLING = os.environ.get("NATIVE_TOOL_CALLING", "").strip().lower() in ("1", "true", "yes")
RECURSION_LIMIT = 25   # hard LangGraph catch
HISTORY_LIMIT = 10     # max prior turns seeded from IntentPayload.history (token/latency bound)
DOC_CHAR_BUDGET = 12000  # max chars of attached-document text injected (local model window is small)
MAX_DOCS_PER_TURN = 10   # cap attached docs per message

# --- attached IMAGES (vision) ------------------------------------------------
# Images are sent to the model IN ADDITION to their docling-extracted text, not
# instead of it: for receipts and dense tables the OCR text is frequently more
# accurate than vision alone, and if the image path fails for any reason the turn
# degrades to exactly the pre-vision behaviour.
IMAGE_MEDIA_TYPES = frozenset({"image/png", "image/jpeg"})
MAX_IMAGES_PER_TURN = 3      # token/cost bound; images are ~1-2k tokens each
MAX_IMAGE_EDGE_PX = 1568     # Anthropic's recommended long-edge cap; larger is downscaled
MAX_IMAGE_BYTES = 4_000_000  # skip anything still over this AFTER downscaling
MODEL_NAME = "gemini-2.5-flash"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
REQUEST_TIMEOUT_MS = 10_000  # bound each model call (FR-006); milliseconds

# Bounded retry for TRANSIENT Supervisor routing failures before degrading. Only
# categories that can plausibly succeed on a re-attempt are retried; auth /
# missing_credential are terminal (a retry cannot fix them).
MAX_ROUTING_RETRIES = 2  # up to 1 + 2 = 3 attempts per Supervisor turn
RETRYABLE_ROUTING_CATEGORIES = frozenset({"timeout", "network", "invalid_output"})

# Multi-model Supervisor support (feature 006). A caller-selected
# `model_preference` on the intent picks the provider; each maps to a provider id
# and the provider's real API model id. Unknown preferences fall back to the
# default. OpenAI/Anthropic SDKs are imported lazily in their client constructors,
# so the service boots without them and a missing SDK degrades like a missing key.
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
DEFAULT_MODEL_PREFERENCE = "gemini-2.5-flash"
MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    # dropdown value       -> (provider, provider API model id)
    "gemini-2.5-flash": ("gemini", "gemini-2.5-flash"),
    "gpt-4o-mini": ("openai", "gpt-4o-mini"),
    # Claude 3.5 Haiku reached end-of-life 2026-02-19 (claude-3-5-haiku-latest /
    # -20241022 now 404). claude-haiku-4-5 is the current Haiku (its documented
    # drop-in replacement); alias used to match the other registry entries.
    "claude-3.5-haiku": ("anthropic", "claude-haiku-4-5"),
}
# Shared JSON Schema for the routing decision, reused as OpenAI's json_schema and
# Anthropic's tool input_schema so every provider is forced to the same shape.
# Feature 007: when next_node == "tool_execution" the model also emits tool_name +
# tool_args (the vault tool call). tool_args is a fixed, typed object so it maps
# onto every provider's structured output; only the field the tool needs is set.
#
# DERIVED from the Pydantic models rather than hand-written. It used to be a literal
# dict, and it silently went stale: the four Google Calendar tools were added to
# `RoutingDecision.tool_name` and to the Supervisor's prompt, but nobody updated the
# copy here. The result was invisible on the default Gemini path (which validates
# against `RoutingDecision` directly) and severe elsewhere — OpenAI's `strict: true`
# hard-enforces the enum, so the model literally could not name a calendar tool, and
# `additionalProperties: False` rejected every calendar argument.
#
# Deriving from the same object the response is validated against means the wire
# schema and the validator cannot disagree. Registry drift is caught by
# tests/test_routing_vocabulary.py, which asserts these match what's dispatchable.


def _literal_values(annotation: object) -> list:
    """Every ``Literal`` value in a possibly-Optional/Union annotation, in order."""
    out: list = []
    if get_origin(annotation) is Literal:
        out.extend(get_args(annotation))
    else:
        for arg in get_args(annotation):
            out.extend(_literal_values(arg))
    seen: set = set()
    return [v for v in out if not (v in seen or seen.add(v))]


def _compact_type(prop: dict) -> dict:
    """Pydantic's ``anyOf`` nullable form -> the compact ``{"type": [...]}`` shape
    that OpenAI's json_schema and Anthropic's input_schema both accept."""
    types = [s["type"] for s in prop.get("anyOf", [prop]) if "type" in s]
    return {"type": types[0] if len(types) == 1 else types}


def _build_routing_json_schema() -> dict:
    names = _literal_values(RoutingDecision.model_fields["tool_name"].annotation)
    args_props = ToolArgs.model_json_schema()["properties"]
    return {
        "type": "object",
        "properties": {
            "next_node": {
                "type": "string",
                "enum": _literal_values(RoutingDecision.model_fields["next_node"].annotation),
            },
            "tool_name": {
                "type": ["string", "null"],
                # None stays last so the "no tool" choice reads the same as before.
                "enum": [*(n for n in names if n is not None), None],
            },
            "tool_args": {
                "type": "object",
                "properties": {k: _compact_type(v) for k, v in args_props.items()},
                "additionalProperties": False,
            },
        },
        "required": ["next_node"],
        "additionalProperties": False,
    }


ROUTING_JSON_SCHEMA: dict = _build_routing_json_schema()

# Memory extractor (feature 008). Shared JSON Schema for the silent profile
# builder's structured output, reused as OpenAI's json_schema and Anthropic's tool
# input_schema (Gemini uses the MemoryExtraction Pydantic model directly).
MEMORY_EXTRACTION_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "preference_type": {
            "type": "string",
            "enum": [
                "favorite", "project_stack", "tool_setting",
                "personal_fact", "workflow", "none",
            ],
        },
        "key_insight": {"type": "string"},
        "value": {"type": "string"},
        "confidence_score": {"type": "number"},
    },
    "required": ["preference_type", "key_insight", "value", "confidence_score"],
    "additionalProperties": False,
}
# Only durable, non-"none" extractions at/above this confidence are persisted.
MEMORY_CONFIDENCE_THRESHOLD = 0.6
# Vault file the extractor upserts learned preferences into (per-user, path-safe).
PREFERENCES_FILE = "user_preferences.md"

# Local Ollama worker (feature 005). All three are read from the environment at
# node-invoke time so tests/deployments can override without rebuilding the graph.
OLLAMA_URL_ENV = "OLLAMA_URL"
OLLAMA_MODEL_ENV = "OLLAMA_MODEL"
OLLAMA_TIMEOUT_MS_ENV = "OLLAMA_TIMEOUT_MS"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
DEFAULT_OLLAMA_TIMEOUT_MS = 120_000  # local 7B generation can be slow; bound it (FR-007)

# Fixed, deterministic per-node usage increment for the stub tool worker node.
TOOL_USAGE = TokenUsage(input_tokens=5, output_tokens=0, total_tokens=5)

WORKER_NODES = ["local_llm", "tool_execution"]

# User-isolated vault filesystem tools exposed to the tool_execution node.
# Keyed by tool name; the node dispatches by name with the run's state-resolved
# user_id (never a caller-supplied one), so every filesystem op is confined to
# the active user's /tmp/vaults/<user_id>/ boundary. See tools/user_vault.py.
TOOL_REGISTRY = {t.name: t for t in USER_VAULT_TOOLS}

# Every tool name tool_execution can actually run. The single gate a requested call
# must pass before reaching a backend, so a hallucinated or stale name is dropped
# rather than dispatched. tests/test_routing_vocabulary.py asserts the Supervisor's
# advertised vocabulary equals this set.
DISPATCHABLE_TOOLS = frozenset(
    set(TOOL_REGISTRY)
    | set(GRAPHRAG_TOOL_REGISTRY)
    | set(CALENDAR_TOOL_REGISTRY)
    | set(REIT_TOOL_REGISTRY)
    | set(WEB_TOOL_REGISTRY)
)

# Name of the LangGraph custom event the local_llm node dispatches per generated
# token; astream_run surfaces it (as on_custom_event) into the SSE stream.
LOCAL_TOKEN_EVENT = "local_llm_token"

# Custom event the tool_execution node dispatches when it runs a vault tool (name +
# args + result). astream_run surfaces it so the UI can show a "reading/searching
# the vault" indicator distinct from assistant tokens (feature 007).
TOOL_CALL_EVENT = "vault_tool_call"


# --- reducers ---------------------------------------------------------------

def add_usage(left: TokenUsage | None, right: TokenUsage | None) -> TokenUsage:
    """Additive reducer for the usage channel: field-wise sum (FR-009)."""
    if left is None:
        return right or TokenUsage()
    if right is None:
        return left
    return TokenUsage(
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
    )


# --- state ------------------------------------------------------------------

class GraphState(TypedDict):
    """State threaded through a run (channel reducers in brackets)."""

    intent: IntentPayload
    # Stable identity of the caller (Supabase-resolved user_id, or the sandbox
    # user for unauthenticated/local calls). Seeded once at run start and read by
    # downstream nodes; no reducer, so it is set-once and carried unchanged.
    user_id: str
    messages: Annotated[list[Message], operator.add]
    # Prior conversation turns supplied by the caller (chat history), seeded once
    # at run start. Kept SEPARATE from `messages` on purpose: `messages` is the
    # this-run working set the Supervisor uses to detect completion (worker
    # replies), so mixing prior turns in there makes it finish before answering
    # the current question. Read only by the answering prompt (_build_local_prompt)
    # to give the model context. Set-once, no reducer.
    prior_context: list[Message]
    # Extracted text of the message's attached documents (budget-capped), injected
    # into the answering prompt so the model can read them. Set-once, no reducer.
    documents: str
    # Attached IMAGE documents for this turn, already downscaled and base64-encoded:
    # [{"media_type": "image/png", "data": "<b64>", "filename": "..."}]. Kept SEPARATE
    # from `documents` so the text path is untouched — an image-free turn behaves
    # byte-identically to before vision existed. Set-once, no reducer.
    document_images: list[dict]
    # Source-document titles from the most recent knowledge_base retrieval, so the
    # streamed answer can cite them at the end. Last-write-wins (set by tool_execution).
    kb_sources: list[str]
    usage: Annotated[TokenUsage, add_usage]
    visited: Annotated[list[str], operator.add]
    step: Annotated[int, operator.add]
    next: str
    status: str
    # Set-once-per-turn by the Supervisor when it routes to tool_execution: the
    # named vault tool + its args ({"name", "args"}) for this turn, or None. A
    # LastValue channel — the Supervisor writes it fresh (dict or None) on every
    # routing turn so tool_execution never replays a stale request. Feature 007.
    tool_request: dict | None
    # Set-once-per-turn by the Supervisor when the model emits native tool calls:
    # a list of {"name", "args"}, possibly several. Same LastValue semantics as
    # `tool_request`, which it supersedes when present — one channel per routing
    # style rather than overloading one, so the structured-output path keeps its
    # exact shape and behaviour while both are supported.
    tool_calls: list[dict] | None
    # Writes proposed but NOT executed, awaiting the user's go-ahead. Set by the
    # write gate; local_llm renders it as the proposal. Same LastValue semantics as
    # the two channels above — written fresh every routing turn so a declined plan
    # is never replayed.
    pending_plan: list[dict] | None
    # Signatures of tool calls already executed THIS TURN, appended by
    # tool_execution. The native router re-emits a call whenever the result doesn't
    # answer the question — asking "what football games are on my calendar?" with no
    # football games on it produced list_calendar_events four times until MAX_STEPS
    # halted the turn and the user got no reply at all. An empty result IS the
    # answer; this is what lets the graph notice the call already happened.
    executed_calls: Annotated[list[str], operator.add]
    # A short instruction for local_llm about the state of a proposed plan (e.g. the
    # user approved something we no longer hold). Keeps the composer from narrating
    # actions that were never dispatched.
    plan_note: str | None
    # Set when the turn hit MAX_STEPS and is composing from partial results. Tells
    # local_llm to answer from what it has AND say it didn't finish — a partial
    # answer presented as complete is the failure mode this whole area keeps
    # producing.
    truncated: bool


# --- model client (feature 004; multi-provider in feature 006) --------------

# provider -> (key, client). One memoized client per provider, rebuilt if its key
# changes. Keyed by provider so gemini/openai/anthropic clients coexist.
_client_cache: dict[str, tuple[str, object]] = {}


def _resolve_model(model_preference: str | None) -> tuple[str, str]:
    """Map a caller preference to (provider, api_model_id); unknown -> default."""
    pref = model_preference or DEFAULT_MODEL_PREFERENCE
    return MODEL_REGISTRY.get(pref, MODEL_REGISTRY[DEFAULT_MODEL_PREFERENCE])


def _construct_gemini(key: str) -> object:
    return genai.Client(api_key=key)


def _construct_openai(key: str) -> object:
    from openai import OpenAI  # lazy: keep the service bootable without the SDK

    return OpenAI(api_key=key, timeout=REQUEST_TIMEOUT_MS / 1000)


def _construct_anthropic(key: str) -> object:
    from anthropic import Anthropic  # lazy import (see above)

    return Anthropic(api_key=key, timeout=REQUEST_TIMEOUT_MS / 1000)


_PROVIDERS: dict[str, tuple[str, object]] = {
    "gemini": (GEMINI_API_KEY_ENV, _construct_gemini),
    "openai": (OPENAI_API_KEY_ENV, _construct_openai),
    "anthropic": (ANTHROPIC_API_KEY_ENV, _construct_anthropic),
}


def get_client(model_preference: str | None = DEFAULT_MODEL_PREFERENCE) -> object | None:
    """Construct (and memoize) the provider client for the selected model.

    The provider is derived from `model_preference` (feature 006). Returns None if
    the provider's API key is unset/blank OR its SDK is not installed, so both a
    missing credential and a missing SDK become a categorized `missing_credential`
    degrade and the service stays bootable without any key/SDK (FR-003). One
    client is cached per provider, keyed on the key value (a changed key rebuilds).
    The key is passed to the SDK and never logged.
    """
    provider, _ = _resolve_model(model_preference)
    key_env, construct = _PROVIDERS[provider]
    key = os.environ.get(key_env)
    if not key or not key.strip():
        return None
    cached = _client_cache.get(provider)
    if cached is None or cached[0] != key:
        try:
            client = construct(key)
        except Exception:
            # SDK missing or client could not be built -> degrade like a missing key.
            return None
        _client_cache[provider] = (key, client)
    return _client_cache[provider][1]


def _load_user_profile(user_id: str) -> str:
    """Read the caller's cross-session profile (user_preferences.md), or "".

    Uses the path-safe, user-isolated read_note so it can only ever read within
    ``/tmp/vaults/<user_id>/``. Any miss (no profile written yet) or read error
    yields an empty string — the prompt simply omits the profile block.
    """
    try:
        return read_note(user_id, PREFERENCES_FILE).strip()
    except Exception:
        return ""


def _user_profile_block(user_id: str) -> str:
    """A '### USER PROFILE & LONG-TERM PREFERENCES' context block, or "" if none.

    Injected into the Supervisor and local worker prompts so a brand-new session
    is aware of the user's durable preferences from turn one, without any node
    having to call a filesystem tool to fetch them.
    """
    profile = _load_user_profile(user_id)
    if not profile:
        return ""
    return (
        "\n### USER PROFILE & LONG-TERM PREFERENCES\n"
        "The block below is this user's PERMANENT, cross-session identity and "
        "preferences, already loaded from their profile. Treat it as durable "
        "ground truth and use it to guide your decision and responses. It is "
        "already provided here — do NOT call a tool to fetch it.\n"
        f"{profile}\n\n"
    )


def _render_history(messages) -> str:
    """Render Message objects as "{source}: {content}" lines (the shared history
    format used by the supervisor + inference prompts)."""
    return "\n".join(f"{m.source}: {m.content}" for m in messages)


DEFAULT_TZ_ENV = "DEFAULT_TIMEZONE"


def _now_context(intent) -> str:
    """A 'current date & time' grounding line for every prompt so the model can
    resolve relative/partial dates and local times. Uses the caller's timezone
    (IntentPayload.timezone, from the browser); falls back to DEFAULT_TIMEZONE or
    UTC. The orchestrator ALWAYS carries the current time — this is injected into
    both the supervisor (which emits tool date args) and the compose prompt.
    """
    tzname = (getattr(intent, "timezone", None) or os.environ.get(DEFAULT_TZ_ENV) or "UTC").strip()
    try:
        tz = ZoneInfo(tzname)
    except Exception:  # unknown zone -> UTC
        tzname, tz = "UTC", ZoneInfo("UTC")
    now = datetime.now(tz)
    off = now.strftime("%z")
    off = f"{off[:3]}:{off[3:]}" if len(off) == 5 else (off or "+00:00")
    return (
        f"Current date & time: {now.strftime('%A')}, {now.strftime('%Y-%m-%d %H:%M')} "
        f"in {tzname} (UTC{off}). Resolve any date/time the user gives relative to "
        f"this: a date with no year means the NEXT upcoming occurrence (this year if "
        f"it is still in the future, otherwise next year); read clock times in the "
        f"user's timezone above.\n"
    )


def _build_prompt(state: GraphState) -> str:
    """Deterministic routing prompt derived from the intent + message history.

    The prompt carries an explicit completion policy: the Supervisor must choose
    `finish` once the intent has already been answered by a worker, and must not
    re-dispatch a worker that has already replied. Without this, a model tends to
    keep routing a general chat back to local_llm until the MAX_STEPS guard fires
    (observed with gemini-2.5-flash). A running worker-reply count is surfaced so
    the decision does not depend on the model re-reading the whole history.
    """
    intent = state["intent"]
    messages = state.get("messages", [])
    # Deliberately THIS RUN's messages only — prior turns are withheld here.
    #
    # This looks like the bug fixed in the native router (which does receive prior
    # turns, so "add those" has an antecedent), but this prompt can choose `finish`
    # and the native one cannot. Measured with prior turns visible here: "what did I
    # just tell you my name was?" routes straight to finish and the user gets no
    # answer at all. The explicit "Answers composed so far: 0" line is not enough to
    # stop it — seeing a prior assistant reply is enough to look answered.
    # See tests/test_orchestrator.py::test_prior_context_reaches_answer_prompt_not_supervisor.
    history = _render_history(messages)
    # Count only ANSWERS (local_llm), not tool runs. A tool_execution result is raw
    # data, not a reply that satisfies the intent; counting it here made the model
    # believe the question was answered and finish before composing anything.
    answers_composed = sum(1 for m in messages if m.source == "local_llm")
    profile_block = _user_profile_block(state.get("user_id", SANDBOX_USER_ID))
    docs_note = (
        "The user attached document(s) to this message; their text is already "
        "available to local_llm — route there to answer from them.\n"
        if state.get("documents")
        else ""
    )
    return (
        "You are the Supervisor in an orchestration graph. Decide the single "
        "next step for this run.\n"
        f"{profile_block}"
        f"{docs_note}"
        "Routing policy:\n"
        "- Route to local_llm to generate a conversational or reasoned reply.\n"
        "- Route to tool_execution to run a tool. You then MUST set tool_name and "
        "tool_args (only the fields that tool needs). Three tool families:\n"
        "  Personal Markdown vault (this user's private notes):\n"
        "    * read_user_note — read one note. tool_args: {\"filename\": <path>}.\n"
        "    * search_user_vault — case-insensitive text/regex search across the "
        "vault's notes. tool_args: {\"query\": <text>}.\n"
        "    * write_user_note — create/update a note. tool_args: "
        "{\"filename\": <path>, \"content\": <text>}.\n"
        "  The vault tools are already scoped to THIS user — never put a user id "
        "or absolute/`..` path in a filename.\n"
        "  Personal knowledge base (this user's saved documents plus shared/global "
        "docs — their \"core knowledge\" that persists across all chats):\n"
        "    * query_knowledge_base — semantic search over the user's knowledge base. "
        "tool_args: {\"query\": <what to look up>}. Returns cited context; already "
        "scoped to THIS user + global docs. Consult it ONCE at the start of a "
        "substantive or topical turn — the user curated this knowledge because they "
        "consider it important, so it should be checked even when you could answer "
        "from general knowledge. This includes FOLLOW-UP turns on an ongoing topic "
        "(e.g. \"give me other options\", \"what about X instead\", \"tell me more\"): "
        "query with terms built from the follow-up PLUS the topic in the conversation "
        "so far. Skip it ONLY for pure small-talk/greetings or a request with no "
        "topical content.\n"
        "    CRITICAL — query the KB AT MOST ONCE per turn. The moment a "
        "query_knowledge_base result already appears in the history for THIS turn "
        "(including a \"No relevant information found\" result), you MUST route to "
        "local_llm to compose the answer — do NOT call query_knowledge_base again, do "
        "NOT re-query with reworded terms. A single retrieval is enough; local_llm "
        "will use that context plus general knowledge to answer.\n"
        "  Google Calendar (this user's own calendar). Use these when the user asks "
        "about their schedule/events/meetings/appointments or wants to add, change, "
        "or cancel something:\n"
        "    * list_calendar_events — view events. tool_args: optional {\"time_min\", "
        "\"time_max\"} (ISO 8601; defaults to the next 7 days), \"query\" (text "
        "search), \"max_results\". Returns events each ending with [id: <event_id>].\n"
        "    * create_calendar_event — add an event. tool_args: {\"summary\", "
        "\"start\", \"end\"} (start/end ISO 8601 datetime, or YYYY-MM-DD for all-day) "
        "plus optional \"description\", \"location\".\n"
        "    * update_calendar_event — change an event. tool_args: {\"event_id\"} plus "
        "the field(s) to change (summary/start/end/location/description).\n"
        "    * delete_calendar_event — remove an event. tool_args: {\"event_id\"}.\n"
        "    To update or delete, you usually must FIRST list_calendar_events to find "
        "the event_id, then call update/delete with that id. Ask the user to confirm "
        "before deleting. For start/end and time_min/time_max emit NAIVE local "
        "datetimes (e.g. 2026-07-20T09:00:00 — NO timezone offset/Z); the calendar "
        "applies the user's timezone and DST. Resolve dates against the current "
        "date/time given below. If a date/time is genuinely ambiguous, ask.\n"
        "  REIT research reports (read-only). Use these — NOT query_knowledge_base — "
        "for any question about a REIT's research reports (what changed, summaries, "
        "which reports exist). \"ARR\", \"ARMOUR\", and \"ARMOUR Residential REIT\" all "
        "mean the same issuer (symbol ARR); \"ORC\", \"Orchid\", \"Orchid Island\", and "
        "\"Orchid Island Capital\" all mean the same issuer (symbol ORC). Pass the issuer "
        "name/symbol as reit_symbol; report ids may be namespaced (e.g. arr:<uuid> or "
        "orc:<uuid>):\n"
        "    * list_reit_issuers — which REITs are covered. tool_args: {} (none).\n"
        "    * list_reit_reports — the reports that exist for a REIT (metadata only, "
        "newest first). Use for \"what reports are available\" or an ambiguous period. "
        "tool_args: {\"reit_symbol\": <e.g. ARR>, optional \"limit\"}. Each line begins "
        "with the report's [id].\n"
        "    * get_latest_reit_report — the newest/current/most-recent report for a "
        "REIT. Use for \"latest\", \"current\", \"most recent\", or \"summarize ARR's "
        "monthly report\". tool_args: {\"reit_symbol\": <e.g. ARR>}.\n"
        "    * get_reit_report — a specific report when you already have its id. "
        "tool_args: {\"report_id\": <id from list_reit_reports>}.\n"
        "    Do NOT claim a report exists unless a tool returned it, and never generate "
        "or alter a report — these tools only read.\n"
        "  The live internet:\n"
        "    * search_web — search the web and get an answer grounded in current "
        "results, with sources. tool_args: {\"query\": <natural-language question>}. "
        "Use for anything current, local, niche, or outside the knowledge base — "
        "sports rosters, prices, news, opening hours, recent events. If a "
        "query_knowledge_base result already said nothing relevant was found and the "
        "question is about the outside world, search instead of answering from "
        "memory or saying you don't know.\n"
        "    * fetch_url — read ONE specific public page. tool_args: {\"url\": <http(s) "
        "URL>}. Use when the user supplies a URL. It does not find pages; use "
        "search_web for that. Private/local addresses are refused.\n"
        "- A tool result is raw DATA, not an answer. After a tool result appears in "
        "the history you MUST either issue another tool call or route to local_llm "
        "to compose the answer from it — NEVER choose finish directly after a "
        "tool_execution result.\n"
        "- Choose finish only once local_llm has produced the answer. If local_llm "
        "has already answered the intent (a chat reply, or a reply composed from a "
        "tool result), you MUST choose finish and never re-dispatch local_llm.\n"
        f"{_now_context(intent)}"
        f"Intent: {intent.intent}\n"
        f"Confidence: {intent.confidence}\n"
        f"Raw input: {intent.raw_input}\n"
        f"Answers composed so far: {answers_composed}\n"
        f"History so far:\n{history or '(none)'}\n"
        "Respond with next_node = one of: local_llm, tool_execution, finish "
        "(plus tool_name + tool_args when tool_execution)."
    )


def _usage_from(source, in_key, out_key, total_key=None, get=getattr) -> TokenUsage:
    """Build a TokenUsage from any provider's usage object/dict. `get` is getattr
    for SDK objects or a dict getter for Ollama's JSON body; `total_key` reads a
    provided total, else it's computed as input+output. Zeros when `source` absent."""
    if source is None:
        return TokenUsage()
    inp = get(source, in_key, 0) or 0
    out = get(source, out_key, 0) or 0
    total = (get(source, total_key, 0) or 0) if total_key else inp + out
    return TokenUsage(input_tokens=inp, output_tokens=out, total_tokens=total)


def _extract_usage(response) -> TokenUsage:
    """Gemini: map response.usage_metadata into TokenUsage (zeros if absent)."""
    return _usage_from(
        getattr(response, "usage_metadata", None),
        "prompt_token_count",
        "candidates_token_count",
        "total_token_count",
    )


def _parse_decision(response) -> RoutingDecision:
    """Validate the model response into a RoutingDecision (raises on invalid)."""
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, RoutingDecision):
        return parsed
    if isinstance(parsed, dict):
        return RoutingDecision.model_validate(parsed)
    text = getattr(response, "text", None)
    if not text:
        raise ValueError("empty model response")
    return RoutingDecision.model_validate(json.loads(text))


def _detail(exc: Exception) -> str:
    """Short, credential-free failure detail."""
    return f"{type(exc).__name__}: {exc}"[:200]


def _categorize_api_error(exc: Exception) -> str:
    """Map a provider API-call exception to a RoutingFailure category (never invalid_output).

    Duck-typed so OpenAI/Anthropic SDK error classes need not be imported: both
    wrap httpx and expose a `status_code`. Used only for call-time failures;
    parsing failures are categorized separately as `invalid_output`.
    """
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    name = type(exc).__name__.lower()
    if "timeout" in name:
        return "timeout"
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None) or getattr(exc, "code", None)
    if status in (401, 403) or "authentication" in name or "permission" in name:
        return "auth"
    return "network"


def request_routing_decision(
    state: GraphState,
    client: object,
    model_preference: str | None = DEFAULT_MODEL_PREFERENCE,
) -> tuple[RoutingDecision | None, RoutingFailure | None, TokenUsage]:
    """Ask the selected provider's model for a routing decision. Never raises.

    Dispatches on the provider derived from `model_preference` (feature 006) and
    normalizes every provider's response into the strict `RoutingDecision`. Returns
    exactly one of (decision, failure) non-None, plus a TokenUsage (zeros when the
    model reports none or on a pre-response failure).
    """
    provider, api_model = _resolve_model(model_preference)
    if provider == "openai":
        return _decide_openai(state, client, api_model)
    if provider == "anthropic":
        return _decide_anthropic(state, client, api_model)
    return _decide_gemini(state, client, api_model)


def _build_native_prompt(state: GraphState) -> str:
    """Prompt for the native tool-calling router.

    Deliberately NOT ``_build_prompt``. That one instructs the model to emit
    next_node/tool_name/tool_args, which actively suppresses native tool calling —
    told to describe a route, the model describes a route instead of calling
    anything, and every turn lands on local_llm with zero tool calls (measured).

    It also omits the prose tool catalogue: with bind_tools the descriptions come
    from tools/catalog.py, and repeating them here would reintroduce exactly the
    duplication that let the calendar tools go stale.
    """
    intent = state["intent"]
    # Prior turns FIRST, then this run's messages. Without the prior turns this
    # prompt showed "Conversation so far: (none)" on every turn, so a request like
    # "now add those to my calendar" arrived as a pronoun with no antecedent and the
    # model — correctly — called nothing. The composer had the history and could only
    # ask a question back, so the user restated, and the loop repeated.
    convo = list(state.get("prior_context", [])) + list(state.get("messages", []))
    history = _render_history(convo)
    profile_block = _user_profile_block(state.get("user_id", SANDBOX_USER_ID))
    docs = state.get("documents", "")
    docs_block = (
        f"The user attached document(s):\n--- ATTACHED DOCUMENTS ---\n{docs}\n"
        f"--- END DOCUMENTS ---\n\n"
        if docs
        else ""
    )
    return (
        "You are the assistant's planning step. Decide which tools, if any, to call "
        "to serve the user's request.\n\n"
        f"{profile_block}"
        f"{docs_block}"
        f"{_now_context(intent)}"
        f"User request: {intent.raw_input}\n"
        f"Conversation so far:\n{history or '(none)'}\n\n"
        "Rules:\n"
        "- Call a tool whenever one can answer or act on the request. Do NOT write a "
        "reply — a separate step composes the wording. Your prose is discarded.\n"
        "- If the request covers SEVERAL items (a schedule with many games, a list of "
        "notes), emit ONE tool call PER ITEM in this single response. Do not do one "
        "and stop, and do not ask which to start with.\n"
        "- If a tool result already appears above, do NOT run the same tool again. "
        "An empty or \"no matching events\" result IS an answer — it means there are "
        "none, and the next step will say so. Re-running the search will not change "
        "it.\n"
        "- If no tool applies — general knowledge, chit-chat, or a question about an "
        "attachment — call nothing.\n"
        "- Never invent ids or dates. Ambiguous ones get resolved against the current "
        "date above; if genuinely unclear, call nothing so the next step can ask.\n"
    )


def _native_chat_model(model_pref: str):
    """A tool-bound LangChain chat model for the routing call, or None.

    Imports are lazy and per-provider so a missing integration package degrades to
    the structured-output path instead of breaking startup — same contract the raw
    SDK clients already have.
    """
    provider, api_model = _resolve_model(model_pref)
    key_env, _ = _PROVIDERS[provider]
    api_key = (os.environ.get(key_env) or "").strip()
    if not api_key:
        return None
    try:
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            model = ChatAnthropic(model=api_model, api_key=api_key, max_tokens=1024)
        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            model = ChatOpenAI(model=api_model, api_key=api_key)
        else:
            from langchain_google_genai import ChatGoogleGenerativeAI
            model = ChatGoogleGenerativeAI(model=api_model, google_api_key=api_key)
    except Exception as exc:  # missing package / bad config -> fall back
        logger.warning("native tool calling unavailable for %s: %s", provider, exc)
        return None
    # Schemas are generated from tools/catalog.py. Nothing is hand-written, so the
    # advertised vocabulary cannot drift from what tool_execution can run.
    return model.bind_tools(ALL_TOOLS)


def _decide_native(
    state: GraphState, model_pref: str
) -> tuple[str | None, list[dict], RoutingFailure | None, TokenUsage]:
    """Route by native tool calling. Returns ``(next_node, tool_calls, failure, usage)``.

    The model either emits tool calls — possibly SEVERAL in one response, which is
    the whole point, since a 12-game schedule used to need 12 supervisor round-trips
    against a bound of 8 — or emits none, which means no tool is needed and the turn
    goes to local_llm to compose. The model's own prose is deliberately discarded:
    composing is local_llm's job with the cheaper streaming model, and that split is
    what keeps this from collapsing into a single-model ReAct agent.
    """
    bound = _native_chat_model(model_pref)
    if bound is None:
        return None, [], RoutingFailure(
            category="missing_credential",
            detail=f"no native tool-calling client for {model_pref!r}",
        ), TokenUsage()

    prompt = _build_native_prompt(state)
    content = _as_content_parts(prompt, _turn_images(state), "langchain")
    try:
        response = bound.invoke([{"role": "user", "content": content}])
    except Exception as exc:  # never crash the graph
        _trace("router.native.FAILED", err=f"{type(exc).__name__}: {exc}")
        return None, [], RoutingFailure(
            category=_categorize_api_error(exc), detail=_detail(exc)
        ), TokenUsage()

    meta = getattr(response, "usage_metadata", None) or {}
    usage = TokenUsage(
        input_tokens=int(meta.get("input_tokens", 0) or 0),
        output_tokens=int(meta.get("output_tokens", 0) or 0),
    )

    calls: list[dict] = []
    for call in getattr(response, "tool_calls", None) or []:
        name = call.get("name")
        if isinstance(name, str) and name in DISPATCHABLE_TOOLS:
            args = call.get("args")
            calls.append({"name": name, "args": args if isinstance(args, dict) else {}})

    _trace("router.native", pid=os.getpid(), n_calls=len(calls),
           names=",".join(c["name"] for c in calls) or "-",
           has_docs=bool(state.get("documents")),
           n_images=len(state.get("document_images") or []),
           prior_turns=len(state.get("prior_context") or []))
    if not calls:
        return "local_llm", [], None, usage
    return "tool_execution", calls, None, usage


def _decide_gemini(
    state: GraphState, client: object, api_model: str
) -> tuple[RoutingDecision | None, RoutingFailure | None, TokenUsage]:
    """Gemini path via the Google GenAI SDK's native structured output (feature 004)."""
    prompt = _build_prompt(state)
    try:
        response = client.models.generate_content(
            model=api_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RoutingDecision,
                http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
            ),
        )
    except httpx.TimeoutException as exc:
        return None, RoutingFailure(category="timeout", detail=_detail(exc)), TokenUsage()
    except errors.ClientError as exc:
        category = "auth" if getattr(exc, "code", None) in (401, 403) else "network"
        return None, RoutingFailure(category=category, detail=_detail(exc)), TokenUsage()
    except errors.APIError as exc:  # ServerError + other API errors
        return None, RoutingFailure(category="network", detail=_detail(exc)), TokenUsage()
    except httpx.HTTPError as exc:  # connect/read/network transport errors
        return None, RoutingFailure(category="network", detail=_detail(exc)), TokenUsage()
    except Exception as exc:  # last-resort safety net: never crash the graph
        return None, RoutingFailure(category="network", detail=_detail(exc)), TokenUsage()

    usage = _extract_usage(response)
    try:
        decision = _parse_decision(response)
    except Exception as exc:  # JSON error / ValidationError / out-of-vocab
        return None, RoutingFailure(category="invalid_output", detail=_detail(exc)), usage
    return decision, None, usage


def _decide_openai(
    state: GraphState, client: object, api_model: str
) -> tuple[RoutingDecision | None, RoutingFailure | None, TokenUsage]:
    """OpenAI path via chat.completions with a strict json_schema response_format."""
    prompt = _build_prompt(state)
    try:
        response = client.chat.completions.create(
            model=api_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "routing_decision",
                    "strict": True,
                    "schema": ROUTING_JSON_SCHEMA,
                },
            },
        )
    except Exception as exc:  # never crash the graph
        return None, RoutingFailure(category=_categorize_api_error(exc), detail=_detail(exc)), TokenUsage()

    usage = _openai_usage(response)
    try:
        content = response.choices[0].message.content
        decision = RoutingDecision.model_validate(json.loads(content))
    except Exception as exc:  # JSON error / ValidationError / out-of-vocab / bad shape
        return None, RoutingFailure(category="invalid_output", detail=_detail(exc)), usage
    return decision, None, usage


def _decide_anthropic(
    state: GraphState, client: object, api_model: str
) -> tuple[RoutingDecision | None, RoutingFailure | None, TokenUsage]:
    """Anthropic path via the Messages API with a forced tool call for structure."""
    prompt = _build_prompt(state)
    tool = {
        "name": "route",
        "description": "Return the single next node for the orchestration graph.",
        "input_schema": ROUTING_JSON_SCHEMA,
    }
    # Attach any images on the first step so the Supervisor can SEE a screenshot when
    # deciding what to do with it. No images -> plain string, exactly as before.
    content = _as_content_parts(prompt, _turn_images(state), "anthropic")
    try:
        response = client.messages.create(
            model=api_model,
            max_tokens=64,
            messages=[{"role": "user", "content": content}],
            tools=[tool],
            tool_choice={"type": "tool", "name": "route"},
        )
    except Exception as exc:  # never crash the graph
        return None, RoutingFailure(category=_categorize_api_error(exc), detail=_detail(exc)), TokenUsage()

    usage = _anthropic_usage(response)
    try:
        block = next(b for b in response.content if getattr(b, "type", None) == "tool_use")
        decision = RoutingDecision.model_validate(block.input)
    except Exception as exc:  # no tool_use block / ValidationError / out-of-vocab
        return None, RoutingFailure(category="invalid_output", detail=_detail(exc)), usage
    return decision, None, usage


def _openai_usage(response) -> TokenUsage:
    """OpenAI: map response.usage into TokenUsage (zeros if absent)."""
    return _usage_from(
        getattr(response, "usage", None),
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    )


def _anthropic_usage(response) -> TokenUsage:
    """Anthropic: map response.usage into TokenUsage (total = input + output)."""
    return _usage_from(
        getattr(response, "usage", None), "input_tokens", "output_tokens"
    )


# --- local Ollama worker (feature 005) --------------------------------------

def _ollama_url() -> str:
    """Local generate endpoint, env-overridable (read at invoke time; FR-010)."""
    return os.environ.get(OLLAMA_URL_ENV) or DEFAULT_OLLAMA_URL


def _ollama_model() -> str:
    """Target local model, env-overridable (FR-010)."""
    return os.environ.get(OLLAMA_MODEL_ENV) or DEFAULT_OLLAMA_MODEL


def _ollama_embed_model() -> str:
    """Embed model to pre-warm — mirrors graph-rag's OLLAMA_EMBED_MODEL default."""
    return os.environ.get("OLLAMA_EMBED_MODEL") or "nomic-embed-text"


async def warm_ollama_models() -> None:
    """Best-effort: load the generation + embed models into Ollama at startup so the
    first user doesn't pay the ~15s cold-load (keep_alive holds them resident after).
    Never raises — Ollama may be unreachable at boot; failures are logged and ignored.
    """
    keep_alive = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")
    gen_url = _ollama_url()
    embed_url = gen_url.replace("/api/generate", "/api/embeddings")
    targets = (
        (gen_url, {"model": _ollama_model(), "prompt": "hi", "stream": False,
                   "options": {"num_predict": 1}, "keep_alive": keep_alive}, "generation"),
        (embed_url, {"model": _ollama_embed_model(), "prompt": "warmup",
                     "keep_alive": keep_alive}, "embed"),
    )
    async with httpx.AsyncClient(timeout=180.0) as client:
        for url, body, label in targets:
            try:
                r = await client.post(url, json=body)
                logger.info("ollama warmup (%s): HTTP %s", label, r.status_code)
            except Exception as exc:  # noqa: BLE001 — warmup must never crash startup
                logger.warning("ollama warmup (%s) skipped: %s: %s", label, type(exc).__name__, exc)


def _ollama_timeout_s() -> float:
    """Per-call time bound in seconds, env-overridable (FR-007). Falls back on junk."""
    raw = os.environ.get(OLLAMA_TIMEOUT_MS_ENV)
    try:
        ms = int(raw) if raw else DEFAULT_OLLAMA_TIMEOUT_MS
    except (TypeError, ValueError):
        ms = DEFAULT_OLLAMA_TIMEOUT_MS
    return ms / 1000.0


_ollama_client: httpx.AsyncClient | None = None


def build_ollama_client() -> httpx.AsyncClient:
    """Return the shared, keep-alive Ollama client (perf: avoids building a fresh
    AsyncClient — and paying TCP/TLS setup — on every request). Lazily created and
    reused across requests; callers must NOT wrap it in ``async with`` (that would
    close the shared client). Still the test seam: tests monkeypatch this to return
    a per-call ``httpx.MockTransport`` client (FR-011)."""
    global _ollama_client
    if _ollama_client is None or _ollama_client.is_closed:
        _ollama_client = httpx.AsyncClient(
            timeout=_ollama_timeout_s(),
            limits=httpx.Limits(max_keepalive_connections=32, max_connections=64),
        )
    return _ollama_client


# What the PLATFORM can do — injected into the answering prompt.
#
# WHY THIS EXISTS: this model composes the reply but has no tools bound; the
# Supervisor is the component that calls them. Without this block the composer
# reasons only about its own abilities and denies things the platform can plainly
# do — observed verbatim: "I cannot directly add events to your calendar", said
# while four working calendar tools were registered. That is a false statement
# about the product, not a harmless hedge.
#
# It must NOT claim the composer can act, because it cannot. The correct behaviour
# is to OFFER; the user's confirmation becomes the next turn, which the Supervisor
# routes to the real tool. That also gives writes a natural confirm step.
CAPABILITIES_BLOCK = (
    "\nWhat this assistant can do (via its tools, on the user's confirmation):\n"
    "  - Google Calendar: view, create, update and delete the user's events.\n"
    "  - Knowledge base: search the user's curated documents.\n"
    "  - REIT research: list issuers and read ARMOUR/Orchid research reports.\n"
    "  - Notes: read, search and write the user's personal notes.\n"
    "  - Attachments: read documents and SEE images the user attaches.\n"
    "  - The live internet: search the web for current information, and read a\n"
    "    specific page when given its URL.\n"
    "NEVER tell the user you are unable to do one of the things listed above, and "
    "never tell them to do it manually. Instead, offer concretely and ask them to "
    "confirm — e.g. \"I can add these 12 games to your calendar. Want me to go "
    "ahead?\" — then the next turn performs it. If a request needs details you do not "
    "have, ask for exactly those.\n"
    "You do NOT run tools in this step. Unless a tool result appears above, never "
    "state or imply that an action happened or is happening — no \"proceeding to "
    "add\", \"I'm adding\", \"adding now\", \"done\", \"added\", \"scheduled\". Said "
    "of work that was never dispatched, those are false, and the user stops checking. "
    "Either ask for confirmation, or report what a tool result above actually says.\n\n"
)


def _describe_call(call: dict) -> str:
    """One human-readable line for a proposed tool call.

    The user is being asked to approve these, so the line has to carry what they'd
    need to spot a mistake — the summary and the date for an event, the path for a
    note — not the raw tool name and a JSON blob.
    """
    name, args = call.get("name", "?"), call.get("args") or {}
    if name == "create_calendar_event":
        when = args.get("start", "?")
        return f"Add \"{args.get('summary', 'Untitled')}\" — {when}"
    if name == "update_calendar_event":
        # Show the new VALUES, not just which fields move. "start" tells the user
        # nothing about whether the change is right.
        changes = ", ".join(f"{k} -> {v}" for k, v in args.items() if k != "event_id")
        return f"Change event [id: {args.get('event_id', '?')}] ({changes or 'no fields'})"
    if name == "delete_calendar_event":
        return f"Delete event [id: {args.get('event_id', '?')}]"
    if name == "write_user_note":
        return f"Write note {args.get('filename', '?')}"
    return f"{name} {args}"


def _pending_plan_block(state: GraphState) -> str:
    """Render writes awaiting approval, with instructions to ask rather than claim.

    Nothing here has run. The wording matters: if the model says "done" the user
    will believe events exist that don't, which is worse than not offering at all.
    """
    note = state.get("plan_note")
    if note:
        return f"\n{note}\n\n"
    plan = state.get("pending_plan")
    if not plan:
        return ""
    lines = "\n".join(f"  {i}. {_describe_call(c)}" for i, c in enumerate(plan, 1))
    return (
        f"\nYou have PROPOSED the following {len(plan)} action(s). They have NOT been "
        "performed yet and are waiting on the user:\n"
        f"{lines}\n"
        "List these back to the user clearly, then ask them to confirm before you "
        "carry them out. Do NOT say the actions are done, scheduled, or added — "
        "nothing has happened yet. If any detail looks wrong or ambiguous, point it "
        "out and ask.\n"
        "Where an action shows [id: ...], that is an internal calendar id and means "
        "nothing to the user — find that id in the calendar listing above and name "
        "the event by its title, date and time instead. NEVER ask someone to approve "
        "deleting or changing a bare id: they cannot check it, and a delete cannot be "
        "undone. If an id does not appear in any listing above, say you cannot "
        "identify that event and ask rather than guessing.\n\n"
    )


def _truncation_block(state: GraphState) -> str:
    """Warn the composer that this turn ran out of steps.

    Without it the model writes a confident, complete-sounding answer from partial
    results, which is worse than the bare warning it replaces: the user cannot tell
    that anything is missing.
    """
    if not state.get("truncated"):
        return ""
    return (
        "\nNOTE: this turn reached its internal step limit, so the information above "
        "may be incomplete and some requested actions may not have run. Answer with "
        "what you actually have, then say plainly that you were not able to finish "
        "and suggest the user rephrase or narrow the request. Do NOT present this as "
        "a complete answer, and do NOT claim any action succeeded unless a tool "
        "result above says so.\n\n"
    )


def _build_local_prompt(state: GraphState) -> str:
    """Deterministic inference prompt from the intent + message history.

    Prefixes the user's long-term profile (if any) so generated replies reflect
    durable preferences from the first turn of a new session.
    """
    intent = state["intent"]
    # Prior conversation (chat history) first, then this-run messages (e.g. tool
    # results), so the model answers the current input with full context.
    convo = list(state.get("prior_context", [])) + list(state.get("messages", []))
    history = _render_history(convo)
    profile_block = _user_profile_block(state.get("user_id", SANDBOX_USER_ID))
    # Extracted text of any documents the user attached to this message.
    docs = state.get("documents", "")
    docs_block = (
        f"The user attached document(s); use their contents to answer.\n"
        f"--- ATTACHED DOCUMENTS ---\n{docs}\n--- END DOCUMENTS ---\n\n"
        if docs
        else ""
    )
    return (
        f"{profile_block}"
        f"{docs_block}"
        f"{_now_context(intent)}"
        f"Intent: {intent.intent}\n"
        f"Raw input: {intent.raw_input}\n"
        f"Conversation so far:\n{history or '(none)'}\n"
        f"{_pending_plan_block(state)}"
        f"{_truncation_block(state)}"
        f"{CAPABILITIES_BLOCK}"
        "Answer the user's request above directly, and stay strictly on its "
        "specific subject — do NOT drift onto related-but-different topics or list "
        "things the user didn't ask about. If retrieved knowledge-base or tool "
        "context appears above, ground your answer in it and prefer it over guessing; "
        "you may extend it with general knowledge only when that stays directly on "
        "topic. Keep it focused and concise, with no unrelated filler."
    )


def _extract_ollama_usage(body: dict) -> TokenUsage:
    """Ollama: map prompt_eval_count/eval_count into TokenUsage (total = in + out)."""
    return _usage_from(
        body,
        "prompt_eval_count",
        "eval_count",
        get=lambda d, k, default: d.get(k, default),
    )


async def generate_local(
    state: GraphState,
    client: httpx.AsyncClient,
    on_token: Callable[[str], Awaitable[None]] | None = None,
) -> tuple[str | None, WorkerFailure | None, TokenUsage]:
    """Stream a generation from the local Ollama service. Never raises.

    Issues one streaming POST. Ollama replies with NDJSON — one JSON object per
    line, each carrying an incremental ``response`` chunk, the final one
    ``done: true`` with the token counts. Each chunk is passed to ``on_token``
    (if given) as it arrives — the seam the SSE endpoint uses for per-token
    streaming — and also accumulated into the full reply so the graph state and
    the non-streaming ``/intent`` path are unchanged. Returns exactly one of
    (text, failure) non-None plus a TokenUsage. Bounded by the client's timeout
    (FR-007). See contracts/local_worker.md.
    """
    # The local model (qwen2.5:7b) has no vision. If images are attached, say so
    # explicitly rather than answering from the docling OCR text as though we had
    # looked at the picture — a confident answer about an image nobody saw is worse
    # than an honest failure. Swap in a vision model (llava/qwen2-vl) to lift this.
    if state.get("document_images"):
        return (
            "I can read the text extracted from your image, but the local model "
            "can't see images. The cloud model is unavailable right now — please "
            "try again shortly.",
            None,
            TokenUsage(),
        )
    payload = {
        "model": _ollama_model(),
        "prompt": _build_local_prompt(state),
        "stream": True,
        # Keep the model resident between turns. Measured cold-load added ~15s to
        # TTFT (25s cold vs 10s warm); with the default 5m Ollama TTL an idle chat
        # or an interleaved embed model eviction reloads qwen2.5:7b on the next
        # message. Hold it for OLLAMA_KEEP_ALIVE (default 30m; set "-1" to never
        # unload if the host has the RAM for embed + generation models together).
        "keep_alive": os.environ.get("OLLAMA_KEEP_ALIVE", "30m"),
    }
    parts: list[str] = []
    saw_response = False
    usage_body: dict = {}
    try:
        async with client.stream("POST", _ollama_url(), json=payload) as response:
            if response.status_code // 100 != 2:
                return None, WorkerFailure(
                    category="invalid_output", detail=f"HTTP {response.status_code}"
                ), TokenUsage()
            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)  # JSONDecodeError -> invalid_output below
                if obj.get("error"):
                    return None, WorkerFailure(
                        category="invalid_output", detail=str(obj["error"])[:200]
                    ), TokenUsage()
                if "response" in obj:
                    saw_response = True
                    chunk = obj["response"] or ""
                    if chunk:
                        parts.append(chunk)
                        if on_token is not None:
                            await on_token(chunk)
                # Token counts ride the final done chunk (or the whole body when a
                # server/mocks return a single non-streamed object).
                if obj.get("prompt_eval_count") is not None or obj.get("eval_count") is not None:
                    usage_body = obj
    except httpx.TimeoutException as exc:
        return None, WorkerFailure(category="timeout", detail=_detail(exc)), TokenUsage()
    except httpx.TransportError as exc:  # ConnectError, read/connect transport failures
        return None, WorkerFailure(category="unreachable", detail=_detail(exc)), TokenUsage()
    except json.JSONDecodeError as exc:  # malformed NDJSON line
        return None, WorkerFailure(category="invalid_output", detail=_detail(exc)), TokenUsage()
    except Exception as exc:  # last-resort safety net: never crash the graph
        return None, WorkerFailure(category="unreachable", detail=_detail(exc)), TokenUsage()

    if not saw_response:  # no chunk ever carried a "response" field
        return None, WorkerFailure(
            category="invalid_output", detail="no 'response' field in stream"
        ), TokenUsage()
    return "".join(parts), None, _extract_ollama_usage(usage_body)


# --- chat title generation (local-only) -------------------------------------

_TITLE_MAX_LEN = 48


def _clean_title(raw: str | None) -> str | None:
    """Normalize a model-produced title, or None if it isn't usable.

    Takes the first non-empty line, strips wrapping quotes/backticks and trailing
    punctuation, collapses whitespace, and caps length. Rejects empty output and
    obvious refusals / full sentences, so a chatty model never overwrites a decent
    default with "Sure, here's a title:". None means "keep the current title".
    """
    if not raw:
        return None
    line = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
    if not line or len(line) > 60:  # a long single line is a sentence, not a label
        return None
    line = " ".join(line.split())  # collapse internal whitespace
    # Strip wrapping quotes/backticks/punctuation from both ends in one pass, so
    # order-dependent cases like '"DC Museums".' fully unwrap.
    line = line.strip("\"'`.!?,:;() \t")
    if not line:
        return None
    if line.lower().startswith(("i ", "i'm", "sorry", "here is", "here's", "sure")):
        return None
    return line[:_TITLE_MAX_LEN].strip()


def _build_title_prompt(turns: list[HistoryTurn]) -> str:
    """Prompt the local model for a short Title-Case topic label, nothing else."""
    convo = "\n".join(f"{t.role}: {t.content}" for t in turns)
    return (
        "You name chat conversations. Read the conversation and reply with a "
        "SHORT topic label of 2 to 5 words in Title Case that captures its "
        "subject. Reply with ONLY the label — no quotes, no punctuation, no "
        "explanation, no leading words like 'Title:'.\n\n"
        f"Conversation:\n{convo}\n\nLabel:"
    )


async def generate_title(
    turns: list[HistoryTurn], client: httpx.AsyncClient
) -> str | None:
    """Summarize a conversation into a short title via the local Ollama model.

    One non-streaming call, reusing the local worker's Ollama config
    (_ollama_url/_ollama_model). Never raises: any HTTP error, timeout,
    unreachable service, or unparsable body yields None so the caller simply keeps
    the existing title. Output is sanitized by _clean_title (None if unusable).
    """
    if not turns:
        return None
    payload = {
        "model": _ollama_model(),
        "prompt": _build_title_prompt(turns),
        "stream": False,
        "options": {"num_predict": 24, "temperature": 0.2},
    }
    try:
        response = await client.post(_ollama_url(), json=payload)
        if response.status_code // 100 != 2:
            return None
        body = response.json()
    except Exception:  # timeout / transport / decode — never crash the endpoint
        return None
    if body.get("error"):
        return None
    return _clean_title(body.get("response"))


# --- nodes ------------------------------------------------------------------

def _degraded(step: int, failure: RoutingFailure, usage: TokenUsage | None = None) -> dict:
    """Safe terminal update after a routing failure (FR-005, FR-008).

    Logs the category AND the secret-free detail so a "degraded" outcome is
    diagnosable from the backend logs — the in-band message only carries the
    category. This is the single funnel for every degrade (missing credential and
    all model-call/parse failures).
    """
    logger.warning(
        "supervisor degraded at step %s: %s: %s", step, failure.category, failure.detail
    )
    return {
        "next": "finish",
        "status": "degraded",
        "step": 1,
        "usage": usage or TokenUsage(),
        "messages": [
            Message(source="supervisor", content=f"routing failure: {failure.category}", step=step)
        ],
    }


# Pure greetings / pleasantries / acknowledgements that never need the knowledge
# base. Stored in normalized form (lowercased, punctuation stripped).
_TRIVIAL_TURNS = frozenset({
    "hi", "hey", "hello", "yo", "hiya", "howdy", "sup", "hey there", "hi there",
    "hello there", "good morning", "good afternoon", "good evening", "good night",
    "morning", "evening", "greetings",
    "thanks", "thank you", "thank you so much", "thanks so much", "ty", "thx",
    "much appreciated", "appreciate it", "cheers",
    "bye", "goodbye", "see you", "see ya", "cya", "later", "take care",
    "ok", "okay", "k", "kk", "cool", "nice", "great", "awesome", "perfect",
    "got it", "sounds good", "makes sense", "understood", "gotcha",
    "yes", "no", "yep", "yeah", "yup", "nope", "nah", "sure", "please",
    "how are you", "hows it going", "how s it going", "whats up", "what s up",
    "how are things",
})


def _is_trivial_turn(raw: str) -> bool:
    """True for a pure greeting / pleasantry / acknowledgement — a turn that never
    needs the knowledge base. Exact-match on the WHOLE normalized input, so it never
    fires on a real question that merely starts with a greeting (e.g. "hello, how do
    I roast a chicken?"). Used to skip the forced KB retrieval on trivial turns. Pure.
    """
    norm = " ".join(re.sub(r"[^a-z0-9\s]", " ", (raw or "").lower()).split())
    return norm in _TRIVIAL_TURNS


def supervisor(state: GraphState) -> dict:
    """Model-driven routing hub. Falls back to a safe finish on any failure."""
    step = state["step"]

    # Layer-2 termination + cost guard: never call the ROUTING model past the bound.
    #
    # Hitting the bound used to finish immediately, which meant the turn ended with
    # nothing composed and the user saw only "No reply produced (status:
    # halted_step_bound)" — work had happened, tools had run, and none of it reached
    # them. Observed for real: "what football games are on my calendar?" looped on
    # list_calendar_events until the bound and returned nothing at all.
    #
    # So spend one final step composing from whatever WAS gathered. The
    # `local_llm not in visited` check is load-bearing: without it local_llm returns
    # here at step+1, the bound is still exceeded, and it routes to local_llm forever
    # until RECURSION_LIMIT (25) kills the run — turning a bad turn into a worse one.
    # MAX_STEPS is 8, so one extra node visit has ample headroom.
    if step >= MAX_STEPS:
        if "local_llm" not in state.get("visited", []):
            return {
                "next": "local_llm",
                "status": "halted_step_bound",
                "step": 1,
                "truncated": True,
                "tool_request": None,
                "tool_calls": None,
                "pending_plan": None,
                "plan_note": None,
                "messages": [
                    Message(
                        source="supervisor",
                        content="route -> local_llm (step bound: compose what we have)",
                        step=step,
                    )
                ],
            }
        return {
            "next": "finish",
            "status": "halted_step_bound",
            "step": 1,
            "messages": [Message(source="supervisor", content="route -> finish (step bound)", step=step)],
        }

    # Deterministic finish fast-path (latency/throughput): local_llm composes the
    # single answer per turn, and the completion policy is "finish once local_llm has
    # replied, never re-dispatch it". So once local_llm is in visited the turn is done
    # — finish WITHOUT a routing model call. Saves the final ~1.2s Gemini round-trip on
    # every turn (post-generation, so it frees the worker sooner under concurrent load)
    # and subsumes the anti-reloop guard below for the local_llm case.
    if "local_llm" in state.get("visited", []):
        return {
            "next": "finish",
            "status": "completed",
            "step": 1,
            "usage": TokenUsage(),
            "tool_request": None,
            "tool_calls": None,
            "pending_plan": None,
            "plan_note": None,
            "messages": [Message(source="supervisor", content="route -> finish (fast-path: answered)", step=step)],
        }

    # Deterministic fast-path (latency): the KB is a retrieve-once-then-compose tool,
    # so once its result is in this run's messages the next hop is ALWAYS local_llm.
    # Skipping the model routing call here removes a ~1.2s Gemini round-trip that sits
    # BEFORE generation, cutting time-to-first-token on every KB-grounded turn. The
    # retrieve-once guard below enforces the same transition when the model is asked;
    # this just avoids paying for a decision that's already determined.
    _kb_prefixes = tuple(f"[tool:{n}]" for n in GRAPHRAG_TOOL_REGISTRY)
    if "local_llm" not in state.get("visited", []) and any(
        m.source == "tool_execution" and m.content.startswith(_kb_prefixes)
        for m in state.get("messages", [])
    ):
        return {
            "next": "local_llm",
            "step": 1,
            "usage": TokenUsage(),
            "tool_request": None,
            "tool_calls": None,
            "pending_plan": None,
            "plan_note": None,
            "messages": [Message(source="supervisor", content="route -> local_llm (fast-path: compose KB)", step=step)],
        }

    # Feature 006: the caller's model_preference selects the provider/model.
    model_pref = getattr(state["intent"], "model_preference", None) or DEFAULT_MODEL_PREFERENCE
    provider, _ = _resolve_model(model_pref)
    client = get_client(model_pref)
    if client is None:
        return _degraded(
            step,
            RoutingFailure(
                category="missing_credential",
                detail=f"no client for model_preference={model_pref!r} (provider={provider})",
            ),
        )

    # Native tool calling (NATIVE_TOOL_CALLING=1). The model may emit SEVERAL calls
    # in one response, which is what makes "add my whole schedule" possible inside
    # the step bound. It is normalized into a RoutingDecision so every deterministic
    # guard below — KB retrieve-once, retrieve-first, the anti-reloop rules — keeps
    # working unchanged; the full list rides alongside on `native_calls`.
    #
    # A failure here falls through to the structured-output path rather than
    # degrading the turn, so enabling this can lose latency but not availability.
    native_calls: list[dict] = []
    if NATIVE_TOOL_CALLING:
        nxt_native, calls, native_failure, native_usage = _decide_native(state, model_pref)
        if native_failure is None:
            native_calls = calls
            first = calls[0] if calls else None
            decision = RoutingDecision(
                next_node=nxt_native,
                tool_name=first["name"] if first else None,
                tool_args=ToolArgs(**(first["args"] if first else {})),
            )
            return _finish_routing(state, step, decision, native_usage, native_calls)
        logger.warning(
            "native tool calling failed (%s: %s); falling back to structured output",
            native_failure.category, native_failure.detail,
        )

    # Bounded retry: a transient timeout/network/invalid_output can succeed on a
    # re-attempt, so retry those before degrading (a non-transient failure such as
    # auth breaks out immediately). Usage is accumulated across attempts so token
    # accounting reflects every model call. FR-005/FR-006.
    usage = TokenUsage()
    decision = None
    failure = None
    max_attempts = 1 + MAX_ROUTING_RETRIES
    for attempt in range(1, max_attempts + 1):
        decision, failure, call_usage = request_routing_decision(state, client, model_pref)
        usage = add_usage(usage, call_usage)
        if failure is None:
            break
        if failure.category not in RETRYABLE_ROUTING_CATEGORIES or attempt == max_attempts:
            break
        logger.warning(
            "supervisor routing attempt %d/%d failed (%s: %s); retrying",
            attempt, max_attempts, failure.category, failure.detail,
        )
    if failure is not None:
        return _degraded(step, failure, usage)

    return _finish_routing(state, step, decision, usage, [])


# Write batches at or above this size are proposed before they run. One or two
# wrong events are trivial to delete by hand; a dozen is a mess, and
# create_calendar_event has no undo. Deletes are always confirmed regardless of
# count. Tunable without a rebuild.
WRITE_CONFIRM_THRESHOLD = int(os.environ.get("WRITE_CONFIRM_THRESHOLD", "3"))
ALWAYS_CONFIRM_TOOLS = frozenset({"delete_calendar_event"})

# Short, unambiguous go-aheads. Deliberately narrow: a miss re-proposes (mildly
# annoying), while a false positive runs writes the user didn't authorize. When in
# doubt this must fail toward asking again.
_AFFIRMATIONS = frozenset({
    "y", "ya", "yes", "yes please", "yep", "yeah", "yup", "sure", "ok", "okay",
    "do it", "go ahead", "go for it", "please do", "confirm", "confirmed",
    "add them", "add them all", "add it", "create them", "sounds good",
    "yes do it", "yes go ahead", "yes add them", "proceed", "approved",
})


def _is_affirmation(raw: str) -> bool:
    """Is this message a bare go-ahead rather than a new instruction?

    Length-capped on purpose: "yes, but move the first one to Friday" is a revision,
    not a confirmation, and must go back through planning rather than releasing the
    batch that was proposed before the change.
    """
    cleaned = re.sub(r"[^a-z ]", "", (raw or "").strip().lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return bool(cleaned) and len(cleaned.split()) <= 4 and cleaned in _AFFIRMATIONS


# Plans proposed on one turn and approved on the next. `pending_plan` is per-run
# state and dies with the turn, so without this the confirmation turn reaches the
# router as a bare "yes" and the model has to reconstruct twelve tool calls from the
# prose it wrote earlier. Observed: it doesn't. It replies "Shall I proceed?" and
# nothing is ever dispatched — the user says yes twice and no events are created.
#
# Replaying the STORED calls also means what runs is exactly what the user was shown
# and approved, rather than a re-derivation that might drift from the list they read.
#
# Keyed by user_id because IntentPayload carries no chat id, so two conversations by
# the same user share one slot: approving in one would run the other's plan. Given a
# confirmation lives for seconds that is unlikely, but it is the reason to add a chat
# id here rather than leave this as-is. In-memory, so a restart drops pending plans —
# which fails safe: nothing runs and the user re-asks.
# --- plan tracing -----------------------------------------------------------
#
# The propose/confirm handshake spans two HTTP requests and four components
# (router -> gate -> store -> composer). When it fails the user sees only the last
# line — "I don't have the details" — which is the same symptom whether the router
# never emitted calls, the gate never stored them, the store was cleared, or the
# process restarted in between. This makes each step observable so a failure can be
# located instead of guessed at.
#
# Own handler at INFO with propagate=False, so the trace appears regardless of how
# the app's root logger is configured. PLAN_TRACE=0 disables it.
PLAN_TRACE = (os.environ.get("PLAN_TRACE", "1").strip().lower()
              not in ("0", "false", "no", "off"))
_trace_log = logging.getLogger("plan_trace")
if PLAN_TRACE and not _trace_log.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s [plan-trace] %(message)s"))
    _trace_log.addHandler(_h)
    _trace_log.setLevel(logging.INFO)
    _trace_log.propagate = False


def _short(value: object, limit: int = 60) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


def _trace(event: str, **fields) -> None:
    """One structured line per decision point. Never raises, never logs content
    that isn't needed to diagnose the handshake (no event bodies, no secrets)."""
    if not PLAN_TRACE:
        return
    try:
        rendered = " ".join(f"{k}={_short(v)}" for k, v in fields.items())
        _trace_log.info("%-22s %s", event, rendered)
    except Exception:
        pass


def _store_pending_plan(user_id: str, chat_id: str | None, calls: list[dict]) -> None:
    where = pending_plans.save(user_id, chat_id, calls)
    _trace("store.save", pid=os.getpid(), key=user_id[:8],
           chat=(chat_id or "-")[:8], n=len(calls), backend=where)


def _take_pending_plan(user_id: str, chat_id: str | None) -> list[dict] | None:
    """Pop the plan awaiting approval, if one is still valid. Single-use."""
    calls, source = pending_plans.take(user_id, chat_id)
    if not calls:
        _trace("store.take.MISS", pid=os.getpid(), key=user_id[:8],
               chat=(chat_id or "-")[:8], source=source)
        return None
    _trace("store.take.HIT", pid=os.getpid(), key=user_id[:8],
           chat=(chat_id or "-")[:8], n=len(calls), source=source)
    return calls


def _clear_pending_plan(user_id: str, chat_id: str | None) -> None:
    """Drop any proposal — the user asked for something else instead."""
    pending_plans.clear(user_id, chat_id)


def _needs_confirmation(calls: list[dict]) -> bool:
    """Should this batch be shown to the user before it runs?"""
    if any(c["name"] in ALWAYS_CONFIRM_TOOLS for c in calls):
        return True
    writes = [c for c in calls if c["name"] in WRITE_TOOLS]
    return len(writes) >= WRITE_CONFIRM_THRESHOLD


def _confirmation_given(state: GraphState) -> bool:
    """Did the user just approve a plan we proposed on the previous turn?

    Requires BOTH a bare affirmation now and an assistant turn before it — so an
    opening "yes" in a fresh conversation cannot release a batch of writes.
    """
    raw = (getattr(state["intent"], "raw_input", "") or "").strip()
    if not _is_affirmation(raw):
        return False
    return any(m.source == "assistant" for m in state.get("prior_context", []))


def _finish_routing(
    state: GraphState,
    step: int,
    decision: RoutingDecision,
    usage: TokenUsage,
    native_calls: list[dict],
) -> dict:
    """Apply the deterministic routing guards and build the supervisor's state update.

    Shared by both routing paths so the guards exist once. The native path normalizes
    its (possibly multi-call) result into `decision` before calling this and passes
    the full list as `native_calls`; the structured-output path passes an empty list.
    """
    nxt = decision.next_node

    # Has the KB already been consulted THIS turn? (a query_knowledge_base result in
    # this run's messages). Shared by both KB guards below.
    _kb_prefixes = tuple(f"[tool:{n}]" for n in GRAPHRAG_TOOL_REGISTRY)
    kb_consulted = any(
        m.source == "tool_execution" and m.content.startswith(_kb_prefixes)
        for m in state.get("messages", [])
    )

    # Deterministic KB retrieve-once guard (do not rely on the model to stop).
    # The KB is a consult-once-per-turn tool: once a query_knowledge_base result is
    # already in THIS run's messages, re-dispatching it adds nothing and can loop
    # the supervisor to the MAX_STEPS halt (observed on follow-up questions like
    # "any other options?"). Redirect any repeat KB call to local_llm so it composes
    # from what was already retrieved (plus general knowledge). Runs before the
    # local_llm guard below so an already-answered turn still finishes cleanly.
    if nxt == "tool_execution" and decision.tool_name in GRAPHRAG_TOOL_REGISTRY and kb_consulted:
        nxt = "local_llm"

    # Deterministic retrieve-first guard (do not rely on the model to consult the KB).
    # The user's curated knowledge base MUST be checked before we answer a fresh turn
    # from general knowledge — Gemini follows the "consult the KB" prose only some of
    # the time (observed: it consulted a follow-up but skipped the direct question).
    # So if it routes STRAIGHT to local_llm on the first step of a turn (no worker has
    # run yet) and the KB hasn't been consulted, redirect that first step to a
    # query_knowledge_base call built from the user's input. The retrieve-once guard
    # then blocks a second query, and once the result is in messages this is false so
    # local_llm composes normally. Gated on the KB being configured so it never fires
    # in tests / KB-less deploys; tool turns (vault) route to tool_execution not
    # local_llm, so they are untouched.
    forced_kb_query: str | None = None
    raw = (getattr(state["intent"], "raw_input", "") or "").strip()
    if (
        nxt == "local_llm"
        and not state.get("visited")
        and not kb_consulted
        and kb_configured()
        # Skip the embed+search+rerank on pure greetings/pleasantries — they never
        # need the KB, and forcing it there just adds latency to a trivial reply.
        and not _is_trivial_turn(raw)
        # A clear REIT-report question belongs to the dedicated REIT tools, not the
        # generic KB. Don't preempt it with a forced query_knowledge_base retrieval
        # (the prompt steers the model to a REIT tool; this is the deterministic
        # backstop for the case where it routed straight to local_llm).
        and not looks_like_reit_reference(raw)
        # A question ABOUT AN ATTACHMENT is self-contained — the answer comes from the
        # document/image the user just supplied, not from the curated KB. Forcing a
        # retrieval here searched the KB for things like "can you see this schedule?",
        # which returns whatever is nearest in vector space and then gets cited as the
        # source of an answer that came entirely from the attachment. It also spent an
        # embed+search+rerank on every attachment turn for nothing.
        and not state.get("documents")
        and not state.get("document_images")
    ):
        if raw:
            forced_kb_query = raw
            nxt = "tool_execution"

    # Deterministic anti-reloop guard (do not rely on the model to terminate).
    # Scoped to local_llm: if the model re-dispatches local_llm after it has
    # ALREADY produced a reply this run, override to a clean finish. This kills the
    # observed gemini-2.5-flash trap where a general chat is routed back to
    # local_llm over and over until the MAX_STEPS halt (see _build_prompt's
    # completion policy — this enforces it in code). tool_execution is deliberately
    # NOT guarded here: a tool-calling loop legitimately makes several tool calls
    # (read, then search, ...), each a distinct turn bounded by MAX_STEPS /
    # RECURSION_LIMIT, so a repeat visit must be allowed.
    visited = state.get("visited", [])
    if nxt == "local_llm" and nxt in visited:
        return {
            "next": "finish",
            "status": "completed",
            "step": 1,
            "usage": usage,
            "tool_request": None,
            "tool_calls": None,
            "pending_plan": None,
            "plan_note": None,
            "messages": [
                Message(
                    source="supervisor",
                    content=f"route -> finish (guard: {nxt} already replied)",
                    step=step,
                )
            ],
        }

    # When routing to a tool, thread the named call to tool_execution via the
    # tool_request channel. Written fresh (dict or None) on EVERY routing turn so
    # tool_execution never replays a stale request from an earlier turn.
    tool_request: dict | None = None
    if forced_kb_query is not None:
        # retrieve-first guard fired: synthesize the KB call the model skipped.
        tool_request = {"name": "query_knowledge_base", "args": {"query": forced_kb_query}}
    elif nxt == "tool_execution" and decision.tool_name:
        tool_request = {
            "name": decision.tool_name,
            "args": decision.tool_args.model_dump(exclude_none=True),
        }

    # The multi-call list only survives when the guards left us on tool_execution and
    # didn't synthesize their own call. A guard that redirected to local_llm, or the
    # retrieve-first guard that replaced the model's choice with a KB query, must not
    # be overridden by a stale batch — so drop it in those cases and let the single
    # `tool_request` stand.
    calls = native_calls if (native_calls and nxt == "tool_execution" and forced_kb_query is None) else None
    if calls:
        # The KB is consult-once-per-turn; strip a repeat from the batch rather than
        # discarding the whole batch, so "check my notes AND add these games" keeps
        # its calendar writes.
        if kb_consulted:
            calls = [c for c in calls if c["name"] not in GRAPHRAG_TOOL_REGISTRY] or None

    # Repeat-call guard. The model re-emits a call when the result doesn't answer the
    # question, but "no matching events" IS the answer — it just doesn't look like one.
    # Observed: "what football games are on my calendar?" against a calendar with none
    # produced list_calendar_events four times, exhausted MAX_STEPS, and returned
    # "No reply produced (status: halted_step_bound)" — the user got nothing at all.
    #
    # Identical calls are dropped outright. Same tool with DIFFERENT args is allowed
    # a bounded number of tries, because narrowing a date range or re-searching with
    # other terms is legitimate; looping on it is not.
    if calls:
        already = list(state.get("executed_calls") or [])
        ran_names = [sig.split(":", 1)[0] for sig in already]
        fresh = [
            c for c in calls
            if _call_signature(c["name"], c["args"]) not in already
            and ran_names.count(c["name"]) < MAX_SAME_TOOL_CALLS
        ]
        if len(fresh) != len(calls):
            _trace("gate.repeat_dropped", dropped=len(calls) - len(fresh),
                   kept=len(fresh), already=len(already))
        calls = fresh
        if not calls:
            # Everything requested has already run. Compose from those results
            # rather than asking for them again.
            _trace("gate.all_repeats", nxt="local_llm", already=len(already))
            nxt = "local_llm"
            tool_request = None
            # None, not [] — the channel means "no request", and an empty list
            # reads as one in some checks while being falsy in others.
            calls = None

    # Write gate. Native tool calling can emit a dozen create_calendar_event calls
    # from one sentence, and there is no undo beyond deleting each event by hand —
    # so a batch of writes is PROPOSED, not run, and the user's next message releases
    # it. Reads are never gated.
    user_id = state.get("user_id", SANDBOX_USER_ID)
    chat_id = getattr(state["intent"], "chat_id", None)
    pending_plan: list[dict] | None = None
    plan_note: str | None = None

    raw_in = (getattr(state["intent"], "raw_input", "") or "").strip()
    prior = state.get("prior_context", []) or []
    _trace(
        "gate.enter",
        pid=os.getpid(), key=user_id[:8], raw=raw_in, nxt=nxt,
        n_calls=len(calls or []),
        affirmation=_is_affirmation(raw_in),
        prior_turns=len(prior),
        prior_assistant=sum(1 for m in prior if m.source == "assistant"),
        confirming=_confirmation_given(state),
        chat=(chat_id or "-")[:8],
    )
    # The Supervisor runs once per STEP, not once per turn, and raw_input stays "yes"
    # for the whole turn. So after the approved batch runs, the next step re-entered
    # this branch, found the plan correctly consumed, and reported "I don't have the
    # details" — an apology emitted immediately AFTER the writes succeeded. The user
    # saw only the apology, retried, and accumulated duplicate events.
    #
    # Once tool_execution has run this turn the confirmation is already honoured;
    # there is nothing left to release and nothing to apologise for.
    tools_ran = "tool_execution" in (state.get("visited") or [])
    if tools_ran and _confirmation_given(state):
        _trace("gate.already_honoured", key=user_id[:8], nxt=nxt)

    if _confirmation_given(state) and not tools_ran:
        approved = _take_pending_plan(user_id, chat_id)
        if approved:
            # Run exactly what was shown and agreed to. Not what the model would
            # regenerate now — the user approved a specific list.
            calls, tool_request, nxt = approved, None, "tool_execution"
            _trace("gate.CONFIRMED", n=len(approved),
                   names=",".join(sorted({c["name"] for c in approved})))
        elif not calls:
            # They said yes, but there is nothing to run: the proposal expired, the
            # process restarted, or it was already used. Say so. Answering a "yes"
            # with narration like "proceeding to add them" while dispatching nothing
            # is precisely the failure this gate exists to prevent.
            _trace("gate.CONFIRM_NO_PLAN", key=user_id[:8], chat=(chat_id or "-")[:8])
            plan_note = (
                "The user just approved something, but no pending plan is on record "
                "(it may have expired or already run). Tell them plainly that you do "
                "not have it any more and ask them to restate what they want done. Do "
                "NOT claim anything is being added or has been added."
            )
            nxt = "local_llm"
    elif calls and _needs_confirmation(calls):
        pending_plan = calls
        _trace("gate.PROPOSE", n=len(calls),
               names=",".join(sorted({c["name"] for c in calls})))
        _store_pending_plan(user_id, chat_id, calls)
        calls = None
        tool_request = None
        nxt = "local_llm"
    else:
        # Any other turn means they moved on; a stale proposal must not linger and
        # fire against a later, unrelated "yes".
        _trace("gate.passthrough", n_calls=len(calls or []), nxt=nxt)
        _clear_pending_plan(user_id, chat_id)

    label = f"route -> {nxt}"
    if pending_plan:
        label = f"route -> {nxt} (proposing {len(pending_plan)} writes for confirmation)"
    elif calls:
        label = f"route -> {nxt} ({len(calls)} calls: {', '.join(c['name'] for c in calls)})"
    elif tool_request is not None:
        label = f"route -> {nxt} ({tool_request['name']})"

    update: dict = {
        "next": nxt,
        "step": 1,
        "usage": usage,
        "tool_request": tool_request,
        "tool_calls": calls,
        "pending_plan": pending_plan,
        "plan_note": plan_note,
        "messages": [Message(source="supervisor", content=label, step=step)],
    }
    if nxt == "finish":
        update["status"] = "completed"
    return update


# Answer-composition model. Default "ollama" (local qwen2.5:7b) so tests and offline
# deploys are unchanged. Set COMPOSE_MODEL=gemini-2.5-flash to offload generation to
# the cloud — ~15-20x faster than CPU for long answers, with a local fallback.
COMPOSE_MODEL_ENV = "COMPOSE_MODEL"


def _compose_model() -> str:
    return (os.environ.get(COMPOSE_MODEL_ENV) or "ollama").strip()


def _is_local_compose() -> bool:
    return _compose_model().lower() in ("", "ollama", "local")


async def generate_cloud(
    state: GraphState,
    on_token: Callable[[str], Awaitable[None]] | None = None,
) -> tuple[str | None, WorkerFailure | None, TokenUsage]:
    """Stream the answer from a cloud model (gemini-2.5-flash) instead of the local
    CPU model. Uses the same prompt as generate_local. Never raises. A mid-stream
    failure AFTER any token is kept as a (partial) success; a failure BEFORE the
    first token returns (None, failure, ...) so local_llm can fall back to Ollama.
    """
    model_pref = _compose_model()
    provider, api_model = _resolve_model(model_pref)
    client = get_client(model_pref)
    if client is None:
        return None, WorkerFailure(category="unreachable",
                                   detail=f"no client for compose model {model_pref!r}"), TokenUsage()
    if provider != "gemini":  # only Gemini streaming is wired; fall back otherwise
        return None, WorkerFailure(category="invalid_output",
                                   detail=f"cloud compose unsupported for provider {provider!r}"), TokenUsage()
    prompt = _build_local_prompt(state)
    parts: list[str] = []
    usage = TokenUsage()
    # Disable 2.5-flash "thinking" for composition — we're writing an answer from
    # already-retrieved context, not solving a reasoning task, and thinking adds
    # several seconds of pre-output latency. Guarded: if the SDK/model rejects the
    # config, the except below degrades to local rather than crashing.
    try:
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
    except Exception:  # SDK without ThinkingConfig -> no override
        config = None
    # Attach images so the composed ANSWER can describe what is in the screenshot,
    # not just the text docling pulled out of it. No images -> plain string.
    contents = _as_content_parts(prompt, _turn_images(state), "gemini")
    try:
        stream = await client.aio.models.generate_content_stream(
            model=api_model, contents=contents, config=config
        )
        async for chunk in stream:
            piece = getattr(chunk, "text", None)
            if piece:
                parts.append(piece)
                if on_token is not None:
                    await on_token(piece)
            if getattr(chunk, "usage_metadata", None) is not None:
                usage = _extract_usage(chunk)  # last chunk carries the running totals
    except Exception as exc:  # never crash the graph
        if parts:  # already streamed a partial answer — keep it rather than regress
            return "".join(parts), None, usage
        category = "timeout" if isinstance(exc, httpx.TimeoutException) else "unreachable"
        return None, WorkerFailure(category=category, detail=_detail(exc)), usage
    return "".join(parts), None, usage


async def local_llm(state: GraphState) -> dict:
    """Live local inference via Ollama (feature 005). Degrades safely on any failure.

    On success records the model's generated text + its reported token usage. On
    any failure records a categorized WorkerFailure message with zero usage. Either
    way, control returns to the Supervisor (this node never sets next/status), and
    the node counts as executed. See specs/005-local-ollama-worker/.
    """
    step = state["step"]

    async def _emit(token: str) -> None:
        # Best-effort per-token streaming: dispatch a LangGraph custom event that
        # astream_run surfaces into the SSE stream. Outside an astream run context
        # (plain ainvoke via POST /intent, or a direct unit-test call) there is no
        # run tree and adispatch raises RuntimeError — swallow it; the full reply
        # is still accumulated and returned.
        try:
            await adispatch_custom_event(LOCAL_TOKEN_EVENT, {"token": token})
        except Exception:
            pass

    if _is_local_compose():
        # Shared client — not closed here (see build_ollama_client); generate_local
        # scopes only the per-request stream/response, not the client.
        text, failure, usage = await generate_local(state, build_ollama_client(), on_token=_emit)
    else:
        # Offload composition to the cloud model (much faster than CPU). Fall back to
        # local Ollama only if it fails BEFORE streaming, so we never regress below
        # the always-available local path.
        text, failure, usage = await generate_cloud(state, on_token=_emit)
        if failure is not None:
            logger.warning(
                "cloud compose failed (%s: %s); falling back to local Ollama",
                failure.category, failure.detail,
            )
            text, failure, usage = await generate_local(state, build_ollama_client(), on_token=_emit)
    if failure is not None:
        return {
            "messages": [
                Message(
                    source="local_llm",
                    content=f"local inference failure: {failure.category}",
                    step=step,
                )
            ],
            "usage": TokenUsage(),
            "visited": ["local_llm"],
            "step": 1,
        }
    return {
        "messages": [Message(source="local_llm", content=text, step=step)],
        "usage": usage,
        "visited": ["local_llm"],
        "step": 1,
    }


def _call_signature(name: str, args: dict) -> str:
    """Stable identity for a tool call, so a repeat is recognisable."""
    try:
        return f"{name}:{json.dumps(args or {}, sort_keys=True, default=str)}"
    except Exception:
        return f"{name}:{args!r}"


def _dispatch_tool(name: str, args: dict, user_id: str) -> tuple[str, list[str] | None]:
    """Run one tool. Returns ``(result_text, kb_source_titles_or_None)``.

    ``user_id`` comes from graph state in every branch — never from a model-supplied
    argument — so no emitted tool call can reach another user's vault, calendar or
    knowledge base. Each ``run_*_tool`` converts its own failures into an
    ``error: ...`` string rather than raising, which is what makes it safe to fan
    these out across a thread pool.
    """
    if name in GRAPHRAG_TOOL_REGISTRY:
        # Per-user (own + global docs); user_id is sent as X-User-Id to the service.
        result, sources = run_graphrag_tool(name, user_id, args)
        return result, sources
    if name in CALENDAR_TOOL_REGISTRY:
        # Per-user: user_id selects whose OAuth tokens are loaded.
        return run_calendar_tool(name, user_id, args), None
    if name in REIT_TOOL_REGISTRY:
        # Read-only and global; user_id threaded only for a uniform signature.
        return run_reit_tool(name, user_id, args), None
    if name in TOOL_REGISTRY:
        return run_vault_tool(name, user_id, args), None
    if name in WEB_TOOL_REGISTRY:
        # Not per-user: the public web is the same for everyone. user_id is threaded
        # only to keep one dispatch signature.
        return run_web_tool(name, user_id, args), None
    return f"error: unknown tool {name!r}", None


def _requested_tool_calls(state: GraphState) -> list[dict]:
    """Normalize the run's tool request(s) to a list of ``{"name", "args"}``.

    Accepts both shapes so the two routing paths can coexist: ``tool_request`` (one
    call, the structured-output router) and ``tool_calls`` (possibly many, the
    native tool-calling router). Unknown names are dropped here rather than passed
    to dispatch, so a hallucinated name can never reach a backend.
    """
    raw = state.get("tool_calls")
    if not raw:
        single = state.get("tool_request")
        raw = [single] if isinstance(single, dict) else []
    calls: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or name not in DISPATCHABLE_TOOLS:
            continue
        args = item.get("args")
        calls.append({"name": name, "args": args if isinstance(args, dict) else {}})
    return calls


def tool_execution(state: GraphState) -> dict:
    """External tool/action node, wired to the user-isolated vault tools.

    If the run carries a ``tool_request`` (``{"name", "args"}``), the named tool
    from :data:`TOOL_REGISTRY` is executed with the run's state-resolved
    ``user_id`` — passed straight from graph state, so no request argument can
    redirect the operation to another user's folder (isolation is enforced in
    tools/user_vault.py). Absent a request the node keeps its deterministic stub
    behavior. Either way it emits a message, fixed usage, and returns control to
    the Supervisor.
    """
    calls = _requested_tool_calls(state)
    user_id = state.get("user_id", SANDBOX_USER_ID)
    _trace("tools.execute", pid=os.getpid(), n=len(calls),
           names=",".join(c["name"] for c in calls) or "-",
           # Args matter for diagnosis: a read that returns too little is usually a
           # too-narrow window or an over-specific search term, invisible without them.
           args=[c["args"] for c in calls][:3])

    if not calls:
        return {
            "messages": [
                Message(source="tool_execution", content="[stub] tool executed", step=state["step"])
            ],
            "usage": TOOL_USAGE,
            "visited": ["tool_execution"],
            "step": 1,
        }

    # Execute. Several calls run concurrently (independent network I/O against
    # different backends), bounded so a 15-game schedule doesn't open 15 sockets to
    # Google Calendar at once. A single call stays on this thread — the overwhelming
    # majority of turns, and no reason to pay for a pool.
    if len(calls) == 1:
        results = [_dispatch_tool(calls[0]["name"], calls[0]["args"], user_id)]
    else:
        with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_TOOL_CALLS, len(calls))) as pool:
            results = list(
                pool.map(lambda c: _dispatch_tool(c["name"], c["args"], user_id), calls)
            )

    # Events are dispatched HERE, on the graph's own thread, not inside the workers.
    # LangChain's callback manager is carried in a contextvar, which pool threads do
    # not inherit — dispatching from a worker silently loses the event and the UI's
    # tool indicator never appears. Emitting after the fact also keeps event order
    # matching call order regardless of which tool finished first.
    messages: list[Message] = []
    kb_sources: list[str] | None = None  # set by the KB tool, for the answer's citation
    for call, (result, sources) in zip(calls, results):
        if sources is not None:
            kb_sources = sources
        messages.append(
            Message(
                source="tool_execution",
                content=f"[tool:{call['name']}] {result}",
                step=state["step"],
            )
        )
        # Best-effort: outside a run context (a direct unit-test call) there is no
        # callback manager and this raises — swallow it; the result is still
        # recorded on the message channel.
        try:
            dispatch_custom_event(
                TOOL_CALL_EVENT,
                {"name": call["name"], "args": call["args"], "result": result},
            )
        except Exception:
            pass

    out: dict = {
        "messages": messages,
        "usage": TOOL_USAGE,
        "visited": ["tool_execution"],
        "step": 1,
        "executed_calls": [_call_signature(c["name"], c["args"]) for c in calls],
    }
    # Record the KB source titles (only when a KB tool ran) so astream_run can cite
    # them at the end of the composed answer. Last-write-wins on the channel.
    if kb_sources is not None:
        out["kb_sources"] = kb_sources
    return out


# --- memory extractor (feature 008) -----------------------------------------

def _seed_messages(payload: IntentPayload) -> list[Message]:
    """Build a run's initial message history from any prior turns the caller passed.

    The gateway is stateless per request, so this is empty by default (today's
    behavior). When a UI reopens a saved conversation (or continues a live one) it
    supplies earlier turns via ``IntentPayload.history``; seeding them gives the
    supervisor routing prompt and the inference prompt real conversational context
    (both already render ``f"{m.source}: {m.content}"`` over ``state["messages"]``).

    Only the most recent ``HISTORY_LIMIT`` turns are kept (oldest dropped, logged)
    to bound tokens/latency. Roles map onto message ``source`` as user→"user",
    assistant→"assistant".

    IMPORTANT: assistant turns must NOT be sourced as "local_llm". The Supervisor's
    completion policy counts WORKER_NODES messages ("local_llm"/"tool_execution")
    as replies already produced *in this run* and finishes as soon as one exists
    (see _build_prompt). Seeding a prior assistant turn as "local_llm" makes it
    think the CURRENT question is already answered, so it routes straight to finish
    and streams nothing. "assistant" keeps the turn as visible context without
    tripping that heuristic. The current message is NOT included here — it stays in
    ``payload.raw_input`` and enters the graph exactly as it does today.
    """
    history: list[HistoryTurn] = list(payload.history or [])
    if len(history) > HISTORY_LIMIT:
        logger.info(
            "seed history truncated: %d turns supplied, keeping last %d",
            len(history),
            HISTORY_LIMIT,
        )
        history = history[-HISTORY_LIMIT:]
    role_source = {"user": "user", "assistant": "assistant"}
    return [
        Message(source=role_source[turn.role], content=turn.content, step=i)
        for i, turn in enumerate(history)
    ]


def _latest_local_reply(messages: list[Message]) -> str:
    """The most recent local_llm reply text (the user-visible assistant answer)."""
    for message in reversed(messages):
        if message.source == "local_llm" and message.content:
            # Skip a degraded failure notice — nothing durable to learn from it.
            if message.content.startswith("local inference failure:"):
                return ""
            return message.content
    return ""


def _build_memory_prompt(user_message: str, assistant_reply: str) -> str:
    """Prompt the silent profile builder to extract at most one durable preference."""
    return (
        "You are a silent, background profile builder for a personal assistant. "
        "Read ONLY the latest exchange and extract AT MOST ONE durable, "
        "cross-session fact about the user worth remembering long-term — e.g. a "
        "favorite thing, a project/tech-stack choice, a tool or workflow setting, "
        "or a stable personal fact. IGNORE transient chat, one-off questions, the "
        "task's subject matter, and generic conversation. If nothing durable is "
        "present, return preference_type=\"none\" with confidence_score 0.0.\n"
        f"User said: {user_message}\n"
        f"Assistant replied: {assistant_reply}\n"
        "Return the fields: preference_type, key_insight (a short stable label), "
        "value (the concrete preference), confidence_score (0.0-1.0)."
    )


def extract_user_preference(
    user_message: str,
    assistant_reply: str,
    client: object,
    model_preference: str | None = DEFAULT_MODEL_PREFERENCE,
) -> MemoryExtraction | None:
    """Ask the selected provider to extract one durable preference. Never raises.

    Dispatches on the same provider registry as the Supervisor and normalizes each
    provider's structured output into a MemoryExtraction. Returns None on any
    failure (call error, unparseable/invalid output) so the node stays best-effort.
    """
    provider, api_model = _resolve_model(model_preference)
    prompt = _build_memory_prompt(user_message, assistant_reply)
    try:
        if provider == "openai":
            response = client.chat.completions.create(
                model=api_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "memory_extraction",
                        "strict": True,
                        "schema": MEMORY_EXTRACTION_JSON_SCHEMA,
                    },
                },
            )
            return MemoryExtraction.model_validate(
                json.loads(response.choices[0].message.content)
            )
        if provider == "anthropic":
            tool = {
                "name": "remember",
                "description": "Record at most one durable user preference.",
                "input_schema": MEMORY_EXTRACTION_JSON_SCHEMA,
            }
            response = client.messages.create(
                model=api_model,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
                tools=[tool],
                tool_choice={"type": "tool", "name": "remember"},
            )
            block = next(b for b in response.content if getattr(b, "type", None) == "tool_use")
            return MemoryExtraction.model_validate(block.input)
        # Gemini: native structured output via the MemoryExtraction schema.
        response = client.models.generate_content(
            model=api_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=MemoryExtraction,
                http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
            ),
        )
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, MemoryExtraction):
            return parsed
        if isinstance(parsed, dict):
            return MemoryExtraction.model_validate(parsed)
        text = getattr(response, "text", None)
        if not text:
            return None
        return MemoryExtraction.model_validate(json.loads(text))
    except Exception:  # call failure / bad shape / invalid output — never raise
        return None


def _record_preference(user_id: str, pref: MemoryExtraction) -> str:
    """Upsert one preference into the user's vault profile via the path-safe utils.

    Reads/writes strictly within ``/tmp/vaults/<user_id>/`` (tools.user_vault
    enforces the isolation boundary). Upserts on ``preference_type + key_insight``:
    an existing bullet for the same key is replaced, otherwise the line is
    appended, so repeated turns refine rather than duplicate. Returns the written
    bullet.
    """
    line = (
        f"- **{pref.preference_type}** | {pref.key_insight}: {pref.value} "
        f"(confidence {pref.confidence_score:.2f})"
    )
    key_prefix = f"- **{pref.preference_type}** | {pref.key_insight}:"
    try:
        existing = read_note(user_id, PREFERENCES_FILE)
    except Exception:  # missing file / any read issue -> start fresh
        existing = ""
    bullets = [b for b in existing.splitlines() if b.startswith("- ")]
    bullets = [b for b in bullets if not b.startswith(key_prefix)]  # upsert
    bullets.append(line)
    content = "# User Preferences\n\n" + "\n".join(bullets) + "\n"
    write_note(user_id, PREFERENCES_FILE, content)
    return line


# Detached background tasks are kept referenced here so the event loop does not
# garbage-collect them mid-flight; each removes itself on completion.
_background_tasks: set[asyncio.Task] = set()


async def extract_and_record_preference(
    user_id: str,
    model_preference: str | None,
    user_message: str,
    assistant_reply: str,
) -> str | None:
    """Extract one durable preference from the exchange and upsert it. Never raises.

    Designed to run fully DETACHED from the request (see
    :func:`schedule_memory_extraction`): the blocking provider call and the file
    write are offloaded to worker threads (``asyncio.to_thread``) so this never
    ties up the event loop that is serving other clients. Returns the recorded
    bullet, or ``None`` when nothing durable/high-confidence was found.
    """
    try:
        if not assistant_reply:
            return None
        client = get_client(model_preference)
        if client is None:
            return None
        pref = await asyncio.to_thread(
            extract_user_preference, user_message, assistant_reply, client, model_preference
        )
        if (
            pref is None
            or pref.preference_type == "none"
            or pref.confidence_score < MEMORY_CONFIDENCE_THRESHOLD
        ):
            return None
        # C-4: hold the per-user vault lock across the local write + S3 mirror so a
        # concurrent request's vault reset can't clobber this profile update.
        async with _vault_locks[user_id]:
            line = await asyncio.to_thread(_record_preference, user_id, pref)
            logger.info("memory: recorded preference for user_id=%s (%s)", user_id, pref.preference_type)
            # Durability: mirror the updated profile back to S3 immediately, in a
            # worker thread (blocking boto3). Best-effort and isolated — a write-back
            # failure keeps the successful local write and never fails the task.
            try:
                await asyncio.to_thread(upload_user_file, user_id, PREFERENCES_FILE)
                logger.info("memory: wrote profile back to S3 for user_id=%s", user_id)
            except Exception:
                logger.warning(
                    "memory: S3 write-back failed for user_id=%s (local copy kept)", user_id
                )
        return line
    except Exception:  # best-effort; a profile-building failure is never fatal
        logger.info("memory: extraction skipped (error)", exc_info=False)
        return None


def schedule_memory_extraction(
    user_id: str, intent: IntentPayload, assistant_reply: str
) -> asyncio.Task | None:
    """Fire-and-forget the memory extraction so it never blocks the response.

    Called right before the (streaming or non-streaming) response returns: it
    schedules :func:`extract_and_record_preference` as a detached task on the
    running loop and returns immediately, so the token stream closes instantly and
    the profile write happens in parallel. Returns the task (callers ignore it;
    tests may await it), or ``None`` when there is no assistant reply to learn from
    or no running loop.
    """
    if not assistant_reply:
        return None
    model_pref = getattr(intent, "model_preference", None) or DEFAULT_MODEL_PREFERENCE
    coro = extract_and_record_preference(user_id, model_pref, intent.raw_input, assistant_reply)
    try:
        task = asyncio.create_task(coro)
    except RuntimeError:  # no running loop (not the normal request path)
        coro.close()
        return None
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


# --- routing + graph --------------------------------------------------------

def route(state: GraphState) -> str:
    """Conditional-edge function: hand back the Supervisor's decision."""
    return state["next"]


def build_graph():
    """Wire and compile the cyclic StateGraph (entry point = supervisor)."""
    builder = StateGraph(GraphState)
    builder.add_node("supervisor", supervisor)
    builder.add_node("local_llm", local_llm)
    builder.add_node("tool_execution", tool_execution)
    builder.set_entry_point("supervisor")
    # finish terminates the graph directly, so the token stream closes instantly.
    # Memory extraction is NOT a graph node — it is fired as a detached background
    # task from run()/astream_run() (schedule_memory_extraction) so it can never
    # block the response or affect the run's status (feature 008).
    builder.add_conditional_edges(
        "supervisor",
        route,
        {"local_llm": "local_llm", "tool_execution": "tool_execution", "finish": END},
    )
    builder.add_edge("local_llm", "supervisor")
    builder.add_edge("tool_execution", "supervisor")
    return builder.compile()


# Compiled once at module load and reused per request.
graph = build_graph()


# --- public API -------------------------------------------------------------

# Per-user vault lock (C-4): serializes the destructive vault sync (reset + download)
# against a concurrent same-user sync and the detached memory write-back, so a
# background profile write is never clobbered by another turn's vault reset.
_vault_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


def _initial_state(payload: IntentPayload, user_id: str) -> GraphState:
    """Seed GraphState for a run — shared by run() and astream_run() so the two
    entrypoints can never diverge (C-3/B1)."""
    return {
        "intent": payload,
        "user_id": user_id,
        "messages": [],
        "prior_context": _seed_messages(payload),
        "documents": "",  # populated by _load_documents in the async prelude
        "document_images": [],  # populated by _load_document_images in the async prelude
        "kb_sources": [],
        "usage": TokenUsage(),
        "visited": [],
        "step": 0,
        "next": "",
        "status": "",
        "tool_request": None,
        "tool_calls": None,
        "pending_plan": None,
        "plan_note": None,
        "truncated": False,
    }


async def _load_documents(user_id: str, document_ids: list[str]) -> str:
    """Fetch + concatenate the extracted text of the message's attached documents,
    capped at DOC_CHAR_BUDGET (the local model's context window is small). Returns
    "" when nothing is attached/readable. Best-effort per doc."""
    if not document_ids:
        return ""
    from services import documents as docstore

    parts: list[str] = []
    remaining = DOC_CHAR_BUDGET
    for doc_id in document_ids[:MAX_DOCS_PER_TURN]:
        try:
            text = await asyncio.to_thread(docstore.fetch_extracted, user_id, doc_id)
        except Exception:
            text = ""
        if not text:
            continue
        chunk = text[:remaining]
        parts.append(chunk)
        remaining -= len(chunk)
        if remaining <= 0:
            parts.append("\n[attached documents truncated to fit the context window]")
            break
    return "\n\n---\n\n".join(parts)


def _downscale_image(data: bytes) -> tuple[bytes, str] | None:
    """Return ``(bytes, media_type)`` for a model-ready image, or None if unusable.

    Downscales so the long edge is at most MAX_IMAGE_EDGE_PX — a full-resolution
    screenshot costs far more tokens without helping the model read it. Blocking
    (Pillow); callers offload with ``asyncio.to_thread``.
    """
    from io import BytesIO

    from PIL import Image

    try:
        with Image.open(BytesIO(data)) as im:
            fmt = (im.format or "").upper()
            if fmt not in ("PNG", "JPEG"):
                return None
            im.load()
            longest = max(im.size)
            if longest > MAX_IMAGE_EDGE_PX:
                scale = MAX_IMAGE_EDGE_PX / longest
                im = im.resize(
                    (max(1, int(im.width * scale)), max(1, int(im.height * scale))),
                    Image.LANCZOS,
                )
            buf = BytesIO()
            if fmt == "PNG":
                im.save(buf, format="PNG", optimize=True)
                media = "image/png"
            else:
                # JPEG cannot carry alpha; convert so a PNG-ish RGBA JPEG can't fail here.
                if im.mode not in ("RGB", "L"):
                    im = im.convert("RGB")
                im.save(buf, format="JPEG", quality=85, optimize=True)
                media = "image/jpeg"
            out = buf.getvalue()
    except Exception:
        return None
    if not out or len(out) > MAX_IMAGE_BYTES:
        return None
    return out, media


async def _load_document_images(user_id: str, document_ids: list[str]) -> list[dict]:
    """Fetch + downscale + base64 the message's attached IMAGES.

    Returns [] when nothing is attached, nothing is an image, or anything at all
    goes wrong — an attachment problem must never break a chat turn, it should just
    degrade to the text-only path. Best-effort per document.
    """
    if not document_ids:
        return []
    import base64

    from services import documents as docstore

    try:
        types = await asyncio.to_thread(
            docstore.fetch_content_types, user_id, document_ids[:MAX_DOCS_PER_TURN]
        )
    except Exception:
        return []

    images: list[dict] = []
    for doc_id in document_ids[:MAX_DOCS_PER_TURN]:
        if len(images) >= MAX_IMAGES_PER_TURN:
            break
        if (types.get(doc_id) or "").split(";")[0].strip().lower() not in IMAGE_MEDIA_TYPES:
            continue
        try:
            raw = await asyncio.to_thread(docstore.fetch_original, user_id, doc_id)
        except Exception:
            continue
        prepared = await asyncio.to_thread(_downscale_image, raw)
        if prepared is None:
            continue
        data, media = prepared
        images.append(
            {
                "media_type": media,
                "data": base64.b64encode(data).decode("ascii"),
                "doc_id": doc_id,
            }
        )
    return images


def _as_content_parts(prompt: str, images: list[dict], provider: str):
    """Shape a prompt (+ optional images) for a provider's content field.

    With NO images this returns the bare ``prompt`` string, so every existing call
    path, test and behaviour is unchanged on the overwhelming majority of turns.
    That is the main guard against regressing the hot path.
    """
    if not images:
        return prompt
    if provider == "anthropic":
        parts: list[dict] = [{"type": "text", "text": prompt}]
        for img in images:
            parts.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img["media_type"],
                        "data": img["data"],
                    },
                }
            )
        return parts
    if provider == "gemini":
        parts = [{"text": prompt}]
        for img in images:
            parts.append(
                {"inline_data": {"mime_type": img["media_type"], "data": img["data"]}}
            )
        return [{"role": "user", "parts": parts}]
    if provider == "langchain":
        # LangChain chat models take OpenAI-style parts and reject the Anthropic
        # shape outright: "ValueError: Unrecognized message part type: image."
        #
        # The native router was passing provider="anthropic" here, so EVERY turn
        # with an attached image raised, fell back to the structured router, and
        # ended up composing an offer instead of proposing a plan. The user then
        # confirmed a plan that had never been stored. A text-only turn worked
        # fine, which is what made this look like a context-loss bug.
        parts = [{"type": "text", "text": prompt}]
        for img in images:
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img['media_type']};base64,{img['data']}"
                    },
                }
            )
        return parts
    return prompt  # unknown provider -> text only, never fail the turn


def _turn_images(state: GraphState) -> list[dict]:
    """Images to attach for THIS model call.

    Only on the first step. The Supervisor runs on every routing decision (up to
    MAX_STEPS), so re-sending a screenshot each time would multiply token cost with
    no benefit — by step 1 the conversation already carries what the model read.
    """
    if state.get("step", 0) != 0:
        return []
    return state.get("document_images") or []


async def _prepare_vault(user_id: str) -> None:
    """Localize the caller's Markdown vault before the graph runs. Best-effort — a
    sync failure must not break the run. Held under the per-user lock so it can't
    race a concurrent same-user sync or the detached memory write-back (C-4)."""
    try:
        async with _vault_locks[user_id]:
            await sync_user_vault(user_id)
    except Exception:
        logger.debug("vault sync skipped for user_id=%s (error)", user_id)


async def run(
    payload: IntentPayload, user_id: str = SANDBOX_USER_ID
) -> OrchestrationOutcome:
    """Orchestrate an accepted intent to a terminating OrchestrationOutcome.

    ``user_id`` is the caller identity resolved at the gateway boundary
    (auth.resolve_user_id); it is seeded into GraphState and defaults to the
    sandbox user so direct/programmatic callers and existing tests need not pass
    it.

    Async because the local_llm worker performs an asynchronous Ollama call
    (feature 005); the graph is driven with ``ainvoke`` (the sync supervisor and
    tool nodes run unchanged under it). Always returns: node errors or a
    recursion-limit breach are captured into ``status="error"`` rather than
    propagating.
    """
    initial = _initial_state(payload, user_id)
    await _prepare_vault(user_id)  # C-3: parity with astream_run (was missing here)
    initial["documents"] = await _load_documents(user_id, payload.document_ids)
    initial["document_images"] = await _load_document_images(user_id, payload.document_ids)
    try:
        final = await graph.ainvoke(initial, config={"recursion_limit": RECURSION_LIMIT})
    except (GraphRecursionError, Exception) as exc:  # noqa: B014 - defensive catch-all
        return OrchestrationOutcome(
            status="error",
            nodes_executed=[],
            messages=[Message(source="orchestrator", content=f"error: {exc}", step=0)],
            usage=TokenUsage(),
            steps=0,
        )

    # Detached, non-blocking profile update from the finished exchange.
    schedule_memory_extraction(
        user_id, payload, _latest_local_reply(final.get("messages", []))
    )

    visited = final.get("visited", [])
    return OrchestrationOutcome(
        status=final.get("status") or "completed",
        nodes_executed=[n for n in visited if n in WORKER_NODES],
        messages=final.get("messages", []),
        usage=final.get("usage", TokenUsage()),
        steps=final.get("step", 0),
    )


async def astream_run(
    payload: IntentPayload, user_id: str = SANDBOX_USER_ID
) -> AsyncIterator[dict]:
    """Drive the graph via ``astream_events`` and yield progressive event dicts.

    Emits ``{"token": <text>}`` chunks progressively, then a terminal
    ``{"status": <final status>}``. local_llm streams true per-token: it calls
    Ollama with ``stream=True`` and dispatches a LOCAL_TOKEN_EVENT custom event
    per chunk, surfaced here as ``on_custom_event``. tool_execution is a
    non-streaming stub whose reply is emitted whole on ``on_chain_end``.

    Fallback (important): if a local_llm run streams NO tokens — a degraded
    Ollama call that only recorded a failure notice, or any non-streamed reply —
    its final ``on_chain_end`` message is emitted instead, so a run that a guard
    or finish stamps ``completed`` never reaches the client with an empty body
    ("No reply produced"). Never raises: a run error is surfaced as an error
    status.

    The caller (router) is responsible for SSE framing; this yields plain dicts.
    """
    initial = _initial_state(payload, user_id)
    final_status = "completed"
    streamed_local_tokens = False
    last_local_reply = ""  # captured for the detached memory extraction
    kb_sources: list[str] = []  # titles from the last KB retrieval, cited after the answer

    # Pre-execution: localize the caller's Markdown vault before the supervisor
    # fires, so downstream nodes read from /tmp/vaults/<user_id>/ rather than
    # reaching across the network mid-run. Best-effort + per-user-locked (see
    # _prepare_vault). Sandbox resolves to a local mock folder (offline).
    await _prepare_vault(user_id)
    initial["documents"] = await _load_documents(user_id, payload.document_ids)
    initial["document_images"] = await _load_document_images(user_id, payload.document_ids)

    try:
        async for event in graph.astream_events(
            initial, version="v2", config={"recursion_limit": RECURSION_LIMIT}
        ):
            etype = event["event"]
            name = event.get("name")
            # A fresh local_llm invocation resets the per-run "did it stream?" flag.
            if etype == "on_chain_start" and name == "local_llm":
                streamed_local_tokens = False
            # Per-token stream from local_llm (see LOCAL_TOKEN_EVENT / _emit).
            if etype == "on_custom_event" and name == LOCAL_TOKEN_EVENT:
                streamed_local_tokens = True
                yield {"token": event["data"]["token"]}
                continue
            # A vault tool ran: surface it as a structured event (not an assistant
            # token) so the UI can show a reading/searching-the-vault indicator.
            if etype == "on_custom_event" and name == TOOL_CALL_EVENT:
                yield {"tool_call": event["data"]}
                continue
            if etype != "on_chain_end":
                continue
            output = event["data"].get("output")
            if not isinstance(output, dict):
                continue
            # Capture the assistant reply for the detached memory extraction, and
            # (if this run streamed NO tokens — a degraded failure notice or any
            # non-streamed reply) surface its final message so the content is not
            # silently dropped behind a "completed" status.
            if name == "local_llm":
                for message in output.get("messages", []):
                    if not message.content:
                        continue
                    if not message.content.startswith("local inference failure:"):
                        last_local_reply = message.content
                    if not streamed_local_tokens:
                        yield {"token": message.content}
            # Capture KB source titles from a tool_execution turn (only KB tools set
            # this) so we can cite them once the answer is composed.
            if "kb_sources" in output:
                kb_sources = output.get("kb_sources") or []
            # The supervisor stamps the terminal status on finish/degrade/halt.
            if output.get("status"):
                final_status = output["status"]
    except (GraphRecursionError, Exception) as exc:  # noqa: B014 - never break the stream
        yield {"status": "error", "detail": f"{type(exc).__name__}: {exc}"[:200]}
        return
    # Cite the KB source document(s) at the end of a completed, KB-grounded answer —
    # deterministic (doesn't rely on the small local model to remember to cite).
    if kb_sources and last_local_reply and final_status in ("completed", "halted_step_bound"):
        src_line = "\n\nSource: " + ", ".join(kb_sources)
        yield {"token": src_line}
        last_local_reply += src_line
    # Detached, non-blocking profile update — the stream closes immediately after
    # the status event while extraction runs in parallel.
    schedule_memory_extraction(user_id, payload, last_local_reply)
    yield {"status": final_status}
