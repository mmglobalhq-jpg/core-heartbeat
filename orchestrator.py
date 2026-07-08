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
from collections import defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Annotated

from typing_extensions import TypedDict

from google import genai
from google.genai import errors, types
import httpx

from langchain_core.callbacks import adispatch_custom_event
from langchain_core.callbacks.manager import dispatch_custom_event
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph

from auth import SANDBOX_USER_ID
from services.storage_sync import sync_user_vault, upload_user_file
from tools.user_vault import USER_VAULT_TOOLS, read_note, run_vault_tool, write_note
from models import (
    HistoryTurn,
    IntentPayload,
    MemoryExtraction,
    Message,
    OrchestrationOutcome,
    RoutingDecision,
    RoutingFailure,
    TokenUsage,
    WorkerFailure,
)

logger = logging.getLogger(__name__)

# --- constants --------------------------------------------------------------

MAX_STEPS = 8          # graceful step bound (Supervisor finishes at/after this)
RECURSION_LIMIT = 25   # hard LangGraph catch
HISTORY_LIMIT = 10     # max prior turns seeded from IntentPayload.history (token/latency bound)
DOC_CHAR_BUDGET = 12000  # max chars of attached-document text injected (local model window is small)
MAX_DOCS_PER_TURN = 10   # cap attached docs per message
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
ROUTING_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "next_node": {"type": "string", "enum": ["local_llm", "tool_execution", "finish"]},
        "tool_name": {
            "type": ["string", "null"],
            "enum": ["read_user_note", "search_user_vault", "write_user_note", None],
        },
        "tool_args": {
            "type": "object",
            "properties": {
                "filename": {"type": ["string", "null"]},
                "query": {"type": ["string", "null"]},
                "content": {"type": ["string", "null"]},
            },
            "additionalProperties": False,
        },
    },
    "required": ["next_node"],
    "additionalProperties": False,
}

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
    history = _render_history(messages)
    worker_replies = sum(1 for m in messages if m.source in WORKER_NODES)
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
        "- Route to tool_execution to act on the user's personal Markdown vault. "
        "You then MUST set tool_name and tool_args. The available tools are:\n"
        "    * read_user_note — read one note. tool_args: {\"filename\": <path>}.\n"
        "    * search_user_vault — case-insensitive text/regex search across the "
        "vault's notes. tool_args: {\"query\": <text>}.\n"
        "    * write_user_note — create/update a note. tool_args: "
        "{\"filename\": <path>, \"content\": <text>}.\n"
        "  The tools are already scoped to THIS user's vault — never put a user id "
        "or absolute/`..` path in a filename.\n"
        "- After a tool result appears in the history, either issue another tool "
        "call or route to local_llm to compose the final answer from it.\n"
        "- Choose finish as soon as the intent has been satisfied. If a worker "
        "has already produced a reply that answers the intent (e.g. a general "
        "chat message), you MUST choose finish. Never re-dispatch local_llm once "
        "it has answered.\n"
        f"Intent: {intent.intent}\n"
        f"Confidence: {intent.confidence}\n"
        f"Raw input: {intent.raw_input}\n"
        f"Worker replies so far: {worker_replies}\n"
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
    try:
        response = client.messages.create(
            model=api_model,
            max_tokens=64,
            messages=[{"role": "user", "content": prompt}],
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
        f"Intent: {intent.intent}\n"
        f"Raw input: {intent.raw_input}\n"
        f"Conversation so far:\n{history or '(none)'}\n"
        "Respond helpfully to the intent above."
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
    payload = {
        "model": _ollama_model(),
        "prompt": _build_local_prompt(state),
        "stream": True,
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


def supervisor(state: GraphState) -> dict:
    """Model-driven routing hub. Falls back to a safe finish on any failure."""
    step = state["step"]

    # Layer-2 termination + cost guard: never call the model past the bound.
    if step >= MAX_STEPS:
        return {
            "next": "finish",
            "status": "halted_step_bound",
            "step": 1,
            "messages": [Message(source="supervisor", content="route -> finish (step bound)", step=step)],
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

    nxt = decision.next_node

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
    if nxt == "tool_execution" and decision.tool_name:
        tool_request = {
            "name": decision.tool_name,
            "args": decision.tool_args.model_dump(exclude_none=True),
        }

    label = f"route -> {nxt}"
    if tool_request is not None:
        label = f"route -> {nxt} ({tool_request['name']})"

    update: dict = {
        "next": nxt,
        "step": 1,
        "usage": usage,
        "tool_request": tool_request,
        "messages": [Message(source="supervisor", content=label, step=step)],
    }
    if nxt == "finish":
        update["status"] = "completed"
    return update


async def local_llm(state: GraphState) -> dict:
    """Live local inference via Ollama (feature 005). Degrades safely on any failure.

    On success records the model's generated text + its reported token usage. On
    any failure records a categorized WorkerFailure message with zero usage. Either
    way, control returns to the Supervisor (this node never sets next/status), and
    the node counts as executed. See specs/005-local-ollama-worker/.
    """
    step = state["step"]
    client = build_ollama_client()

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

    # Shared client — not closed here (see build_ollama_client); generate_local
    # scopes only the per-request stream/response, not the client.
    text, failure, usage = await generate_local(state, client, on_token=_emit)
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
    request = state.get("tool_request")
    if isinstance(request, dict) and request.get("name") in TOOL_REGISTRY:
        name = request["name"]
        args = request.get("args") or {}
        user_id = state.get("user_id", SANDBOX_USER_ID)
        result = run_vault_tool(name, user_id, args)
        content = f"[tool:{name}] {result}"
        # Surface the call for the SSE stream so the UI can show a
        # reading/searching indicator. Best-effort: outside a run context (a
        # direct unit-test call) there is no callback manager and this raises —
        # swallow it; the result is still recorded on the message channel.
        try:
            dispatch_custom_event(
                TOOL_CALL_EVENT, {"name": name, "args": args, "result": result}
            )
        except Exception:
            pass
    else:
        content = "[stub] tool executed"
    return {
        "messages": [Message(source="tool_execution", content=content, step=state["step"])],
        "usage": TOOL_USAGE,
        "visited": ["tool_execution"],
        "step": 1,
    }


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
        "usage": TokenUsage(),
        "visited": [],
        "step": 0,
        "next": "",
        "status": "",
        "tool_request": None,
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

    # Pre-execution: localize the caller's Markdown vault before the supervisor
    # fires, so downstream nodes read from /tmp/vaults/<user_id>/ rather than
    # reaching across the network mid-run. Best-effort + per-user-locked (see
    # _prepare_vault). Sandbox resolves to a local mock folder (offline).
    await _prepare_vault(user_id)
    initial["documents"] = await _load_documents(user_id, payload.document_ids)

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
            # The supervisor stamps the terminal status on finish/degrade/halt.
            if output.get("status"):
                final_status = output["status"]
    except (GraphRecursionError, Exception) as exc:  # noqa: B014 - never break the stream
        yield {"status": "error", "detail": f"{type(exc).__name__}: {exc}"[:200]}
        return
    # Detached, non-blocking profile update — the stream closes immediately after
    # the status event while extraction runs in parallel.
    schedule_memory_extraction(user_id, payload, last_local_reply)
    yield {"status": final_status}
