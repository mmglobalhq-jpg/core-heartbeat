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

import json
import operator
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Annotated

from typing_extensions import TypedDict

from google import genai
from google.genai import errors, types
import httpx

from langchain_core.callbacks import adispatch_custom_event
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph

from auth import SANDBOX_USER_ID
from services.storage_sync import sync_user_vault
from tools.user_vault import USER_VAULT_TOOLS, run_vault_tool
from models import (
    IntentPayload,
    Message,
    OrchestrationOutcome,
    RoutingDecision,
    RoutingFailure,
    TokenUsage,
    WorkerFailure,
)

# --- constants --------------------------------------------------------------

MAX_STEPS = 8          # graceful step bound (Supervisor finishes at/after this)
RECURSION_LIMIT = 25   # hard LangGraph catch
MODEL_NAME = "gemini-2.5-flash"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
REQUEST_TIMEOUT_MS = 10_000  # bound each model call (FR-006); milliseconds

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
ROUTING_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "next_node": {"type": "string", "enum": ["local_llm", "tool_execution", "finish"]}
    },
    "required": ["next_node"],
    "additionalProperties": False,
}

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
    usage: Annotated[TokenUsage, add_usage]
    visited: Annotated[list[str], operator.add]
    step: Annotated[int, operator.add]
    next: str
    status: str


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
    history = "\n".join(f"{m.source}: {m.content}" for m in messages)
    worker_replies = sum(1 for m in messages if m.source in WORKER_NODES)
    return (
        "You are the Supervisor in an orchestration graph. Decide the single "
        "next step for this run.\n"
        "Routing policy:\n"
        "- Route to local_llm to generate a conversational or reasoned reply, or "
        "to tool_execution to run an external action/tool.\n"
        "- Choose finish as soon as the intent has been satisfied. If a worker "
        "has already produced a reply that answers the intent (e.g. a general "
        "chat message), you MUST choose finish. Never re-dispatch a worker that "
        "has already answered.\n"
        f"Intent: {intent.intent}\n"
        f"Confidence: {intent.confidence}\n"
        f"Raw input: {intent.raw_input}\n"
        f"Worker replies so far: {worker_replies}\n"
        f"History so far:\n{history or '(none)'}\n"
        "Respond with next_node = one of: local_llm, tool_execution, finish."
    )


def _extract_usage(response) -> TokenUsage:
    """Map the response's usage_metadata into TokenUsage (zeros if absent)."""
    um = getattr(response, "usage_metadata", None)
    if um is None:
        return TokenUsage()
    return TokenUsage(
        input_tokens=getattr(um, "prompt_token_count", 0) or 0,
        output_tokens=getattr(um, "candidates_token_count", 0) or 0,
        total_tokens=getattr(um, "total_token_count", 0) or 0,
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
    """Map OpenAI's response.usage into TokenUsage (zeros if absent)."""
    u = getattr(response, "usage", None)
    if u is None:
        return TokenUsage()
    return TokenUsage(
        input_tokens=getattr(u, "prompt_tokens", 0) or 0,
        output_tokens=getattr(u, "completion_tokens", 0) or 0,
        total_tokens=getattr(u, "total_tokens", 0) or 0,
    )


def _anthropic_usage(response) -> TokenUsage:
    """Map Anthropic's response.usage into TokenUsage (total = input + output)."""
    u = getattr(response, "usage", None)
    if u is None:
        return TokenUsage()
    inp = getattr(u, "input_tokens", 0) or 0
    out = getattr(u, "output_tokens", 0) or 0
    return TokenUsage(input_tokens=inp, output_tokens=out, total_tokens=inp + out)


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


def build_ollama_client() -> httpx.AsyncClient:
    """Construct an AsyncClient bound to the configured timeout (feature 005).

    This is the test seam: the local_llm node calls it at invoke time, so tests
    monkeypatch ``orchestrator.build_ollama_client`` to return a client wired with
    ``httpx.MockTransport`` — no network, no daemon (FR-011).
    """
    return httpx.AsyncClient(timeout=_ollama_timeout_s())


def _build_local_prompt(state: GraphState) -> str:
    """Deterministic inference prompt from the intent + message history."""
    intent = state["intent"]
    history = "\n".join(f"{m.source}: {m.content}" for m in state.get("messages", []))
    return (
        f"Intent: {intent.intent}\n"
        f"Raw input: {intent.raw_input}\n"
        f"History so far:\n{history or '(none)'}\n"
        "Respond helpfully to the intent above."
    )


def _extract_ollama_usage(body: dict) -> TokenUsage:
    """Map Ollama's prompt_eval_count/eval_count into TokenUsage (zeros if absent)."""
    inp = body.get("prompt_eval_count") or 0
    out = body.get("eval_count") or 0
    return TokenUsage(input_tokens=inp, output_tokens=out, total_tokens=inp + out)


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


# --- nodes ------------------------------------------------------------------

def _degraded(step: int, failure: RoutingFailure, usage: TokenUsage | None = None) -> dict:
    """Safe terminal update after a routing failure (FR-005, FR-008)."""
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

    decision, failure, usage = request_routing_decision(state, client, model_pref)
    if failure is not None:
        return _degraded(step, failure, usage)

    nxt = decision.next_node

    # Deterministic anti-reloop guard (do not rely on the model to terminate).
    # If the model tries to re-dispatch a worker that has ALREADY produced a
    # reply this run, override the decision to `finish`. This kills the observed
    # gemini-2.5-flash trap where a general chat is routed back to local_llm over
    # and over until the MAX_STEPS guard fires (see _build_prompt's completion
    # policy — this enforces it in code). Each worker still runs at most once per
    # run; a genuine second worker (e.g. tool_execution after local_llm) is
    # unaffected because it is not yet in `visited`.
    visited = state.get("visited", [])
    if nxt in WORKER_NODES and nxt in visited:
        return {
            "next": "finish",
            "status": "completed",
            "step": 1,
            "usage": usage,
            "messages": [
                Message(
                    source="supervisor",
                    content=f"route -> finish (guard: {nxt} already replied)",
                    step=step,
                )
            ],
        }

    update: dict = {
        "next": nxt,
        "step": 1,
        "usage": usage,
        "messages": [Message(source="supervisor", content=f"route -> {nxt}", step=step)],
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

    async with client:
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
        user_id = state.get("user_id", SANDBOX_USER_ID)
        result = run_vault_tool(name, user_id, request.get("args") or {})
        content = f"[tool:{name}] {result}"
    else:
        content = "[stub] tool executed"
    return {
        "messages": [Message(source="tool_execution", content=content, step=state["step"])],
        "usage": TOOL_USAGE,
        "visited": ["tool_execution"],
        "step": 1,
    }


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
    initial: GraphState = {
        "intent": payload,
        "user_id": user_id,
        "messages": [],
        "usage": TokenUsage(),
        "visited": [],
        "step": 0,
        "next": "",
        "status": "",
    }
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
    initial: GraphState = {
        "intent": payload,
        "user_id": user_id,
        "messages": [],
        "usage": TokenUsage(),
        "visited": [],
        "step": 0,
        "next": "",
        "status": "",
    }
    final_status = "completed"
    streamed_local_tokens = False

    # Pre-execution: localize the caller's Markdown vault before the supervisor
    # fires, so downstream nodes read from /tmp/vaults/<user_id>/ rather than
    # reaching across the network mid-run (Phase 2 of the multi-user bridge). The
    # sandbox user resolves to a local mock folder, so this is credential-free
    # offline. Best-effort: a vault-sync failure must not break the stream — the
    # run proceeds with whatever context is already local.
    try:
        await sync_user_vault(user_id)
    except Exception:
        pass

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
            if etype != "on_chain_end":
                continue
            output = event["data"].get("output")
            if not isinstance(output, dict):
                continue
            # tool_execution is a non-streaming stub — emit its reply as one chunk.
            if name == "tool_execution":
                for message in output.get("messages", []):
                    yield {"token": message.content}
            # local_llm normally streams live tokens; if this run streamed NONE
            # (a degraded call that only recorded a failure notice, or any
            # non-streamed reply), surface its final message so the content is not
            # silently dropped behind a "completed" status.
            if name == "local_llm" and not streamed_local_tokens:
                for message in output.get("messages", []):
                    if message.content:
                        yield {"token": message.content}
            # The supervisor stamps the terminal status on finish/degrade/halt.
            if output.get("status"):
                final_status = output["status"]
    except (GraphRecursionError, Exception) as exc:  # noqa: B014 - never break the stream
        yield {"status": "error", "detail": f"{type(exc).__name__}: {exc}"[:200]}
        return
    yield {"status": final_status}
