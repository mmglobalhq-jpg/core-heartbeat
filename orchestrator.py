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
from typing import Annotated

from typing_extensions import TypedDict

from google import genai
from google.genai import errors, types
import httpx

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph

from models import (
    IntentPayload,
    Message,
    OrchestrationOutcome,
    RoutingDecision,
    RoutingFailure,
    TokenUsage,
)

# --- constants --------------------------------------------------------------

MAX_STEPS = 8          # graceful step bound (Supervisor finishes at/after this)
RECURSION_LIMIT = 25   # hard LangGraph catch
MODEL_NAME = "gemini-2.5-flash"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
REQUEST_TIMEOUT_MS = 10_000  # bound each model call (FR-006); milliseconds

# Fixed, deterministic per-node usage increments for the stub worker nodes.
LLM_USAGE = TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30)
TOOL_USAGE = TokenUsage(input_tokens=5, output_tokens=0, total_tokens=5)

WORKER_NODES = ["local_llm", "tool_execution"]


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
    messages: Annotated[list[Message], operator.add]
    usage: Annotated[TokenUsage, add_usage]
    visited: Annotated[list[str], operator.add]
    step: Annotated[int, operator.add]
    next: str
    status: str


# --- model client (feature 004) ---------------------------------------------

_client_cache: tuple[str, genai.Client] | None = None


def get_client() -> genai.Client | None:
    """Construct (and memoize) a GenAI client from GEMINI_API_KEY.

    Returns None if the key is unset/blank so a missing credential becomes a
    categorized failure and the service stays bootable without a key (FR-003).
    The client is cached and reused across Supervisor visits; the cache is keyed
    on the key value, so a changed key rebuilds. The key is passed to the SDK
    and never logged.
    """
    global _client_cache
    key = os.environ.get(GEMINI_API_KEY_ENV)
    if not key or not key.strip():
        return None
    if _client_cache is None or _client_cache[0] != key:
        _client_cache = (key, genai.Client(api_key=key))
    return _client_cache[1]


def _build_prompt(state: GraphState) -> str:
    """Deterministic routing prompt derived from the intent + message history."""
    intent = state["intent"]
    history = "\n".join(f"{m.source}: {m.content}" for m in state.get("messages", []))
    return (
        "You are the Supervisor in an orchestration graph. Decide the single "
        "next step for this run.\n"
        f"Intent: {intent.intent}\n"
        f"Confidence: {intent.confidence}\n"
        f"Raw input: {intent.raw_input}\n"
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


def request_routing_decision(
    state: GraphState, client: genai.Client
) -> tuple[RoutingDecision | None, RoutingFailure | None, TokenUsage]:
    """Ask the model for a routing decision. Never raises.

    Returns exactly one of (decision, failure) non-None, plus a TokenUsage
    (zeros when the model reports none or on a pre-response failure).
    """
    prompt = _build_prompt(state)
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
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

    client = get_client()
    if client is None:
        return _degraded(step, RoutingFailure(category="missing_credential", detail="GEMINI_API_KEY not set"))

    decision, failure, usage = request_routing_decision(state, client)
    if failure is not None:
        return _degraded(step, failure, usage)

    nxt = decision.next_node
    update: dict = {
        "next": nxt,
        "step": 1,
        "usage": usage,
        "messages": [Message(source="supervisor", content=f"route -> {nxt}", step=step)],
    }
    if nxt == "finish":
        update["status"] = "completed"
    return update


def local_llm(state: GraphState) -> dict:
    """Stubbed local model inference: deterministic output + fixed usage."""
    return {
        "messages": [Message(source="local_llm", content="[stub] local inference result", step=state["step"])],
        "usage": LLM_USAGE,
        "visited": ["local_llm"],
        "step": 1,
    }


def tool_execution(state: GraphState) -> dict:
    """Stubbed external tool/action: deterministic output + fixed usage."""
    return {
        "messages": [Message(source="tool_execution", content="[stub] tool executed", step=state["step"])],
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

def run(payload: IntentPayload) -> OrchestrationOutcome:
    """Orchestrate an accepted intent to a terminating OrchestrationOutcome.

    Always returns: node errors or a recursion-limit breach are captured into
    ``status="error"`` rather than propagating.
    """
    initial: GraphState = {
        "intent": payload,
        "messages": [],
        "usage": TokenUsage(),
        "visited": [],
        "step": 0,
        "next": "",
        "status": "",
    }
    try:
        final = graph.invoke(initial, config={"recursion_limit": RECURSION_LIMIT})
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
