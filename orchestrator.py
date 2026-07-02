"""LangGraph orchestration engine for core-heartbeat (feature 003).

Takes an accepted IntentPayload and drives it through a small cyclic graph:

    supervisor --(conditional)--> local_llm | tool_execution | END
    local_llm      --> supervisor
    tool_execution --> supervisor

The Supervisor is the entry point and routing hub. Nodes are deterministic
stubs (no real inference or tools). Termination is guaranteed three ways: the
Supervisor's finish decision, an explicit MAX_STEPS bound, and LangGraph's
recursion_limit as a hard catch. See specs/003-langgraph-orchestration/.
"""

import operator
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph

from models import IntentPayload, Message, OrchestrationOutcome, TokenUsage

# --- constants --------------------------------------------------------------

MAX_STEPS = 8          # graceful step bound (Supervisor finishes at/after this)
RECURSION_LIMIT = 25   # hard LangGraph catch; comfortably above the plan's needs
NOOP_INTENTS = {"ping", "noop"}  # intents the Supervisor finishes immediately

# Fixed, deterministic per-node usage increments (a full 2-hop run -> 35 total).
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

    intent: IntentPayload                          # last-value-wins; set at input
    messages: Annotated[list[Message], operator.add]   # append-only history
    usage: Annotated[TokenUsage, add_usage]            # additive accumulator
    visited: Annotated[list[str], operator.add]        # worker-node log -> routing
    step: Annotated[int, operator.add]                 # step counter -> MAX_STEPS
    next: str                                      # Supervisor's routing decision
    status: str                                    # terminal status at finish


# --- nodes ------------------------------------------------------------------

def supervisor(state: GraphState) -> dict:
    """Entry point / routing hub: inspect intent + state, decide next or finish."""
    step = state["step"]
    visited = state.get("visited", [])
    intent_id = state["intent"].intent

    if intent_id in NOOP_INTENTS:
        decision, status = "finish", "completed"
    elif step >= MAX_STEPS:
        decision, status = "finish", "halted_step_bound"
    else:
        pending = [n for n in WORKER_NODES if n not in visited]
        if pending:
            decision, status = pending[0], ""
        else:
            decision, status = "finish", "completed"

    update: dict = {
        "next": decision,
        "step": 1,
        "messages": [Message(source="supervisor", content=f"route -> {decision}", step=step)],
    }
    if status:
        update["status"] = status
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
    ``status="error"`` rather than propagating (edge case), so the gateway never
    hangs or crashes.
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
