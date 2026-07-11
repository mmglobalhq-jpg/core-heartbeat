"""Knowledge-base (Graph-RAG) tool for the orchestrator.

Calls the internal Graph-RAG service's retrieve-only query endpoint, scoped to the
caller's ``user_id`` (returns that user's own docs + the global tier). It returns
ranked chunks + ``[doc:<id>]`` citations as compact CONTEXT — the service does not
generate an answer (retrieve_only); the orchestrator's local_llm composes the reply
from this context alongside general knowledge and chat history.

Mirrors the httpx pattern in tools/fund_holdings.py (test-injectable transport, retry
on transient 5xx / transport errors). ``user_id`` is threaded from graph state (like
the vault tools) and sent as the ``X-User-Id`` header — never a model-supplied arg, so
a query can't be redirected to another user's knowledge base.
"""
from __future__ import annotations

import os
import time

import httpx

GRAPHRAG_URL_ENV = "GRAPHRAG_SERVICE_URL"
GRAPHRAG_KEY_ENV = "GRAPHRAG_API_KEY"

REQUEST_TIMEOUT_S = 30.0
MAX_ATTEMPTS = 3
RETRY_BACKOFF_S = 0.4
DEFAULT_TOP_K = 5
MAX_CHUNKS = 8      # cap chunks fed into the (small) local model's context
MAX_CHARS = 900     # per-chunk char cap for the context block

# Test seam: unit tests set this to an ``httpx.MockTransport`` to exercise the tool
# without a live service. None -> real network.
_transport: httpx.BaseTransport | None = None


class KnowledgeBaseError(Exception):
    """Raised when the KB service is unreachable or misconfigured."""


def kb_configured() -> bool:
    """Is the KB service configured (URL present)? Gates deterministic KB routing so
    it never fires in tests / KB-less deployments. Pure (reads env)."""
    return bool(os.environ.get(GRAPHRAG_URL_ENV))


def _base_url() -> str:
    url = os.environ.get(GRAPHRAG_URL_ENV)
    if not url:
        raise KnowledgeBaseError(f"{GRAPHRAG_URL_ENV} is not set")
    return url.rstrip("/")


def _headers(user_id: str) -> dict[str, str]:
    key = os.environ.get(GRAPHRAG_KEY_ENV)
    if not key:
        raise KnowledgeBaseError(f"{GRAPHRAG_KEY_ENV} is not set")
    return {
        "Authorization": f"Bearer {key}",
        "X-User-Id": user_id,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _client(user_id: str) -> httpx.Client:
    return httpx.Client(
        base_url=_base_url(), headers=_headers(user_id),
        timeout=REQUEST_TIMEOUT_S, transport=_transport,
    )


def _post(path: str, user_id: str, json_body: dict) -> httpx.Response:
    """POST to the KB service, retrying transient 5xx / transport failures."""
    last_exc: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            with _client(user_id) as c:
                r = c.post(path, json=json_body)
        except httpx.TransportError as exc:
            last_exc = exc
            if attempt + 1 < MAX_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_S * (attempt + 1))
                continue
            raise
        if r.status_code >= 500 and attempt + 1 < MAX_ATTEMPTS:
            last_exc = httpx.HTTPStatusError(
                f"server error {r.status_code}", request=r.request, response=r
            )
            time.sleep(RETRY_BACKOFF_S * (attempt + 1))
            continue
        r.raise_for_status()
        return r
    raise last_exc  # pragma: no cover - loop always returns or raises above


def format_context(payload: dict) -> str:
    """Turn the retrieve-only response into a compact, title-cited context block. Pure."""
    chunks = payload.get("chunks") or []
    if not chunks:
        return "No relevant information found in the knowledge base."
    lines: list[str] = []
    for c in chunks[:MAX_CHUNKS]:
        title = (c.get("title") or "Untitled document").strip()
        content = " ".join((c.get("content") or "").split())
        if len(content) > MAX_CHARS:
            content = content[:MAX_CHARS] + "…"
        lines.append(f"[{title}] {content}")
    return "\n\n".join(lines)


def source_titles(payload: dict) -> list[str]:
    """Distinct source-document titles that fed the context (for the answer's citation)."""
    seen: set[str] = set()
    out: list[str] = []
    for s in payload.get("sources") or []:
        title = (s.get("title") or "").strip()
        if title and title not in seen:
            seen.add(title)
            out.append(title)
    return out


def query_knowledge_base(user_id: str, query: str, top_k: int = DEFAULT_TOP_K) -> tuple[str, list[str]]:
    """Retrieve the user's (own + global) KB context. Returns (context, source_titles)."""
    if not query or not query.strip():
        return "error: empty query", []
    r = _post("/api/query", user_id, {
        "query": query.strip(),
        "options": {"retrieve_only": True, "top_k": top_k},
    })
    payload = r.json()
    return format_context(payload), source_titles(payload)


# --- dispatch (name -> callable(user_id, args) -> (context, source_titles)) ---

_DISPATCH = {
    "query_knowledge_base": lambda uid, a: query_knowledge_base(uid, a.get("query") or ""),
}

# The set of tool names the tool_execution node recognizes as KB tools.
GRAPHRAG_TOOL_REGISTRY = frozenset(_DISPATCH)


def run_graphrag_tool(name: str, user_id: str, args: dict | None = None) -> tuple[str, list[str]]:
    """Execute a registered KB tool by name for a user; never raises.

    Returns ``(context, source_titles)`` — the context string for the model and the
    distinct source-doc titles so the orchestrator can cite them. An unknown tool, a
    missing arg, or any backend failure yields ``("error: ...", [])`` so the graph
    keeps running.
    """
    fn = _DISPATCH.get(name)
    if fn is None:
        return f"error: unknown tool {name!r}", []
    try:
        return fn(user_id, args or {})
    except httpx.HTTPStatusError as exc:
        return f"error: KB service returned {exc.response.status_code}", []
    except KnowledgeBaseError as exc:
        return f"error: {exc}", []
    except Exception as exc:  # never crash the graph on a bad tool call
        return f"error: {type(exc).__name__}: {exc}", []
