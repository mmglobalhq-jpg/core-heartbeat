"""Knowledge-base (Graph-RAG) tools for the orchestrator.

Three tools over the internal Graph-RAG service, all scoped to the caller's
``user_id`` (that user's own docs + the global tier):

``query_knowledge_base``
    Corpus-wide retrieve-only search. Returns ranked chunks as compact CONTEXT — the
    service does not generate an answer (retrieve_only); the orchestrator's local_llm
    composes the reply from this context alongside general knowledge and chat history.
    Its context budget is deliberately SMALL (see DEFAULT_TOP_K/MAX_CHARS below):
    measured, ~34s of a 37s time-to-first-token was CPU prefill of injected chunks.

``summarize_document``
    ONE named document, via the service's /api/summarize. This exists because
    ``query_knowledge_base`` structurally cannot answer "summarize document X":
    it is a top-k nearest-neighbour search over the whole corpus, so it returns a
    handful of fragments from whichever documents are nearest to a title-shaped
    query — which, for a weekly publication whose issues share a title, is the wrong
    issue. Measured before the fix: 3 of 4 chunks came from the wrong week, and the
    4 x 600-char budget gave the model ~2.4KB of a ~198KB document to summarize from.
    This tool is document-scoped, samples evenly across the whole document, carries
    the LLM-derived abstract written at ingest, and gets its own larger budget.

``list_knowledge_base_documents``
    The titles in the knowledge base, so a document can be named before it is
    summarized (and so "what do I have on X?" is answerable at all).

Uses a test-injectable httpx transport and retries on transient 5xx / transport
errors. ``user_id`` is threaded from graph state (like
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
# KB context size fed into the (small, CPU-bound) local model. Smaller = far less
# prompt prefill = faster TTFT: measured ~34s of a recipe turn's ~37s TTFT was CPU
# prefill of the injected chunks. Tunable via env to trade grounding for speed
# without a rebuild (raise for more complete answers, lower for faster ones).
DEFAULT_TOP_K = int(os.environ.get("KB_TOP_K", "4"))       # chunks retrieved (was 5)
MAX_CHUNKS = int(os.environ.get("KB_MAX_CHUNKS", "4"))     # chunks kept in the prompt (was 8)
MAX_CHARS = int(os.environ.get("KB_MAX_CHARS", "600"))     # per-chunk char cap (was 900)

# Relevance floor for reranked chunks. The retriever ALWAYS returns its top_k nearest
# neighbours, however semantically distant they are — so without a floor an unrelated
# question ("can you see this schedule?") still comes back with the closest documents
# and they get cited as the answer's source. Measured cross-encoder scores on this
# corpus separate cleanly: genuinely relevant queries top out at +1.8 to +5.0, while
# unrelated ones sit at -9 to -11. 0.0 is the cross-encoder relevance boundary and
# lands in the middle of that ~10-point gap.
KB_MIN_SCORE = float(os.environ.get("KB_MIN_SCORE", "0.0"))

# Budgets for summarize_document. Much larger than the chat-path budget above, and
# deliberately so: the small caps exist to keep TTFT down on ORDINARY turns, but a
# summary request is a turn where the user has explicitly asked us to read a whole
# document, and 4 x 600 chars cannot summarize anything. No relevance floor is applied
# either — the user named the document, so its least-relevant passage still belongs to
# the right document, which is more than the corpus-wide floor could promise.
SUMMARY_MAX_CHUNKS = int(os.environ.get("KB_SUMMARY_MAX_CHUNKS", "12"))
SUMMARY_MAX_CHARS = int(os.environ.get("KB_SUMMARY_MAX_CHARS", "1500"))

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


def _get(path: str, user_id: str) -> httpx.Response:
    """GET from the KB service, retrying transient 5xx / transport failures."""
    last_exc: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            with _client(user_id) as c:
                r = c.get(path)
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


def relevant_chunks(payload: dict) -> list[dict]:
    """Chunks that clear KB_MIN_SCORE, so an unrelated question doesn't get answered
    (or cited) from whatever happened to be nearest in vector space.

    Fail-open by design: if NO chunk carries a ``score`` the floor cannot be applied,
    so every chunk is kept. That keeps this working against a KB service that doesn't
    return scores rather than silently returning nothing. Pure.
    """
    chunks = payload.get("chunks") or []
    scored = [c for c in chunks if isinstance(c.get("score"), (int, float))]
    if not scored:
        return list(chunks)
    return [c for c in chunks if float(c.get("score", 0.0)) >= KB_MIN_SCORE]


def format_context(payload: dict) -> str:
    """Turn the retrieve-only response into a compact, title-cited context block. Pure."""
    chunks = relevant_chunks(payload)
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
    """Distinct source-document titles that fed the context (for the answer's citation).

    Derived from the chunks that SURVIVED the relevance floor, not from the raw
    ``sources`` list, so the citation reflects what the model was actually shown. The
    retriever returns its nearest neighbours regardless of distance, so citing the raw
    list attaches an authoritative-looking source to an answer that never used it.

    Falls back to ``sources`` when the payload carries no chunks (older service shape,
    and what the unit tests exercise).
    """
    chunks = relevant_chunks(payload)
    if not chunks and not (payload.get("chunks") or []):
        seen: set[str] = set()
        out: list[str] = []
        for s in payload.get("sources") or []:
            title = (s.get("title") or "").strip()
            if title and title not in seen:
                seen.add(title)
                out.append(title)
        return out

    seen = set()
    out = []
    for c in chunks[:MAX_CHUNKS]:
        title = (c.get("title") or "").strip()
        if title and title not in seen:
            seen.add(title)
            out.append(title)
    return out


def format_document_digest(payload: dict) -> str:
    """Render /api/summarize into a context block the composer can summarize from. Pure.

    Order matters: the ingest-time abstract comes FIRST so a model that reads only the
    head of a long context still gets an accurate whole-document overview, and the
    sampled excerpts follow as supporting detail it can quote and expand from.
    """
    doc = payload.get("document") or {}
    title = (doc.get("title") or "Untitled document").strip()
    lines: list[str] = [f"DOCUMENT: {title}"]
    if doc.get("created_at"):
        lines.append(f"Added to the knowledge base: {str(doc['created_at'])[:10]}")
    if doc.get("summary"):
        lines.append(f"Overview (written when the document was ingested): {doc['summary'].strip()}")

    chunks = payload.get("chunks") or []
    if not chunks:
        lines.append("No readable passages were stored for this document.")
        return "\n".join(lines)

    lines.append(f"\nExcerpts from this document ({len(chunks[:SUMMARY_MAX_CHUNKS])} passages, in document order):")
    for i, c in enumerate(chunks[:SUMMARY_MAX_CHUNKS], start=1):
        content = " ".join((c.get("content") or "").split())
        if len(content) > SUMMARY_MAX_CHARS:
            content = content[:SUMMARY_MAX_CHARS] + "…"
        lines.append(f"\n[{i}] {content}")
    return "\n".join(lines)


def format_document_list(payload: dict) -> str:
    """Render /api/documents as a plain title list. Pure."""
    docs = payload.get("documents") or []
    if not docs:
        return "The knowledge base has no documents saved."
    lines = [f"{len(docs)} document(s) in the knowledge base:"]
    for d in docs:
        title = (d.get("title") or "Untitled").strip()
        added = str(d.get("created_at") or "")[:10]
        lines.append(f"- {title}" + (f" (added {added})" if added else ""))
    return "\n".join(lines)


def _titles_hint(user_id: str) -> str:
    """Available titles, for the 'no such document' message. Never raises."""
    try:
        docs = (_get("/api/documents", user_id).json().get("documents") or [])[:15]
    except Exception:  # noqa: BLE001 - a hint is optional; the error message is not
        return ""
    titles = [f'"{(d.get("title") or "Untitled").strip()}"' for d in docs]
    return (" Documents currently in the knowledge base: " + "; ".join(titles)) if titles else ""


def summarize_document(user_id: str, document: str, focus: str | None = None) -> tuple[str, list[str]]:
    """Digest ONE named document. Returns (context, source_titles).

    A reference that matches nothing returns an error string naming the documents that
    DO exist, rather than falling back to a corpus-wide search. Falling back is what
    produced a confident summary of the wrong week's report — a wrong answer that looks
    exactly like a right one is worse than an answer that says it couldn't find the
    document.
    """
    if not document or not document.strip():
        return "error: no document named", []
    body: dict = {"document_title": document.strip(), "max_chunks": SUMMARY_MAX_CHUNKS}
    if focus and focus.strip():
        body["focus"] = focus.strip()
    try:
        r = _post("/api/summarize", user_id, body)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return (
                f'error: no document in the knowledge base matches "{document.strip()}".'
                + _titles_hint(user_id),
                [],
            )
        raise
    payload = r.json()
    title = ((payload.get("document") or {}).get("title") or "").strip()
    return format_document_digest(payload), ([title] if title else [])


def list_knowledge_base_documents(user_id: str) -> tuple[str, list[str]]:
    """List the titles saved in the knowledge base. Returns (context, [])."""
    return format_document_list(_get("/api/documents", user_id).json()), []


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
    "summarize_document": lambda uid, a: summarize_document(
        uid, a.get("document") or "", a.get("focus")
    ),
    "list_knowledge_base_documents": lambda uid, _a: list_knowledge_base_documents(uid),
}

# The set of tool names the tool_execution node recognizes as KB tools.
GRAPHRAG_TOOL_REGISTRY = frozenset(_DISPATCH)

# The subset the supervisor's consult-once-per-turn guard applies to.
#
# That guard exists because a corpus-wide retrieval re-run with reworded terms returns
# the same chunks and loops the supervisor to its step bound. It must NOT cover the
# document tools: "what do I have on securitized products?" followed by "summarize the
# August 28 one" is two DIFFERENT questions, and the second one is the whole point of
# these tools. Naming the subset explicitly means adding a KB tool no longer silently
# widens the guard.
KB_RETRIEVE_ONCE_TOOLS = frozenset({"query_knowledge_base"})


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
