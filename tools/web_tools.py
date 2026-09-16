"""Web search and page reading.

Until now the assistant could not reach the internet at all. Asked something the
knowledge base didn't cover it would answer from training data and — because the
composer is told to offer rather than refuse — sometimes offer to "search the web",
which it could not do. These are the two tools that make that offer true.

``search_web`` uses Gemini's built-in Google Search grounding rather than a
third-party search API: the credential already exists, Google runs the query and
returns an answer grounded in current results with source URLs, and there is no new
service to operate. The trade is that we get Google's synthesis rather than raw
results to rank ourselves.

``fetch_url`` reads one specific page. Its safety is not in this module — see
services/web.py, which is where the SSRF checks live, and why.
"""

from __future__ import annotations

import logging
import os

from services.web import WebFetchError, fetch_page

logger = logging.getLogger(__name__)

SEARCH_MODEL_ENV = "WEB_SEARCH_MODEL"
DEFAULT_SEARCH_MODEL = "gemini-2.5-flash"
GOOGLE_KEY_ENVS = ("GOOGLE_API_KEY", "GEMINI_API_KEY")
MAX_SEARCH_CHARS = 6000


# How many sources find_sources returns. Six is enough to cover the plausible
# sites for a question without turning the tool result into a link dump the model
# has to wade through before it can read anything.
DEFAULT_SOURCE_COUNT = 6
MAX_SOURCE_COUNT = 10


def _google_key() -> str | None:
    for env in GOOGLE_KEY_ENVS:
        value = (os.environ.get(env) or "").strip()
        if value:
            return value
    return None


def search_web(user_id: str, args: dict) -> str:
    """Answer a question from current web results, with sources."""
    query = (args.get("query") or "").strip()
    if not query:
        return "error: empty search query"
    key = _google_key()
    if not key:
        return "error: web search is not configured (no Google API key)"

    try:
        from google import genai
        from google.genai import types

        from services import llm_ledger

        client = genai.Client(api_key=key)
        model = os.environ.get(SEARCH_MODEL_ENV) or DEFAULT_SEARCH_MODEL
        # Grounded search is billed TWICE: the tokens, and the grounded prompt itself
        # (free below a daily allowance, then per 1,000). The ledger records both, which
        # nothing did before — this call was entirely invisible (doc 07 3a).
        with llm_ledger.attempt("gemini", model, "heartbeat.web_search") as record:
            response = client.models.generate_content(
                model=model,
                contents=query,
                config=types.GenerateContentConfig(
                    # Grounding is a config-level tool, not a function declaration — it
                    # cannot be mixed with our own tool schemas, which is why search is
                    # its own isolated call rather than part of the routing request.
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                ),
            )
            record.ok(
                getattr(response, "usage_metadata", None),
                model_served=getattr(response, "model_version", None),
                request_id=getattr(response, "response_id", None),
                grounded_prompts=1,
            )
    except Exception as exc:
        logger.warning("web search failed: %s", exc)
        return f"error: web search failed ({type(exc).__name__})"

    text = (getattr(response, "text", "") or "").strip()
    if not text:
        return "No results found for that search."
    if len(text) > MAX_SEARCH_CHARS:
        text = text[:MAX_SEARCH_CHARS] + "\n\n[truncated]"

    # Surface the grounding sources so the answer can be attributed rather than
    # presented as though the model simply knew it.
    urls: list[str] = []
    try:
        for candidate in getattr(response, "candidates", None) or []:
            meta = getattr(candidate, "grounding_metadata", None)
            for chunk in getattr(meta, "grounding_chunks", None) or []:
                web = getattr(chunk, "web", None)
                uri = getattr(web, "uri", None)
                title = getattr(web, "title", None) or uri
                if uri and uri not in urls:
                    urls.append(uri)
                    text_line = f"- {title}: {uri}"
                    if len(urls) == 1:
                        text += "\n\nSources:"
                    text += f"\n{text_line}"
                if len(urls) >= 6:
                    break
    except Exception:  # sources are a nicety; never fail the tool over them
        pass
    return text


def search_web_raw(query: str, *, max_results: int = 10) -> list[dict[str, str]]:
    """Grounded search returning STRUCTURED results — a reading list, not an answer.

    ``search_web`` above answers a question in prose. This returns the underlying
    sources as data so a caller can pick pages and read them with ``fetch_url``.
    Re-parsing the prose form to recover URLs would be lossy and would break the
    moment the wording changed, so the grounding metadata is read directly.

    The URLs are Google ``grounding-api-redirect`` links rather than the sites
    themselves. ``fetch_url`` follows them (re-validating every hop), so they are
    usable as-is — verified 2026-08-25. They do expire: a stale one returns 429,
    so fetch within the same turn rather than storing them.

    Returns ``[{"url", "title", "snippet", "source"}, ...]``; empty on any failure,
    because a caller with no sources is better than one that raised. ``snippet`` is
    always empty — grounding metadata carries no excerpt — and ``title`` is often
    just the domain, so the choice of what to read is made on the domain. The real
    page title is the first line of what ``fetch_url`` returns.
    """
    key = _google_key()
    if not key or not query.strip():
        return []
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model=os.environ.get(SEARCH_MODEL_ENV) or DEFAULT_SEARCH_MODEL,
            # Was hardcoded to "What are the most significant news stories about:",
            # which was the briefing's framing leaking into a general helper — it
            # turned "Memphis weather this week" into a search for news ABOUT the
            # weather. Callers that want news say so in their query.
            contents=query,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
    except Exception as exc:
        logger.warning("structured web search failed for %r: %s", query, exc)
        return []

    results: list[dict[str, str]] = []
    seen: set[str] = set()
    try:
        for candidate in getattr(response, "candidates", None) or []:
            meta = getattr(candidate, "grounding_metadata", None)
            for chunk in getattr(meta, "grounding_chunks", None) or []:
                web = getattr(chunk, "web", None)
                uri = getattr(web, "uri", None)
                if not uri or uri in seen:
                    continue
                seen.add(uri)
                results.append(
                    {
                        "url": uri,
                        "title": getattr(web, "title", None) or uri,
                        "snippet": "",
                        "source": getattr(web, "domain", None) or getattr(web, "title", "") or "",
                    }
                )
                if len(results) >= max_results:
                    return results
    except Exception:  # noqa: BLE001 — metadata shape varies by SDK version
        logger.debug("could not read grounding metadata", exc_info=True)
    return results


def find_sources(user_id: str, args: dict) -> str:
    """List web pages worth reading for a query, for ``fetch_url`` to open."""
    query = (args.get("query") or "").strip()
    if not query:
        return "error: empty search query"
    # `or` would fold an explicit 0 into the default; be explicit instead, so a
    # nonsense count clamps to a usable one rather than silently meaning something
    # different from what was asked.
    raw = args.get("max_results")
    try:
        requested = DEFAULT_SOURCE_COUNT if raw is None else int(raw)
    except (TypeError, ValueError):
        requested = DEFAULT_SOURCE_COUNT
    count = max(1, min(requested, MAX_SOURCE_COUNT))

    results = search_web_raw(query, max_results=count)
    if not results:
        return (
            f"No sources found for {query!r}. Say that the search returned nothing "
            "rather than answering from memory."
        )
    lines = [f"Pages worth reading for {query!r} — THIS ORDER IS NOT A RANKING:"]
    for n, r in enumerate(results, 1):
        label = r.get("source") or r.get("title") or "(unknown site)"
        lines.append(f"{n}. {label} — {r['url']}")
    lines.append(
        "Call fetch_url on the one or two most likely to carry the detail. Choose "
        "by WHICH SITE is likely to be authoritative, not by position. The "
        "organisation's own site beats an aggregator, a directory or a social media "
        "page — asked for zoo ticket prices, memphiszoo.org is the answer and "
        "facebook.com is not, whichever came first here. If a page comes back with "
        "little usable text, read the next source rather than giving up. These links "
        "expire, so read them on this turn."
    )
    return "\n".join(lines)


def fetch_url(user_id: str, args: dict) -> str:
    """Read one public web page and return its text."""
    url = (args.get("url") or "").strip()
    if not url:
        return "error: no URL given"
    try:
        final_url, title, text = fetch_page(url)
    except WebFetchError as exc:
        return f"error: {exc}"
    except Exception as exc:
        logger.warning("fetch_url failed for %r: %s", url, exc)
        return f"error: could not read that page ({type(exc).__name__})"
    header = f"{title}\n{final_url}\n\n" if title else f"{final_url}\n\n"
    return header + text


_DISPATCH = {
    "search_web": search_web,
    "find_sources": find_sources,
    "fetch_url": fetch_url,
}

WEB_TOOL_REGISTRY = frozenset(_DISPATCH)


def run_web_tool(name: str, user_id: str, args: dict | None = None) -> str:
    """Execute a registered web tool; never raises."""
    fn = _DISPATCH.get(name)
    if fn is None:
        return f"error: unknown tool {name!r}"
    try:
        return fn(user_id, args or {})
    except Exception as exc:
        return f"error: {type(exc).__name__}: {exc}"
