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

        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model=os.environ.get(SEARCH_MODEL_ENV) or DEFAULT_SEARCH_MODEL,
            contents=query,
            config=types.GenerateContentConfig(
                # Grounding is a config-level tool, not a function declaration — it
                # cannot be mixed with our own tool schemas, which is why search is
                # its own isolated call rather than part of the routing request.
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
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
    """Grounded search returning STRUCTURED results.

    ``search_web`` above answers a question in prose for the assistant to speak.
    The briefing pipeline needs the underlying sources as data — it ranks and
    deduplicates them, then fetches the survivors. Re-parsing the prose form to
    recover the URLs would be lossy and would break the moment the wording
    changed, so the grounding metadata is read directly here instead.

    Returns ``[{"url", "title", "snippet", "source"}, ...]``; empty on any
    failure, because a briefing missing one source is better than a briefing that
    did not run.
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
            contents=f"What are the most significant news stories about: {query}",
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
