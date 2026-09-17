"""Recognise a main-chat request that is really about one of the user's SAVED documents.

WHY THIS EXISTS (2026-09-17)
The knowledge base moved out of the main chat into Knowledge chat. The main chat's
prompts say so, but a replay of real prompts showed that prose alone does not hold:
asked "summarize the August 28 securitized products report", the model either asked
which REIT it meant or — once — invented a Morgan Stanley "Overweight" recommendation
from nothing. Before the move, a forced KB search had at least supplied real text.

So, like the platform's other routing guards, this is deterministic: when the router
called no tool and the message is clearly about a saved document, the main chat answers
with a fixed pointer to Knowledge chat instead of letting the composer improvise.

WHAT COUNTS
  * an explicit reference ("my saved research", "my documents", "knowledge base"), or
  * a report-shaped request ("report", "note", "weekly", "research"…) that names a
    third-party publisher, or shares two distinctive words with a document title in
    the user's own library (one quick call to the KB service, only for report-shaped
    messages).
ARR/ARMOUR/ORC/Orchid references are never matched: those reports are in this chat.
"""
from __future__ import annotations

import os
import re

import httpx

_REIT = re.compile(r"\b(arr|armour|orc|orchid)\b", re.I)
_EXPLICIT = re.compile(
    r"\b(my\s+(saved\s+)?(research|documents?|docs|files|reports|pdfs?)|saved\s+(research|documents?|reports?|files?)"
    r"|knowledge\s*base|knowledge\s+chat)\b",
    re.I,
)
_REPORTISH = re.compile(
    r"\b(reports?|research|notes?|weekly|monthly|quarterly|publications?|documents?|tracker|dashboard"
    r"|monitor|outlook|pdf|issue|brief(?!ing))\b",
    re.I,
)
_PUBLISHER = re.compile(
    r"\b(j\.?\s?p\.?\s?morgan|jpm|morgan\s+stanley|goldman(\s+sachs)?|barclays|citi(group)?|bofa|bank\s+of\s+america"
    r"|wells\s+fargo|ubs|deutsche|nomura|credit\s+suisse|kbra|moody'?s|fitch)\b",
    re.I,
)
_STOP = frozenset(
    "what with from that this have about their there which would could should north america "
    "report reports research weekly issue note notes document summary summarize highlights".split()
)

LIST_TIMEOUT_S = 3.0


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in _STOP}


def _library_titles(user_id: str) -> list[dict]:
    url = (os.environ.get("GRAPHRAG_SERVICE_URL") or "").rstrip("/")
    key = os.environ.get("GRAPHRAG_API_KEY") or ""
    if not url or not key:
        return []
    try:
        r = httpx.get(
            f"{url}/api/documents",
            headers={"Authorization": f"Bearer {key}", "X-User-Id": user_id},
            timeout=LIST_TIMEOUT_S,
        )
        r.raise_for_status()
        return r.json().get("documents") or []
    except Exception:
        return []  # a hint, never a dependency


def match_title(text: str, docs: list[dict]) -> str | None:
    """The library document the message names, if two or more distinctive title words
    appear in it; a matching number (a date's day, a year) breaks ties."""
    words = _words(text)
    numbers = set(re.findall(r"\b\d{1,4}\b", text))
    best: tuple[int, int, str] | None = None
    for d in docs:
        title = (d.get("title") or "").strip()
        shared = len(words & _words(title))
        if shared < 2:
            continue
        num_hits = len(numbers & set(re.findall(r"\b\d{1,4}\b", title)))
        key = (shared, num_hits, title)
        if best is None or key[:2] > best[:2]:
            best = key
    return best[2] if best else None


def saved_document_reference(text: str, user_id: str, *, fetch=_library_titles) -> str | None:
    """None when the message is not about a saved document. Otherwise the matched title,
    or "" when it clearly is one but no single title was identified."""
    raw = text or ""
    if not raw.strip() or _REIT.search(raw):
        return None
    if _EXPLICIT.search(raw):
        return match_title(raw, fetch(user_id)) or ""
    if not _REPORTISH.search(raw):
        return None
    title = match_title(raw, fetch(user_id))
    if title:
        return title
    return "" if _PUBLISHER.search(raw) else None


def pointer_reply(title: str) -> str:
    named = f" (“{title}”)" if title else ""
    return (
        f"That's about one of your saved documents{named}, and those are answered in "
        "**Knowledge chat** — switch to *Knowledge* at the top of the sidebar and ask there. "
        "It answers only from your documents and shows the passages it used."
    )
