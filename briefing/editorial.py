"""The human-editorial style pass — constrained so it cannot change the facts.

WHY THIS IS NOT JUST A PROMPT
"Rewrite this for tone but do not change any facts" is a request, not a
guarantee. Models drop a qualifier, round a number, promote "may" to "will", or
swap a name that appeared once for one that appeared often — all while following
the instruction as they understood it. In a briefing, that is the worst possible
failure: the prose gets *better*, which makes the error more convincing.

So the constraint is mechanical. The factual surface of the text is extracted
before and after the rewrite, and the rewrite is rejected if it changed:

* numbers (including percentages, money and dates written numerically)
* calendar dates and years
* capitalised entity names
* URLs
* negation and hedging ("not", "no", "may", "expected to", "reportedly")

The last one is the subtle case and the reason this is not simply a number
diff. "The deal is not expected to close" and "the deal is expected to close"
share every number and every name.

On rejection the ORIGINAL text is kept. A plainer sentence that is true beats a
polished one that is not.
"""

from __future__ import annotations

import logging
import re

from briefing.llm import EscalationBudget, LLMUnavailable, generate
from briefing.models import Section, tidy

logger = logging.getLogger(__name__)


EDITORIAL_SYSTEM = (
    "You are a copy editor for a daily briefing. Improve rhythm, cut padding and "
    "fix clumsy phrasing. You may NOT change what the text says: keep every "
    "number, date, name, quantity, negation and hedge exactly as written. Do not "
    "add facts, do not remove qualifiers, do not resolve uncertainty the text "
    "leaves open, do not add links. Return only the edited prose."
)


# --- factual surface ---------------------------------------------------------

_NUMBER = re.compile(r"(?<![\w.])[$€£]?\d[\d,]*(?:\.\d+)?%?", re.I)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_MONTH = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", re.I)
_URL = re.compile(r"https?://[^\s<>\"')\]]+", re.I)
# A capitalised token not at the start of a sentence — a rough proper-noun proxy.
# Rough is fine: this is a comparison between two texts, so a consistent
# over-count on both sides cancels out.
_ENTITY = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-zA-Z'’-]{2,})\b", re.M)

_HEDGES = frozenset(
    "not no never none cannot without may might could would should likely unlikely "
    "expected reportedly allegedly apparently possibly probably potential proposed "
    "planned alleged claims claimed denied disputed".split()
)


def factual_surface(text: str) -> dict[str, set[str]]:
    """The parts of a text that must survive an edit unchanged."""
    lowered = (text or "").lower()
    return {
        "numbers": {m.group(0).lstrip("$€£").rstrip("%") for m in _NUMBER.finditer(text or "")},
        "years": set(_YEAR.findall(text or "")),
        "months": {m.group(0)[:3].lower() for m in _MONTH.finditer(text or "")},
        "urls": set(_URL.findall(text or "")),
        "entities": set(_ENTITY.findall(text or "")),
        "hedges": {w for w in re.findall(r"[a-z']+", lowered) if w in _HEDGES},
    }


def surface_diff(before: str, after: str) -> dict[str, dict[str, list[str]]]:
    """What the edit changed about the factual surface. Empty means safe.

    ADDITIONS AND REMOVALS ARE BOTH VIOLATIONS. Removing a hedge invents
    certainty; adding one invents doubt. Removing a number drops a fact; adding
    one fabricates it.

    Entities are the exception: an editor legitimately replaces a repeated proper
    noun with a pronoun, so entity *removal* is tolerated while introducing a new
    entity is not.
    """
    a, b = factual_surface(before), factual_surface(after)
    changes: dict[str, dict[str, list[str]]] = {}
    for field in a:
        added = sorted(b[field] - a[field])
        removed = sorted(a[field] - b[field])
        if field == "entities":
            removed = []
        if added or removed:
            changes[field] = {}
            if added:
                changes[field]["added"] = added
            if removed:
                changes[field]["removed"] = removed
    return changes


class EditorialRejected(RuntimeError):
    """The rewrite altered the factual surface."""


def polish(
    text: str,
    *,
    budget: EscalationBudget | None = None,
    prefer_hosted: bool = True,
) -> tuple[str, dict]:
    """Style-edit one passage. Returns ``(text, meta)``.

    Never raises for a bad edit — it returns the original with the rejection
    recorded, because losing a section to a failed polish would be a worse
    outcome than an unpolished section.
    """
    source = tidy(text)
    if not source:
        return source, {"edited": False, "reason": "empty"}

    try:
        edited, model = generate(
            f"Edit the passage below.\n\nPASSAGE:\n{source}",
            system=EDITORIAL_SYSTEM,
            budget=budget,
            prefer_hosted=prefer_hosted,
            temperature=0.4,
        )
    except LLMUnavailable as exc:
        logger.info("editorial pass skipped (%s)", exc)
        return source, {"edited": False, "reason": "model_unavailable"}

    edited = tidy(edited)
    if not edited:
        return source, {"edited": False, "reason": "empty_result"}

    changes = surface_diff(source, edited)
    if changes:
        logger.warning("editorial rewrite rejected — factual surface changed: %s", changes)
        return source, {"edited": False, "reason": "facts_changed", "changes": changes,
                        "model": model}

    # A rewrite that discards most of the passage has not edited it, it has
    # replaced it — and a much shorter text can preserve every number while
    # dropping the sentence that gave them meaning.
    if len(edited) < 0.5 * len(source):
        logger.warning("editorial rewrite rejected — lost %d%% of the text",
                       round(100 * (1 - len(edited) / len(source))))
        return source, {"edited": False, "reason": "too_short", "model": model}

    return edited, {"edited": True, "model": model}


def polish_sections(
    sections: list[Section],
    *,
    budget: EscalationBudget | None = None,
) -> tuple[list[Section], dict]:
    """Run the editorial pass across a briefing.

    Only the Deep Dive is escalated. The Top 5 are two or three sentences each,
    where the hosted model's advantage is small and the budget is better spent on
    the long piece.
    """
    meta = {"polished": 0, "rejected": 0, "reasons": {}}
    for section in sections:
        edited, result = polish(
            section.body,
            budget=budget,
            prefer_hosted=(section.kind == "deep_dive"),
        )
        section.body = edited
        if result.get("edited"):
            meta["polished"] += 1
        else:
            meta["rejected"] += 1
            reason = result.get("reason", "unknown")
            meta["reasons"][reason] = meta["reasons"].get(reason, 0) + 1
    return sections, meta
