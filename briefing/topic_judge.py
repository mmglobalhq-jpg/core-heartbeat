"""Decide whether a story is genuinely about a user's topic.

WHY A MODEL AND NOT JUST EMBEDDINGS
Embeddings answer "is this text nearby in meaning", which is not the same
question. Measured on the live corpus, `"UGA football"` scored highest against a
World Cup soccer story and a Ted Lasso newsletter, and the Dawg Sports item it
did surface was basketball recruiting. Cosine similarity has no way to know that
UGA means the University of Georgia, that its football team is coached by Kirby
Smart, or that college football and association football are different sports.
A model knows all three.

WHAT THIS IS AND IS NOT
It is a PRECISION filter, not a recall mechanism. It only ever judges candidates
that embeddings or exact tokens already surfaced, and it can only take topics
away — never add them. Recall stays the job of the feeds and the embedding pass,
which are cheap; judging all ~130 items every run would cost far more and buy
nothing, because a story no candidate pass surfaced was not going to be promoted
anyway.

COST
One local call per topic, with that topic's candidates batched into it. Three
topics is three small local calls. Hosted escalation happens only when the local
model returns something unparseable, and it spends from the same
`EscalationBudget` as the rest of the briefing, so the ceiling is shared and
cannot be exceeded by adding topics.

IT FAILS OPEN, DELIBERATELY
If the model is unreachable, or returns nothing usable, every candidate keeps the
topic it already had. That matches how the rest of the pipeline degrades —
embeddings unavailable falls back to token overlap — and it means an Ollama
outage produces a slightly less well-targeted briefing rather than one with no
tailoring at all. Failing closed would silently delete the feature on the day the
model was down, which is the worse outcome and the harder one to notice.

DETERMINISM
Temperature 0, and verdicts are applied only as a filter over an already
deterministic candidate list, so the blast radius of any model variance is one
topic label on one story rather than a reordering of the whole briefing. This is
weaker than the guarantee the rest of ranking has, and it is a real trade: §7 #5
of the briefing doc records that nondeterministic ranking once produced two
different briefings from one input. Set `BRIEFING_TOPIC_JUDGE=0` to turn the
whole pass off and return to purely deterministic selection.
"""

from __future__ import annotations

import json
import logging
import re

from briefing import config
from briefing.llm import EscalationBudget, EscalationExhausted, generate_escalated, generate_local
from briefing.models import RawItem
from briefing.untrusted import wrap

logger = logging.getLogger(__name__)

SYSTEM = (
    "You classify news stories by subject for a personal news briefing. "
    "You are given one subject and a numbered list of story headlines. "
    "Decide which stories are genuinely ABOUT that subject. "
    "Use your knowledge of what the subject refers to: abbreviations, "
    "institutions, people associated with it, and the difference between "
    "sports, leagues and organisations that share a word. "
    "Be strict. A story that merely shares a word with the subject is not about "
    "it. Reply with JSON only, no prose."
)

_PROMPT = """Subject: {topic}

Below is a numbered list of candidate story headlines.

{fenced}

For the subject "{topic}", return JSON of exactly this shape and nothing else:

{{"relevant": [<numbers of the stories genuinely about the subject>]}}

Include a number only if the story is really about that subject. If none of them
are, return {{"relevant": []}}. Do not explain."""


def _candidates_block(items: list[RawItem]) -> str:
    lines = []
    for n, it in enumerate(items, 1):
        summary = (it.summary or "").strip().replace("\n", " ")
        lines.append(f"{n}. {it.title}" + (f" — {summary[:180]}" if summary else ""))
    return "\n".join(lines)


def _parse(reply: str, count: int) -> set[int] | None:
    """Indices the model called relevant, or None if it did not answer usably.

    None and an empty set mean different things and must not be conflated: an
    empty set is the model saying "none of these", which is a real and common
    answer for a topic the feeds do not cover. None is a failure, and only None
    triggers escalation or fail-open.
    """
    if not reply:
        return None
    match = re.search(r"\{.*\}", reply, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    raw = data.get("relevant")
    if not isinstance(raw, list):
        return None
    out: set[int] = set()
    for value in raw:
        try:
            n = int(value)
        except (ValueError, TypeError):
            continue
        # Silently drop out-of-range indices rather than failing the batch: a
        # model that hallucinates "7" out of 5 candidates has still answered
        # usefully about the five that exist.
        if 1 <= n <= count:
            out.add(n)
    return out


def judge_topic(topic: str, items: list[RawItem], *,
                budget: EscalationBudget | None = None) -> set[str] | None:
    """Item hashes genuinely about ``topic``. None if the model did not answer."""
    if not items:
        return set()
    prompt = _PROMPT.format(
        topic=topic,
        fenced=wrap(_candidates_block(items), label="CANDIDATE HEADLINES"),
    )
    reply = ""
    try:
        reply = generate_local(prompt, system=SYSTEM, temperature=0.0)
    except Exception as exc:  # noqa: BLE001 — any local failure is escalatable
        logger.warning("local topic judge failed for %r (%s)", topic, exc)

    picked = _parse(reply, len(items))
    if picked is None and budget is not None and budget.remaining > 0:
        logger.info("local judge unusable for %r; escalating", topic)
        try:
            reply = generate_escalated(prompt, system=SYSTEM, budget=budget,
                                       temperature=0.0)
            picked = _parse(reply, len(items))
        except EscalationExhausted:
            logger.info("no escalation budget left for topic judging")
        except Exception as exc:  # noqa: BLE001
            logger.warning("escalated topic judge failed for %r (%s)", topic, exc)

    if picked is None:
        return None
    return {items[n - 1].hash for n in picked}


def judge(items_by_topic: dict[str, list[RawItem]], *,
          budget: EscalationBudget | None = None) -> tuple[dict[str, set[str]], dict]:
    """Confirm each topic's candidates.

    Returns ``(confirmed, stats)`` where ``confirmed`` maps topic -> item hashes
    the model kept. A topic missing from ``confirmed`` was not judged (model
    unavailable) and its candidates must be left exactly as they were.
    """
    confirmed: dict[str, set[str]] = {}
    stats: dict[str, dict] = {}
    for topic, candidates in items_by_topic.items():
        kept = judge_topic(topic, candidates, budget=budget)
        if kept is None:
            stats[topic] = {"candidates": len(candidates), "judged": False}
            continue
        confirmed[topic] = kept
        stats[topic] = {
            "candidates": len(candidates),
            "kept": len(kept),
            "rejected": len(candidates) - len(kept),
            "judged": True,
        }
    return confirmed, {"topics": stats, "enabled": True}


def enabled() -> bool:
    return bool(config.TOPIC_JUDGE)
