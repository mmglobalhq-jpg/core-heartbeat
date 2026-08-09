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

PROMPT_VERSION = "2026-08-09.3"
"""Bumped whenever the wording below changes, and recorded in run_meta.

v1 was too strict and it took a live run to see it: told to "be strict" and to
include a story only if it was "really about" the subject, the model kept 3 of 8
for "Financial Markets" and struck the Lisa Cook story from "Mortgages" even
though the piece turned on mortgage-fraud allegations. Reading a subject as
"the headline is exclusively this" throws away most of what a person following
that subject actually wants. v2 asks the question the reader asks — would
someone following this want to see it — while keeping the distinction that
made the judge worth having: a shared word is not a shared subject."""

SYSTEM = (
    "You curate a personal news briefing. You are given one subject and a "
    "numbered list of story headlines. Decide which stories a person who follows "
    "that subject would want to see. "
    "Include a story when the subject is a real part of it — the main subject, a "
    "significant element, or a direct cause or consequence of it. The story does "
    "not have to be exclusively about the subject, and it does not have to use "
    "the same words. "
    "Exclude a story that shares only a word or a surface association. Different "
    "sports, leagues, organisations, companies or senses of a word are different "
    "subjects. "
    "Use what you know about the subject: abbreviations, institutions, people, "
    "teams, places and events associated with it. "
    "Reply with JSON only, no prose."
)

_PROMPT = """Subject: {topic}

Below is a numbered list of candidate story headlines.

{fenced}

For the subject "{topic}", return JSON of exactly this shape and nothing else:

{{"relevant": [<numbers of the stories worth showing>]}}

Include a story if someone following "{topic}" would want to read it — the
subject can be the main story, a significant part of it, or a direct cause or
consequence. Leave out stories that merely share a word with it, or that are
about a different sport, league, organisation or sense of the word. When a story
is genuinely borderline, include it. If none qualify, return {{"relevant": []}}.
Do not explain."""


INTERPRET_SYSTEM = (
    "You normalise the subjects a person typed into their news preferences. "
    "For each subject, return the standard name for what they plainly meant and "
    "a few extra terms that news stories about it would actually use — team "
    "names, people, institutions, tickers, common abbreviations. "
    "Correct obvious misspellings. Do not reinterpret a subject into something "
    "else: if you cannot tell what was meant, return it unchanged. "
    "Reply with JSON only, no prose."
)

_INTERPRET_PROMPT = """Here are the subjects a reader follows:

{fenced}

Return JSON of exactly this shape and nothing else, one key per subject
EXACTLY as it was written above:

{{"<subject as written>": {{"name": "<standard name>", "terms": ["<other terms news would use>"]}}}}

Correct clear misspellings — a subject one or two letters away from a well-known
name is a typo, not a different subject. Keep "name" short: it is used to search
headlines. Give at most five extra terms, and only ones that would really appear
in a story about the subject. If a subject is already correct, return it
unchanged with useful extra terms. Do not explain."""


def interpret(topics: list[str], *,
              budget: EscalationBudget | None = None) -> tuple[dict[str, dict], dict]:
    """Resolve what each typed subject actually means, before any matching.

    WHY THIS RUNS FIRST, AND WHY THE JUDGE COULD NOT DO IT.
    The judge is a precision filter: it only ever sees candidates that embeddings
    or exact tokens already surfaced, and it can only take a topic away. A
    misspelled subject fails BEFORE any of that — "Wall Streat" embeds nowhere
    near a market story and shares no token with one, so it surfaces nothing and
    the model is never asked. The failure is upstream of the only place a model
    was looking. Interpretation has to happen before candidate selection or it
    cannot help at all.

    Returns ``{original: {"name": str, "terms": [str]}}``. The ORIGINAL string
    stays the key and stays what the user sees; only matching uses the resolved
    form, so a wrong interpretation degrades targeting rather than silently
    rewriting someone's preferences.

    Fails open like everything else here: no reply means topics are used exactly
    as typed, which is the behaviour that existed before this step.
    """
    topics = [t.strip() for t in (topics or []) if t and t.strip()]
    if not topics:
        return {}, {}
    prompt = _INTERPRET_PROMPT.format(
        fenced=wrap("\n".join(f"- {t}" for t in topics), label="READER SUBJECTS"))
    stats: dict = {"asked": len(topics)}
    raw = ""
    if budget is not None and budget.remaining > 0:
        try:
            raw = generate_escalated(prompt, system=INTERPRET_SYSTEM, budget=budget,
                                     temperature=0.0)
            stats["model"] = "hosted"
        except EscalationExhausted:
            logger.info("no escalation budget for topic interpretation")
        except Exception as exc:  # noqa: BLE001
            logger.warning("hosted topic interpretation failed (%s)", exc)
    if not raw:
        try:
            raw = generate_local(prompt, system=INTERPRET_SYSTEM, temperature=0.0)
            stats["model"] = "local"
        except Exception as exc:  # noqa: BLE001
            logger.warning("local topic interpretation failed (%s)", exc)

    out: dict[str, dict] = {}
    match = re.search(r"\{.*\}", raw or "", re.S)
    if match:
        try:
            data = json.loads(match.group(0))
        except (ValueError, TypeError):
            data = {}
        if isinstance(data, dict):
            for topic in topics:
                entry = data.get(topic)
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("name") or "").strip() or topic
                terms = [str(x).strip() for x in (entry.get("terms") or [])
                         if str(x).strip()][:5]
                out[topic] = {"name": name, "terms": terms}
    if not out:
        stats["interpreted"] = False
        return {}, stats
    stats["interpreted"] = True
    stats["corrected"] = {t: v["name"] for t, v in out.items()
                          if v["name"].lower() != t.lower()}
    return out, stats


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


def _combined_prompt(items_by_topic: dict[str, list[RawItem]]) -> str:
    """One prompt covering every topic.

    Deliberately not one call per topic. Measurement forced this: the local model
    is not reliable enough for the task (it returned unparseable JSON for one
    topic and wrongly kept zero for another, where the hosted model got both
    right), so the hosted model has to be reachable in the normal path — and at
    one call per topic that would cost an escalation per topic and scale with how
    many subjects someone follows. Combined, the whole pass is ONE call however
    many topics there are.
    """
    blocks = []
    for topic, items in items_by_topic.items():
        blocks.append(f"### Subject: {topic}\n{_candidates_block(items)}")
    body = "\n\n".join(blocks)
    return f"""You are given several subjects, each with its own numbered list of
candidate story headlines.

{wrap(body, label="CANDIDATE HEADLINES")}

Return JSON of exactly this shape and nothing else, with one key per subject
exactly as written above:

{{"<subject>": [<numbers worth showing for that subject>]}}

Numbering restarts at 1 within each subject. Include a story if someone
following that subject would want to read it — the subject can be the main
story, a significant part of it, or a direct cause or consequence. Leave out
stories that merely share a word, or that are about a different sport, league,
organisation or sense of the word. When a story is genuinely borderline,
include it. A subject with nothing relevant gets an empty list. Do not explain."""


def _parse_combined(reply: str, items_by_topic: dict[str, list[RawItem]]
                    ) -> dict[str, set[str]] | None:
    """Topic -> confirmed item hashes, or None if the reply was unusable."""
    if not reply:
        return None
    match = re.search(r"\{.*\}", reply, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    out: dict[str, set[str]] = {}
    for topic, items in items_by_topic.items():
        raw = data.get(topic)
        if not isinstance(raw, list):
            continue          # topic absent from the reply: leave it untouched
        kept: set[str] = set()
        for value in raw:
            try:
                n = int(value)
            except (ValueError, TypeError):
                continue
            if 1 <= n <= len(items):
                kept.add(items[n - 1].hash)
        out[topic] = kept
    return out or None


def judge(items_by_topic: dict[str, list[RawItem]], *,
          budget: EscalationBudget | None = None) -> tuple[dict[str, set[str]], dict]:
    """Confirm every topic's candidates in one pass.

    HOSTED FIRST, and that is a change of position made on measurement rather
    than preference. The first version tried local and escalated only when the
    reply would not parse. On real headlines the local model produced unparseable
    JSON for one topic and confidently kept ZERO for another where the hosted
    model kept the right story — and a confidently wrong answer never triggers
    escalation, so the bad verdict simply stood. Local remains the fallback, so a
    briefing still gets some judging with no hosted key or no budget left.
    """
    if not items_by_topic:
        return {}, {}

    prompt = _combined_prompt(items_by_topic)
    stats: dict = {"prompt": PROMPT_VERSION, "enabled": True,
                   "candidates": {t: len(v) for t, v in items_by_topic.items()}}
    verdict = None

    if budget is not None and budget.remaining > 0:
        try:
            verdict = _parse_combined(
                generate_escalated(prompt, system=SYSTEM, budget=budget, temperature=0.0),
                items_by_topic)
            stats["model"] = "hosted"
        except EscalationExhausted:
            logger.info("no escalation budget for topic judging; using local")
        except Exception as exc:  # noqa: BLE001
            logger.warning("hosted topic judge failed (%s); falling back to local", exc)

    if verdict is None:
        try:
            verdict = _parse_combined(
                generate_local(prompt, system=SYSTEM, temperature=0.0), items_by_topic)
            stats["model"] = "local"
        except Exception as exc:  # noqa: BLE001
            logger.warning("local topic judge failed (%s)", exc)

    if verdict is None:
        # Fail open: every candidate keeps the topic it already had.
        stats["judged"] = False
        stats["model"] = stats.get("model", "none")
        return {}, stats

    stats["judged"] = True
    stats["kept"] = {t: len(v) for t, v in verdict.items()}
    return verdict, stats


def enabled() -> bool:
    return bool(config.TOPIC_JUDGE)
