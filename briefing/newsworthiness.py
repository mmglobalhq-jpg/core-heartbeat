"""Drop items that are not stories.

THE PROBLEM
A briefing led with "How to watch Valkyries vs. Sparks: TV channel and streaming
options" and deep-dived on a fantasy football 'Do Not Draft' list. Both ranked
well and neither is news. Ranking scores corroboration, freshness and source
weight — none of which can tell an article from a schedule. The Athletic
publishes a how-to-watch page per fixture, so they arrive fresh, in volume, every
day.

WHY A PATTERN LIST IS ACCEPTABLE HERE, WHEN `untrusted.detect_injection`
DELIBERATELY REFUSES TO BE ONE
That module argues a blocklist is worse than nothing because an ATTACKER
rephrases around it and the filter invites trust it has not earned. Nothing here
is adversarial. These are templated pages generated per fixture by a CMS; nobody
is trying to evade us, the wording is stable because it is machine-produced, and
the failure mode of a miss is one weak item in a briefing rather than a defeated
security control. Different threat model, different tool.

It will still rot. So:

* every pattern below was measured against a live 200-item corpus rather than
  imagined — the set that fired is 6 how-to-watch pages and 1 fantasy column,
  3.5% of the pool, with zero false positives after tuning;
* two candidate patterns were REMOVED for false positives rather than kept for
  completeness. `as it happened` matched "Labor open to 'sensible amendments' on
  gambling reforms, minister says – as it happened", which is real political
  reporting carrying a liveblog suffix. `injury report` would have dropped
  "Sources: Commanders' Tunsil suffers torn tricep", which is a genuine sports
  story;
* the count and the reasons are reported in `run_meta.non_news`, so silent drift
  shows up as a number moving instead of as headlines quietly disappearing.

Set `BRIEFING_FILTER_NON_NEWS=0` to disable.
"""

from __future__ import annotations

import re

from briefing import config
from briefing.models import RawItem

# Each entry is (reason, pattern). The reason is what gets counted in run_meta,
# so a pattern that starts over-firing is attributable to itself.
_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    # Per-fixture viewing pages. "how to watch X vs Y" is a CMS template.
    ("how_to_watch", re.compile(r"\bhow to watch\b|\bwhere to watch\b|\bwhat channel\b", re.I)),
    # Broadcast/streaming logistics rather than an account of an event.
    ("broadcast_info", re.compile(r"\bTV channel\b|\blive stream\b|\bstreaming options?\b", re.I)),
    # Fantasy sports advice columns.
    ("fantasy", re.compile(
        r"\bfantasy (football|basketball|baseball|hockey|sports)\b"
        r"|\bdo not draft\b|\bwaiver wire\b|\bstart[/ ]sit\b"
        r"|\bdraft (rankings|sleepers|busts)\b", re.I)),
    # Gambling cards. Deliberately NOT a bare \bodds\b — "odds of a recession"
    # is ordinary business writing and matched real stories in testing.
    ("betting", re.compile(
        r"\bbetting odds\b|\bodds and picks\b|\bprop bets?\b|\bbest bets\b"
        r"|\bpicks and predictions\b|\bparlay\b", re.I)),
    # Rolling coverage pages. `as it happened` is excluded on purpose — see the
    # module docstring; it matched genuine political reporting.
    ("liveblog", re.compile(r"\blive updates\b|\blive blog\b", re.I)),
)


def reason(item: RawItem) -> str | None:
    """Why this item is not a story, or None if it is one.

    Matches on the TITLE only. Summaries quote article text and mention
    broadcast details often enough that including them cost real stories in
    testing.
    """
    title = item.title or ""
    for name, pattern in _RULES:
        if pattern.search(title):
            return name
    return None


def filter_items(items: list[RawItem]) -> tuple[list[RawItem], dict]:
    """Drop non-stories. Returns the survivors and a per-reason count."""
    if not config.FILTER_NON_NEWS:
        return items, {}
    kept: list[RawItem] = []
    counts: dict[str, int] = {}
    for item in items:
        why = reason(item)
        if why:
            counts[why] = counts.get(why, 0) + 1
        else:
            kept.append(item)
    # An empty result would mean the filter ate the briefing. Refuse rather than
    # hand back nothing: a briefing of listings beats no briefing, and this is a
    # quality preference, not a correctness rule.
    if not kept:
        return items, {"disabled_would_empty": len(items)}
    return kept, counts
