#!/usr/bin/env python3
"""Check every configured feed: permitted, parseable, and carrying fresh items.

Feeds rot quietly. A publisher moves a URL, tightens robots.txt, or lets a feed
go stale, and the briefing simply gets thinner — nothing errors. This makes that
visible on demand.

    scripts/check_feeds.py            # default sources
    scripts/check_feeds.py --url URL  # check one candidate before adding it

Exits non-zero if any configured feed is unusable, so it can be wired to a timer
later if that turns out to be worth doing.
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FRESH_HOURS = 36


def check(name: str, url: str) -> tuple[bool, str]:
    import feedparser

    from briefing import config
    from briefing.sources import robots_allows

    if not robots_allows(url):
        # Distinguish "they said no" from "we could not read the rules" — the
        # first is a decision to respect, the second is usually our bug.
        return False, "robots.txt disallows this path for our agent"

    parsed = feedparser.parse(url, agent=config.USER_AGENT)
    entries = getattr(parsed, "entries", [])
    if not entries:
        return False, f"no entries (bozo={int(getattr(parsed, 'bozo', 0))})"

    now = dt.datetime.now(dt.UTC)
    fresh = 0
    for entry in entries:
        stamp = entry.get("published_parsed") or entry.get("updated_parsed")
        if stamp:
            age = (now - dt.datetime(*stamp[:6], tzinfo=dt.UTC)).total_seconds() / 3600
            if age <= FRESH_HOURS:
                fresh += 1

    note = f"{len(entries):3} entries, {fresh:3} within {FRESH_HOURS}h"
    if fresh == 0:
        # Parseable but useless to a daily briefing.
        return False, note + "  <-- nothing recent"
    return True, note


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="check a single candidate feed")
    args = parser.parse_args()

    if args.url:
        ok, note = check("candidate", args.url)
        print(f"  {'OK  ' if ok else 'FAIL'}  {args.url}\n        {note}")
        return 0 if ok else 1

    from briefing.sources import DEFAULT_SOURCES

    failures = 0
    for spec in DEFAULT_SOURCES:
        ok, note = check(spec.name, spec.url or "")
        if not ok:
            failures += 1
        print(f"  {'OK  ' if ok else 'FAIL'}  {spec.name:24} {note}")

    print(f"\n  {len(DEFAULT_SOURCES) - failures}/{len(DEFAULT_SOURCES)} feeds usable")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
