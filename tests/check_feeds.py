#!/usr/bin/env python3
"""Check feeds through the pipeline's OWN robots and RSS code. NOT a unit test.

    docker exec core-heartbeat python3 tests/check_feeds.py            # the live list
    docker exec core-heartbeat python3 tests/check_feeds.py URL [URL…] # candidates

`sources.py` has told readers to "re-check with tests/check_feeds.py rather than
assuming" since the feature shipped, and the file did not exist. Written
2026-08-15 while adding business and world feeds.

WHY IT MUST USE THIS MODULE'S OWN CODE
A feed a browser can read is not necessarily one this crawler may read. The
default list carried "Reuters via Google News" for the feature's entire first
day, counted as healthy, returning nothing — because an all-skipped result is a
truthy list. Checking with curl would have agreed it was fine.

It makes real network requests and honours robots.txt, so it lives outside the
pytest suite: the suite must stay offline and deterministic.

READ THE THREE COLUMNS TOGETHER
  robots  DENIED means this crawler may not read it, whatever a browser shows.
  items   readable entries, i.e. not skipped for a paywall/login/anti-bot wall.
  dated   entries carrying a publish time. An UNDATED feed is nearly useless
          here: ranking is recency-weighted, so undated items score as age 0 and
          quietly outrank real news.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/app")

from briefing.models import SourceSpec  # noqa: E402
from briefing.sources import DEFAULT_SOURCES, provider_for, robots_allows  # noqa: E402


def check(spec: SourceSpec) -> tuple[str, int, int, str]:
    try:
        if not robots_allows(spec.url):
            return "DENIED", 0, 0, "robots.txt disallows this crawler"
    except Exception as exc:
        return "ERR", 0, 0, f"robots: {type(exc).__name__}"
    try:
        items = provider_for(spec).fetch(spec)
    except Exception as exc:
        return "ok", 0, 0, f"fetch failed: {type(exc).__name__}: {str(exc)[:40]}"

    readable = [i for i in items if i.readable]
    dated = [i for i in readable if i.published_at]
    if not readable:
        skipped = [i for i in items if i.skipped_reason]
        why = f" ({skipped[0].skipped_reason[:30]})" if skipped else ""
        return "ok", 0, 0, f"NO READABLE ITEMS{why}"
    if not dated:
        return "ok", len(readable), 0, "UNDATED — recency ranking cannot work"
    return "ok", len(readable), len(dated), f"e.g. {readable[0].title[:40]}"


def main(argv: list[str]) -> int:
    if argv:
        specs = [SourceSpec("rss", f"candidate {i + 1}", "candidate", u, 1.0)
                 for i, u in enumerate(argv)]
    else:
        specs = list(DEFAULT_SOURCES)

    print(f"{'topic':12} {'name':26} {'robots':7} {'items':6} {'dated':6}  note")
    print("-" * 92)
    bad = 0
    for spec in specs:
        status, n, dated, note = check(spec)
        if status != "ok" or n == 0 or dated == 0:
            bad += 1
        print(f"{spec.topic:12} {spec.name[:26]:26} {status:7} {n:<6} {dated:<6} {note}")
    print()
    print(f"{len(specs) - bad}/{len(specs)} usable")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
