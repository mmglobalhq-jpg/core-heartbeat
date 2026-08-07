"""Collect candidate stories from the configured sources.

Two stages, deliberately separated:

1. **Discover** — ask every source what exists. Cheap: feeds and search return
   titles and links without fetching articles.
2. **Enrich** — fetch full text for the handful of items that survive ranking.

Doing it the other way round — fetching everything, then ranking — would mean
several hundred HTTP requests to produce five paragraphs. Ranking on titles and
summaries first cuts that to the number of stories actually used.
"""

from __future__ import annotations

import datetime as dt
import logging
from concurrent.futures import ThreadPoolExecutor

from briefing import config
from briefing.models import RawItem, SourceSpec
from briefing.sources import FetchProvider, provider_for
from briefing.untrusted import detect_injection

logger = logging.getLogger(__name__)


def discover(specs: list[SourceSpec]) -> tuple[list[RawItem], dict]:
    """Ask every source for candidates. Returns ``(items, stats)``.

    Sources are polled concurrently but each failure is contained: a dead feed
    produces zero items and a log line, never an exception that ends the run. A
    briefing assembled from four of five sources is a briefing; an exception is
    not.
    """
    items: list[RawItem] = []
    stats = {"sources_total": len(specs), "sources_failed": 0, "sources_ok": 0}

    def one(spec: SourceSpec) -> list[RawItem]:
        try:
            return provider_for(spec).fetch(spec)
        except Exception as exc:  # noqa: BLE001
            logger.warning("source %s (%s) failed: %s", spec.name, spec.kind, exc)
            return []

    workers = max(1, min(config.FETCH_CONCURRENCY, len(specs) or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for spec, produced in zip(specs, pool.map(one, specs)):
            items.extend(produced)
            # A source counts as OK only if it produced something USABLE. A list
            # holding one skip-marker is truthy, so `if produced:` reported a
            # fully blocked feed as healthy — every production run said
            # "sources_ok=5/5" while one source was contributing nothing at all.
            usable = sum(1 for i in produced if i.readable)
            if usable:
                stats["sources_ok"] += 1
            else:
                stats["sources_failed"] += 1
                reasons = {i.skipped_reason for i in produced if i.skipped_reason}
                logger.warning("source %s produced no usable items%s", spec.name,
                               f" ({', '.join(sorted(reasons))})" if reasons else "")

    fresh = _filter_fresh(items)
    stats["discovered"] = len(items)
    stats["fresh"] = len(fresh)
    stats["skipped"] = sum(1 for i in items if i.skipped_reason)
    return fresh[: config.MAX_TOTAL_ITEMS], stats


def _filter_fresh(items: list[RawItem]) -> list[RawItem]:
    """Drop stale stories and items ingestion could not read.

    An item with no timestamp is KEPT. Plenty of feeds omit one, and dropping
    them would quietly bias the briefing toward whichever publishers happen to
    populate the field.
    """
    now = dt.datetime.now(dt.UTC)
    kept: list[RawItem] = []
    for item in items:
        if not item.readable:
            continue
        if item.published_at is not None and item.age_hours(now) > config.ITEM_MAX_AGE_HOURS:
            continue
        kept.append(item)
    kept.sort(key=lambda i: (i.published_at or now), reverse=True)
    return kept


def enrich(items: list[RawItem]) -> tuple[list[RawItem], dict]:
    """Fetch full text for items that will actually be written up.

    Runs after ranking. Items whose fetch is refused keep their title and summary
    and carry a ``skipped_reason`` — the write-up then works from the summary
    alone rather than the story vanishing, and the run reports what it could not
    read.
    """
    fetcher = FetchProvider()
    stats = {"enriched": 0, "blocked": 0, "injection_signals": {}}

    def one(item: RawItem) -> RawItem:
        if item.body:
            return item
        spec = SourceSpec("fetch", item.source_name, item.topic, item.url)
        fetched = fetcher.fetch_one(spec, item.url)
        if fetched.skipped_reason:
            item.skipped_reason = fetched.skipped_reason
            return item
        item.body = fetched.body
        return item

    workers = max(1, min(config.FETCH_CONCURRENCY, len(items) or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        enriched = list(pool.map(one, items))

    for item in enriched:
        if item.skipped_reason:
            stats["blocked"] += 1
        elif item.body:
            stats["enriched"] += 1
        # Record what looked hostile without acting on it. See untrusted.py:
        # this is telemetry, not a gate.
        for signal in detect_injection(f"{item.title}\n{item.body or ''}"):
            stats["injection_signals"][signal] = stats["injection_signals"].get(signal, 0) + 1

    if stats["injection_signals"]:
        logger.info("injection-like patterns seen in fetched content: %s", stats["injection_signals"])
    return enriched, stats
