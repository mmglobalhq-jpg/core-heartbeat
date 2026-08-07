"""Where stories come from.

Three provider kinds behind one interface, so the pipeline does not care which
produced an item:

* ``rss``    — a feed. The polite, reliable path: feeds exist to be read by
               machines, so nothing about using one is adversarial.
* ``search`` — Gemini's Google Search grounding, reusing the tool built for the
               assistant. Good for "what happened today about X" where no single
               feed covers it.
* ``fetch``  — a single seed page, read through the SSRF-guarded fetcher.

ROBOTS AND REFUSALS
``robots.txt`` is honoured when ``BRIEFING_RESPECT_ROBOTS`` is on (the default),
and a disallow is recorded as ``skipped_reason`` rather than silently dropped —
a run must be able to say what it declined to read. Paywalls, login walls and
anti-bot challenges are treated the same way: detected, recorded, and left alone.
Nothing in here attempts to defeat an access control. If a site does not want to
be read by a program, the correct output is a note saying so.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
import urllib.robotparser
from collections import defaultdict
from typing import Protocol
from urllib.parse import urlsplit

from briefing import config
from briefing.models import RawItem, SourceSpec, tidy
from services.web import WebFetchError, fetch_page

logger = logging.getLogger(__name__)


class SourceProvider(Protocol):
    """Anything that can produce candidate stories for a topic."""

    kind: str

    def fetch(self, spec: SourceSpec) -> list[RawItem]: ...


# --- politeness --------------------------------------------------------------

_last_request_at: dict[str, float] = defaultdict(float)
_robots_cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}


def _host(url: str) -> str:
    return urlsplit(url).netloc.lower()


def throttle(url: str) -> None:
    """Space out requests to the same host. This job has no deadline."""
    host = _host(url)
    wait = config.PER_HOST_DELAY_S - (time.monotonic() - _last_request_at[host])
    if wait > 0:
        time.sleep(wait)
    _last_request_at[host] = time.monotonic()


def robots_allows(url: str) -> bool:
    """Does this host's robots.txt permit our user agent to fetch this URL?

    A robots.txt that cannot be retrieved is treated as permissive, which is the
    convention the standard describes — the alternative would make a single
    unreachable file silently empty the briefing.
    """
    if not config.RESPECT_ROBOTS:
        return True
    host = _host(url)
    if host not in _robots_cache:
        parts = urlsplit(url)
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(f"{parts.scheme}://{parts.netloc}/robots.txt")
        try:
            parser.read()
        except Exception as exc:  # noqa: BLE001 — unreachable robots.txt is not fatal
            logger.debug("robots.txt unreadable for %s (%s); treating as allowed", host, exc)
            parser = None
        _robots_cache[host] = parser
    parser = _robots_cache[host]
    return True if parser is None else parser.can_fetch(config.USER_AGENT, url)


# Phrases that mean "you are not the intended reader". Detecting these lets the
# run REPORT a paywall; it never triggers an attempt to get around one.
_BLOCKED_MARKERS = (
    ("paywall", ("subscribe to continue", "subscribers only", "this article is for subscribers",
                 "become a member to read", "unlock this article")),
    ("login_required", ("sign in to continue", "log in to read", "create a free account to")),
    ("anti_bot", ("enable javascript and cookies to continue", "verify you are human",
                  "checking your browser", "are you a robot", "access denied")),
)


def classify_block(text: str) -> str | None:
    """Name the access control a page is showing us, if any."""
    lowered = (text or "").lower()
    for reason, markers in _BLOCKED_MARKERS:
        if any(m in lowered for m in markers):
            return reason
    return None


# --- providers ---------------------------------------------------------------


class RssProvider:
    """Read an RSS/Atom feed.

    Feeds are the preferred source: publishing one is an explicit invitation to
    machine readers, and the entries carry a title, link and timestamp without
    any page scraping at all.
    """

    kind = "rss"

    def fetch(self, spec: SourceSpec) -> list[RawItem]:
        import feedparser

        if not spec.url:
            return []
        if not robots_allows(spec.url):
            return [_skipped(spec, spec.url, "robots_disallow")]
        throttle(spec.url)
        parsed = feedparser.parse(spec.url, agent=config.USER_AGENT)
        if getattr(parsed, "bozo", 0) and not parsed.entries:
            logger.warning("feed %s unreadable: %s", spec.url, getattr(parsed, "bozo_exception", ""))
            return [_skipped(spec, spec.url, "feed_unreadable")]

        items: list[RawItem] = []
        for entry in parsed.entries[: config.MAX_ITEMS_PER_SOURCE]:
            link = (entry.get("link") or "").strip()
            title = tidy(entry.get("title") or "")
            if not link or not title:
                continue
            items.append(
                RawItem(
                    url=link,
                    title=title,
                    source_name=spec.name,
                    topic=spec.topic,
                    summary=tidy(entry.get("summary") or "", limit=600) or None,
                    published_at=_entry_time(entry),
                )
            )
        return items


def _entry_time(entry) -> dt.datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        value = entry.get(key)
        if value:
            try:
                return dt.datetime(*value[:6], tzinfo=dt.UTC)
            except (TypeError, ValueError):
                continue
    return None


class FetchProvider:
    """Read one page through the SSRF-guarded fetcher.

    Reuses ``services.web.fetch_page``, which resolves the host before connecting,
    re-validates every redirect hop, refuses private and link-local addresses over
    both IPv4 and IPv6, and caps the response size. None of that is re-implemented
    here — a second, subtly different fetcher would be a second attack surface.
    """

    kind = "fetch"

    def fetch(self, spec: SourceSpec) -> list[RawItem]:
        if not spec.url:
            return []
        return [self.fetch_one(spec, spec.url)]

    def fetch_one(self, spec: SourceSpec, url: str, *, attempts: int = 2) -> RawItem:
        """Read one page, retrying once on a transient failure.

        The first production briefing could not read 4 of its 6 selected
        articles, and the write-ups fell back to headlines — visible in the
        output as a padded Deep Dive. The cause was not paywalls or robots.txt
        but plain TIMEOUTS: re-fetching the same URLs by hand succeeded.

        So one retry, and only for transport failures. A robots disallow or a
        paywall is a decision, not a hiccup — retrying those would be pestering a
        host that already said no.
        """
        if not robots_allows(url):
            return _skipped(spec, url, "robots_disallow")

        last_reason = "fetch_failed"
        for attempt in range(1, attempts + 1):
            throttle(url)
            try:
                final_url, title, text = fetch_page(url)
                break
            except WebFetchError as exc:
                # A refusal from the far end, not a transport hiccup — do not retry.
                last_reason = f"fetch_failed: {exc}"
                return _skipped(spec, url, last_reason)
            except Exception as exc:  # noqa: BLE001 — one bad page must not end the run
                last_reason = f"fetch_failed: {type(exc).__name__}"
                if attempt < attempts:
                    logger.info("retrying %s after %s", url, type(exc).__name__)
                    continue
                logger.warning("unexpected fetch failure for %s: %s", url, exc)
                return _skipped(spec, url, last_reason)

        blocked = classify_block(text)
        if blocked:
            # Recorded, not circumvented.
            return _skipped(spec, final_url, blocked, title=title)
        return RawItem(
            url=final_url,
            title=tidy(title) or url,
            source_name=spec.name,
            topic=spec.topic,
            body=text,
        )


class SearchProvider:
    """Ask the assistant's existing grounded-search tool what happened.

    Returns headline-level items with source URLs; bodies are filled in later by
    the fetch stage for whichever items survive ranking. Searching is cheap,
    fetching is not, so the order matters.
    """

    kind = "search"

    def fetch(self, spec: SourceSpec) -> list[RawItem]:
        from tools.web_tools import search_web_raw

        try:
            results = search_web_raw(spec.topic, max_results=config.MAX_ITEMS_PER_SOURCE)
        except Exception as exc:  # noqa: BLE001 — search is best-effort
            logger.warning("search failed for %r: %s", spec.topic, exc)
            return []
        items: list[RawItem] = []
        for result in results:
            url = (result.get("url") or "").strip()
            title = tidy(result.get("title") or "")
            if url and title:
                items.append(
                    RawItem(
                        url=url,
                        title=title,
                        source_name=result.get("source") or spec.name,
                        topic=spec.topic,
                        summary=tidy(result.get("snippet") or "", limit=600) or None,
                    )
                )
        return items


def _skipped(spec: SourceSpec, url: str, reason: str, *, title: str = "") -> RawItem:
    return RawItem(
        url=url,
        title=tidy(title) or f"[not read] {spec.name}",
        source_name=spec.name,
        topic=spec.topic,
        skipped_reason=reason,
    )


PROVIDERS: dict[str, SourceProvider] = {
    "rss": RssProvider(),
    "fetch": FetchProvider(),
    "search": SearchProvider(),
}


def provider_for(spec: SourceSpec) -> SourceProvider:
    try:
        return PROVIDERS[spec.kind]
    except KeyError:
        raise ValueError(f"unknown source kind: {spec.kind!r}") from None


# --- default registry --------------------------------------------------------
# Feeds only. Every one of these is a publisher-operated feed intended for
# machine consumption, on a general-interest topic, with no access control.

TOPIC_SOURCE_WEIGHT = 1.25
"""Topic results outrank the general feeds slightly.

A user who typed a topic asked for it. Without a nudge, five general feeds
producing ~76 items drown a search returning ~10, and the briefing looks
identical to one with no topics set. Not so high that a topic can take the whole
list — MAX_PER_SOURCE still caps any single source."""


DEFAULT_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec("rss", "NPR News", "top stories",
               "https://feeds.npr.org/1001/rss.xml", weight=1.1),
    SourceSpec("rss", "BBC World", "world",
               "https://feeds.bbci.co.uk/news/world/rss.xml", weight=1.1),
    SourceSpec("rss", "Reuters via Google News", "business",
               "https://news.google.com/rss/search?q=business+when:1d&hl=en-US&gl=US&ceid=US:en"),
    SourceSpec("rss", "Ars Technica", "technology",
               "https://feeds.arstechnica.com/arstechnica/index", weight=1.0),
    SourceSpec("rss", "Hacker News front page", "technology",
               "https://hnrss.org/frontpage", weight=0.9),
)


def sources_for(topics: list[str] | None) -> tuple[SourceSpec, ...]:
    """The sources to poll for one user.

    The default feeds ALWAYS run. A briefing built only from a user's topics
    would silently omit the day's major news, and someone who typed "sailing"
    should still hear that the government fell.

    Each topic becomes one grounded search. Deduplicated case-insensitively so
    "AI" and "ai" do not double-weight the same subject.
    """
    specs = list(DEFAULT_SOURCES)
    seen: set[str] = set()
    for raw in topics or []:
        topic = (raw or "").strip()
        key = topic.lower()
        if not topic or key in seen:
            continue
        seen.add(key)
        specs.append(
            SourceSpec("search", f"Topic: {topic}", topic, weight=TOPIC_SOURCE_WEIGHT)
        )
    return tuple(specs)
