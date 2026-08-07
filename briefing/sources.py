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
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
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
        _robots_cache[host] = _load_robots(url)
    parser = _robots_cache[host]
    return True if parser is None else parser.can_fetch(config.USER_AGENT, url)


def _load_robots(url: str) -> urllib.robotparser.RobotFileParser | None:
    """Fetch and parse robots.txt using OUR user agent.

    ``RobotFileParser.read()`` fetches with ``Python-urllib/x.y``, and a great
    many sites answer that with 403 as basic bot defence. The parser then treats
    401/403 as *disallow everything* — so a site that merely dislikes the default
    agent was recorded as forbidding all access.

    That is not hypothetical. ft.com returns 403 to ``Python-urllib`` and 200 to a
    real agent string; its rules allow ``/rss/`` and disallow only ``/login``,
    ``/search``, ``/myft`` and similar. This pipeline was refusing FT's public
    feed on the strength of a robots.txt it had never actually read.

    A 401/403 received while identifying ourselves honestly still means
    disallow-all, which is the convention and is deliberately preserved.
    """
    parts = urlsplit(url)
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(robots_url)
    request = urllib.request.Request(robots_url, headers={"User-Agent": config.USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            body = response.read(512_000).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            # Refused even when asking honestly — treat as off-limits.
            parser.disallow_all = True
            logger.info("robots.txt for %s returned %s; treating host as disallowed",
                        parts.netloc, exc.code)
            return parser
        # 404 and friends: no robots.txt means no restriction.
        logger.debug("no robots.txt for %s (HTTP %s); treating as allowed",
                     parts.netloc, exc.code)
        return None
    except Exception as exc:  # noqa: BLE001 — unreachable robots.txt is not fatal
        logger.debug("robots.txt unreadable for %s (%s); treating as allowed",
                     parts.netloc, exc)
        return None
    parser.parse(body.splitlines())
    return parser


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

    TITLES MUST BE RESOLVED BY FETCHING. Gemini's grounding metadata returns the
    SITE for ``web.title`` — "businesswire.com", not the article's headline. Used
    as-is that produces a Top 5 entry called "reuters.com", and worse, every
    search result looks near-identical to the clusterer because domain names
    share no distinctive tokens.

    So unlike the feed providers, this one fetches each result at discovery time
    to get a real title. That trades the "rank before fetch" saving for
    correctness — but only over the handful of results a topic returns, not the
    ~76 items the feeds produce, so the cost is bounded and opt-in.
    """

    kind = "search"

    def fetch(self, spec: SourceSpec) -> list[RawItem]:
        from tools.web_tools import search_web_raw

        try:
            results = search_web_raw(spec.topic, max_results=config.MAX_ITEMS_PER_SOURCE)
        except Exception as exc:  # noqa: BLE001 — search is best-effort
            logger.warning("search failed for %r: %s", spec.topic, exc)
            return []

        candidates = [
            (r.get("url", "").strip(), tidy(r.get("source") or r.get("title") or ""))
            for r in results
            if (r.get("url") or "").strip()
        ]
        if not candidates:
            return []

        fetcher = FetchProvider()

        def resolve(pair: tuple[str, str]) -> RawItem | None:
            url, site = pair
            item = fetcher.fetch_one(spec, url)
            if item.skipped_reason or not item.title.strip():
                # No usable headline and no text. Including it would put a bare
                # domain in front of the reader.
                logger.info("dropping search result with no resolvable title: %s (%s)",
                            url, item.skipped_reason or "empty title")
                return None
            item.source_name = site or spec.name
            item.topic = spec.topic
            return item

        workers = max(1, min(config.FETCH_CONCURRENCY, len(candidates)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            resolved = list(pool.map(resolve, candidates))
        return [i for i in resolved if i is not None]


def _skipped(spec: SourceSpec, url: str, reason: str, *, title: str = "") -> RawItem:
    return RawItem(
        url=url,
        title=tidy(title) or f"[not read] {spec.name}",
        source_name=spec.name,
        topic=spec.topic,
        skipped_reason=reason,
    )



class SiteProvider:
    """Read headlines off a public page that has no feed.

    The fallback of last resort, used only when discovery found no feed at all.
    It reads ONE page — the section or homepage the user pointed at — extracts
    same-host links whose anchor text is long enough to be a headline, and stops.
    It does not crawl onward, render JavaScript, or vary its identity.

    Timestamps are unavailable this way, so items carry no ``published_at`` and
    survive the freshness filter on the strength of the page being current. That
    is a real weakness of scraping versus a feed, and it is why every other route
    is tried first.
    """

    kind = "site"

    def fetch(self, spec: SourceSpec) -> list[RawItem]:
        from briefing.discovery import extract_headline_links, _raw_html

        if not spec.url:
            return []
        if not robots_allows(spec.url):
            return [_skipped(spec, spec.url, "robots_disallow")]
        throttle(spec.url)
        html = _raw_html(spec.url)
        if not html:
            return [_skipped(spec, spec.url, "fetch_failed")]

        blocked = classify_block(html)
        if blocked:
            return [_skipped(spec, spec.url, blocked)]

        items: list[RawItem] = []
        for url, headline in extract_headline_links(html, spec.url)[: config.MAX_ITEMS_PER_SOURCE]:
            items.append(
                RawItem(
                    url=url,
                    title=headline,
                    source_name=spec.name,
                    topic=spec.topic,
                )
            )
        if not items:
            return [_skipped(spec, spec.url, "no_headlines_found")]
        return items


PROVIDERS: dict[str, SourceProvider] = {
    "rss": RssProvider(),
    "fetch": FetchProvider(),
    "search": SearchProvider(),
    "site": SiteProvider(),
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


# Every feed below was verified on 2026-08-07: robots.txt permits it under our
# user agent, it parses, and it carried dated, recent entries. Two candidates were
# dropped for deliberate refusals, not for being awkward:
#
#   AP News       apnews.com/robots.txt has `Disallow: /*.rss`
#   MarketWatch   robots.txt returns 403 even to an honest agent, on both hosts
#
# Re-check with tests/check_feeds.py rather than assuming; feeds rot quietly.
DEFAULT_SOURCES: tuple[SourceSpec, ...] = (
    # --- world & general ---
    SourceSpec("rss", "BBC News", "world",
               "http://feeds.bbci.co.uk/news/rss.xml", weight=1.1),
    SourceSpec("rss", "NPR Top Stories", "top stories",
               "https://feeds.npr.org/1001/rss.xml", weight=1.1),
    SourceSpec("rss", "Guardian World", "world",
               "https://www.theguardian.com/world/rss", weight=1.05),
    # --- business & economy ---
    SourceSpec("rss", "FT Home", "business",
               "https://www.ft.com/rss/home", weight=1.1),
    SourceSpec("rss", "CNBC Top News", "business",
               "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    SourceSpec("rss", "BBC Business", "business",
               "https://feeds.bbci.co.uk/news/business/rss.xml"),
    SourceSpec("rss", "NPR Business", "business",
               "https://feeds.npr.org/1006/rss.xml"),
    # The Economist's feed returns ~300 entries spanning weeks and is
    # malformed enough that feedparser flags bozo (it still parses). Low weight:
    # useful analysis, but it would otherwise flood the pool on volume alone.
    SourceSpec("rss", "The Economist", "business",
               "https://www.economist.com/latest/rss.xml", weight=0.85),
    # --- technology ---
    SourceSpec("rss", "Ars Technica", "technology",
               "https://feeds.arstechnica.com/arstechnica/index"),
    SourceSpec("rss", "The Verge", "technology",
               "https://www.theverge.com/rss/index.xml"),
    SourceSpec("rss", "Hacker News front page", "technology",
               "https://news.ycombinator.com/rss", weight=0.9),
    # --- science ---
    SourceSpec("rss", "ScienceDaily", "science",
               "https://www.sciencedaily.com/rss/all.xml", weight=0.9),
    SourceSpec("rss", "NASA Breaking News", "science",
               "https://www.nasa.gov/rss/dyn/breaking_news.rss", weight=0.9),
)


def sources_for(topics: list[str] | None = None,
                user_sources: list[dict] | None = None) -> tuple[SourceSpec, ...]:
    """The sources to poll. Currently the defaults, whatever the topics.

    TOPICS CANNOT ADD SOURCES, and it is worth recording why rather than leaving
    the next person to rediscover it. Two routes were tried against production
    and both are closed:

    * **Gemini grounded search** returns
      ``vertexaisearch.cloud.google.com/grounding-api-redirect/...`` wrappers,
      not article URLs, and that host's robots.txt disallows automated fetching.
    * **Google News RSS search** would give real headlines and real publisher
      URLs, but ``news.google.com/robots.txt`` is ``Disallow: /`` with an
      allow-list that excludes ``/rss/``.

    Bing and Yahoo news search feeds return no entries for our agent. No
    general-purpose news search is available to us on terms we are willing to
    accept, and overriding a robots directive to make a feature work is not
    something this module does.

    So topics steer RANKING instead — see ``dedup.topic_boost``. That works only
    over what the default feeds already carry: a topic no feed covers still
    yields nothing, and the honest fix for that is letting users add their own
    feeds, which is not built yet.
    """
    specs = list(DEFAULT_SOURCES)
    for row in user_sources or []:
        url = (row.get("url") or "").strip()
        kind = row.get("kind") or "rss"
        if not url or kind not in PROVIDERS:
            continue
        specs.append(
            SourceSpec(
                kind,
                row.get("name") or _host(url),
                row.get("topic") or "custom",
                url,
                # Slightly above the defaults: someone who went to the trouble of
                # adding a source wants to see it. Not high enough to let one
                # feed take the list — MAX_PER_SOURCE still applies.
                weight=float(row.get("weight") or 1.2),
            )
        )
    return tuple(specs)
