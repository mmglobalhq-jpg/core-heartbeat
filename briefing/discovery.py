"""Turn a URL a user pasted into a usable source — or an honest refusal.

THE ORDER MATTERS, and it is deliberately least-invasive first:

1. **Is it already a feed?** Parse it. Done.
2. **Does the page advertise a feed?** ``<link rel="alternate">`` is a publisher
   telling machine readers where to go. Following that is the polite path and it
   works for most news sites — pasting ``https://www.ft.com/`` should just work.
3. **Do the conventional feed paths exist?** ``/rss``, ``/feed``, ``/rss.xml``…
4. **Only then, read the page itself** and extract headline links.

Step 4 is the only step that is "scraping", and it exists because some sites
genuinely have no feed. It reads one public HTML page that robots.txt permits,
with our real user agent, at the same politeness delay as everything else. It
does not render JavaScript, does not rotate identities, and does not touch
anything behind a login or a paywall.

WHAT THIS WILL NOT DO
If robots.txt says no, discovery returns a refusal and the source is not added.
If the page is a paywall or an anti-bot challenge, same. The user gets told
which, in plain words, rather than the feature quietly half-working.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit

from briefing import config
from briefing.models import tidy
from briefing.sources import classify_block, robots_allows, throttle
from services.web import WebFetchError, fetch_page

logger = logging.getLogger(__name__)

# Tried in order when a page advertises nothing. Ordinary conventions, not
# guesses at private endpoints.
COMMON_FEED_PATHS = (
    "/rss", "/feed", "/rss.xml", "/feed.xml", "/atom.xml",
    "/index.xml", "/rss/index.xml", "/feeds/all.atom.xml",
    # ft.com serves /rss/home while returning 403 for its own homepage HTML.
    # Conventional enough to be worth trying, and it is the difference between
    # "FT works" and "FT cannot be added".
    "/rss/home", "/news/rss", "/feeds/rss",
)

MIN_HEADLINE_CHARS = 28
"""Anchor text shorter than this is nearly always navigation — "Markets",
"Sign in", "More". Real headlines are sentences."""

MAX_DISCOVERED_LINKS = 40


@dataclass
class SourceCandidate:
    """What we found, or why we did not."""

    ok: bool
    kind: str = ""          # 'rss' | 'site'
    url: str = ""
    name: str = ""
    reason: str = ""        # populated when ok is False
    sample: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok, "kind": self.kind, "url": self.url,
            "name": self.name, "reason": self.reason, "sample": self.sample[:5],
        }


def _feed_entries(url: str) -> tuple[int, str, list[str]]:
    """``(entry_count, feed_title, sample_headlines)`` for a candidate feed URL."""
    import feedparser

    parsed = feedparser.parse(url, agent=config.USER_AGENT)
    entries = getattr(parsed, "entries", []) or []
    titles = [tidy(e.get("title") or "") for e in entries[:5]]
    feed_title = tidy((getattr(parsed, "feed", {}) or {}).get("title") or "")
    return len(entries), feed_title, [t for t in titles if t]


def _site_name(url: str) -> str:
    return urlsplit(url).netloc.lower().removeprefix("www.")


_FEED_LINK_RE = re.compile(
    r"""<link[^>]+?(?:rel=["']?alternate["']?)[^>]*?>""", re.I | re.S)
_HREF_RE = re.compile(r"""href=["']([^"']+)["']""", re.I)
_TYPE_RE = re.compile(r"""type=["']([^"']+)["']""", re.I)


def advertised_feeds(html: str, base_url: str) -> list[str]:
    """Feed URLs the page itself points at, via ``<link rel="alternate">``."""
    found: list[str] = []
    for tag in _FEED_LINK_RE.findall(html or ""):
        type_match = _TYPE_RE.search(tag)
        if not type_match or "xml" not in type_match.group(1).lower():
            continue
        href_match = _HREF_RE.search(tag)
        if href_match:
            absolute = urljoin(base_url, href_match.group(1))
            if absolute not in found:
                found.append(absolute)
    return found


_ANCHOR_RE = re.compile(r"""<a[^>]+href=["']([^"']+)["'][^>]*>(.*?)</a>""", re.I | re.S)
_TAG_RE = re.compile(r"<[^>]+>")


def extract_headline_links(html: str, base_url: str) -> list[tuple[str, str]]:
    """``(url, headline)`` pairs that look like articles on a section page.

    Heuristic and deliberately conservative: same host, link text long enough to
    be a sentence, deduplicated by URL. A site with no feed gets a usable list;
    a site whose markup defeats this gets rejected at validation rather than
    silently contributing navigation links to someone's briefing.
    """
    host = urlsplit(base_url).netloc.lower()
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for href, inner in _ANCHOR_RE.findall(html or ""):
        text = tidy(_TAG_RE.sub(" ", inner))
        if len(text) < MIN_HEADLINE_CHARS:
            continue
        absolute = urljoin(base_url, href)
        parts = urlsplit(absolute)
        if parts.scheme not in ("http", "https") or parts.netloc.lower() != host:
            continue
        key = absolute.split("#")[0]
        if key in seen:
            continue
        seen.add(key)
        out.append((key, text))
        if len(out) >= MAX_DISCOVERED_LINKS:
            break
    return out


def discover(url: str) -> SourceCandidate:
    """Work out what the user just pasted. Never raises."""
    url = (url or "").strip()
    if not url:
        return SourceCandidate(False, reason="No URL given.")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return SourceCandidate(False, reason="That does not look like a web address.")

    if not robots_allows(url):
        return SourceCandidate(
            False,
            reason=f"{parts.netloc} does not allow automated access to that address "
                   "(robots.txt). It cannot be added.",
        )

    # 1. Already a feed?
    try:
        count, title, sample = _feed_entries(url)
    except Exception:  # noqa: BLE001
        count, title, sample = 0, "", []
    if count:
        return SourceCandidate(True, "rss", url, title or _site_name(url), sample=sample)

    # 2. Conventional feed paths, BEFORE touching the page.
    #
    # Ordered this way because a site can serve its feed happily while refusing
    # its own homepage to non-browser agents — ft.com returns 200 for
    # /rss/home and 403 for /. Reading the HTML first made FT undiscoverable
    # for a reason that had nothing to do with whether FT publishes a feed.
    origin = f"{parts.scheme}://{parts.netloc}"
    for path in COMMON_FEED_PATHS:
        candidate = urljoin(origin, path)
        if not robots_allows(candidate):
            continue
        try:
            count, title, sample = _feed_entries(candidate)
        except Exception:  # noqa: BLE001
            continue
        if count:
            return SourceCandidate(True, "rss", candidate,
                                   title or _site_name(url), sample=sample)

    # 3. Read the page — for feeds it advertises, and for the headline fallback.
    throttle(url)
    try:
        final_url, page_title, text = fetch_page(url)
    except WebFetchError as exc:
        return SourceCandidate(False, reason=f"Could not read that page: {exc}")
    except Exception as exc:  # noqa: BLE001
        logger.info("discovery fetch failed for %s: %s", url, exc)
        return SourceCandidate(False, reason="Could not read that page.")

    blocked = classify_block(text)
    if blocked:
        pretty = {"paywall": "sits behind a paywall",
                  "login_required": "requires signing in",
                  "anti_bot": "blocks automated readers"}.get(blocked, blocked)
        return SourceCandidate(
            False,
            reason=f"That page {pretty}, so it cannot be read automatically. "
                   "If the site publishes an RSS feed, add that instead.",
        )

    # `fetch_page` returns extracted text, not markup, so re-read the raw HTML
    # for link discovery. Same guarded fetcher, one extra request.
    html = _raw_html(final_url)

    # 4. Feeds the page advertises.
    for candidate in advertised_feeds(html, final_url):
        if not robots_allows(candidate):
            continue
        try:
            count, title, sample = _feed_entries(candidate)
        except Exception:  # noqa: BLE001
            continue
        if count:
            return SourceCandidate(True, "rss", candidate,
                                   title or _site_name(url), sample=sample)

    # 5. No feed anywhere — fall back to reading the page for headlines.
    links = extract_headline_links(html, final_url)
    if len(links) >= 3:
        return SourceCandidate(
            True, "site", final_url, tidy(page_title) or _site_name(url),
            sample=[text for _, text in links[:5]],
        )

    return SourceCandidate(
        False,
        reason="No feed found, and no headlines could be read from that page. "
               "Try linking directly to the site's news or RSS page.",
    )


def _raw_html(url: str) -> str:
    """Fetch raw markup through the same SSRF-guarded path.

    Uses services.web's validation rather than a second, subtly different
    fetcher — a parallel implementation would be a parallel attack surface.
    """
    import httpx

    from services.web import MAX_BYTES, REQUEST_TIMEOUT_S, _validate

    try:
        safe_url = _validate(url)
        with httpx.Client(timeout=REQUEST_TIMEOUT_S, follow_redirects=False) as client:
            response = client.get(safe_url, headers={"User-Agent": config.USER_AGENT})
        if response.status_code >= 400:
            return ""
        return response.content[:MAX_BYTES].decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        logger.debug("raw html fetch failed for %s: %s", url, exc)
        return ""
