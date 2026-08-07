"""Internal data shapes for the briefing pipeline.

Plain dataclasses rather than Pydantic models: nothing here is parsed from an
untrusted wire format. Model *output* is validated where it is parsed, in
``compose`` and ``editorial``.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit


@dataclass(frozen=True)
class SourceSpec:
    """A place to look for stories."""

    kind: str  # 'rss' | 'search' | 'fetch'
    name: str
    topic: str
    url: str | None = None
    weight: float = 1.0


# Tracking parameters carry no meaning and defeat deduplication: the same article
# arrives from two feeds with different utm_source and looks like two stories.
_TRACKING_PREFIXES = ("utm_", "mc_", "pk_")
_TRACKING_KEYS = {"fbclid", "gclid", "igshid", "ref", "ref_src", "cmpid", "smid"}


def normalize_url(url: str) -> str:
    """Canonical form used for the dedup key.

    Lowercases the host, drops the fragment, strips tracking parameters and
    removes a trailing slash. Deliberately does NOT drop meaningful query
    parameters — plenty of sites still identify articles with ``?id=``, and
    collapsing those would merge unrelated stories into one.
    """
    parts = urlsplit(url.strip())
    query = "&".join(
        p
        for p in parts.query.split("&")
        if p
        and (key := p.split("=", 1)[0].lower()) not in _TRACKING_KEYS
        and not key.startswith(_TRACKING_PREFIXES)
    )
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, query, ""))


def url_hash(url: str) -> str:
    return hashlib.sha256(normalize_url(url).encode("utf-8")).hexdigest()


@dataclass
class RawItem:
    """One story as ingested, before any model has seen it.

    ``body`` is UNTRUSTED DATA. It came off the public web and may contain text
    engineered to look like instructions. Never interpolate it into a prompt
    without going through ``untrusted.fence``.
    """

    url: str
    title: str
    source_name: str
    topic: str
    summary: str | None = None
    body: str | None = None
    published_at: dt.datetime | None = None
    fetched_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.UTC))
    # Set when ingestion declined to read the item — robots.txt disallow, a
    # paywall, a login wall, an anti-bot challenge. The item is kept rather than
    # dropped so the run can report what it did not read.
    skipped_reason: str | None = None

    @property
    def hash(self) -> str:
        return url_hash(self.url)

    @property
    def readable(self) -> bool:
        return self.skipped_reason is None and bool(self.title.strip())

    def age_hours(self, now: dt.datetime | None = None) -> float:
        if self.published_at is None:
            return 0.0
        now = now or dt.datetime.now(dt.UTC)
        published = self.published_at
        if published.tzinfo is None:
            published = published.replace(tzinfo=dt.UTC)
        return max(0.0, (now - published).total_seconds() / 3600.0)


@dataclass
class ScoredItem:
    item: RawItem
    score: float
    cluster_id: str
    # Other items judged to be the same story. Kept so the write-up can say
    # "reported by N outlets", and so the sources are auditable.
    duplicates: list[RawItem] = field(default_factory=list)


@dataclass
class Section:
    """One rendered piece of a briefing."""

    kind: str  # 'top' | 'deep_dive'
    rank: int
    headline: str
    body: str
    url: str
    source_name: str | None = None
    published_at: dt.datetime | None = None


@dataclass
class BriefingDraft:
    user_id: str
    briefing_date: dt.date
    sections: list[Section] = field(default_factory=list)
    run_meta: dict = field(default_factory=dict)

    @property
    def top(self) -> list[Section]:
        return sorted((s for s in self.sections if s.kind == "top"), key=lambda s: s.rank)

    @property
    def deep_dive(self) -> Section | None:
        return next((s for s in self.sections if s.kind == "deep_dive"), None)


_WHITESPACE = re.compile(r"\s+")


def tidy(text: str, limit: int | None = None) -> str:
    """Collapse whitespace and optionally truncate on a word boundary."""
    cleaned = _WHITESPACE.sub(" ", (text or "")).strip()
    if limit is None or len(cleaned) <= limit:
        return cleaned
    cut = cleaned[:limit].rsplit(" ", 1)[0]
    return (cut or cleaned[:limit]).rstrip() + "…"
