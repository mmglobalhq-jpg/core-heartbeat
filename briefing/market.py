"""Market data for the briefing's opening section.

TWO SOURCES, DELIBERATELY
- **Equity indices** come from ``yfinance`` (Yahoo). Daily closes are read from
  ``history()`` rather than ``fast_info.lastPrice``: during market hours
  ``lastPrice`` is a LIVE quote, and labelling a live number as "the close" is the
  kind of error nobody catches because it looks right. ``history()`` returns
  settled daily bars at any hour.
- **Treasury yields** come from FRED's constant-maturity series (DGS2/5/10/30),
  the Fed's own H.15 release, through the API key this platform already holds for
  the REIT research engine.

THEY DO NOT SHARE AN AS-OF DATE, AND THAT IS NOT A BUG
H.15 publishes a day behind the equity close, so a morning briefing shows
equities for the previous session and yields for the one before it. Each block
carries its own date rather than one shared heading, because a single date over
two differently-dated blocks is a quiet lie about one of them.

NOTHING HERE MAY FAIL A BRIEFING
Every entry point returns partial data or ``None``. A market feed being down is
not a reason for the reader to get no briefing, and an exception escaping into
``build_briefing`` would do exactly that. Failures are recorded in the snapshot's
``errors`` so the run reports them instead of silently rendering a gap.

ON YAHOO
yfinance is not affiliated with Yahoo and states its data is intended for
personal use. This is a single-user personal briefing, which is that use — but it
is a departure from the "documented, robots-permitted sources only" rule the news
pipeline follows (doc 30 §5), and it is recorded here rather than left implicit.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# --- configuration -----------------------------------------------------------

ENABLED = os.environ.get("BRIEFING_MARKET", "1") not in ("0", "false", "False", "")

# (yahoo symbol, display label). Order is the display order.
INDICES: tuple[tuple[str, str], ...] = (
    ("^GSPC", "S&P 500"),
    ("^IXIC", "Nasdaq"),
    ("^DJI", "Dow"),
)

# (FRED series, display label). DGS30 is carried because the REIT engine already
# pulls it and a curve without the long end reads oddly next to a 2s10s spread.
TENORS: tuple[tuple[str, str], ...] = (
    ("DGS2", "2y"),
    ("DGS5", "5y"),
    ("DGS10", "10y"),
    ("DGS30", "30y"),
)

# Spreads to report, as (short leg, long leg). Rendered in basis points.
SPREADS: tuple[tuple[str, str], ...] = (("2y", "10y"), ("5y", "10y"))

FRED_BASE = os.environ.get("FRED_BASE_URL", "https://api.stlouisfed.org/fred")
FRED_KEY_ENV = "FRED_API_KEY"
FETCH_TIMEOUT_S = float(os.environ.get("BRIEFING_MARKET_TIMEOUT_S", "20"))
# How far back to ask for. Needs to clear a long weekend plus a holiday at each
# end; asking for more costs nothing and protects the "previous close" lookup.
LOOKBACK_DAYS = 21


# --- shapes ------------------------------------------------------------------


@dataclass(frozen=True)
class IndexQuote:
    label: str
    close: float
    change: float
    pct_change: float
    as_of: dt.date
    prev_as_of: dt.date


@dataclass(frozen=True)
class YieldPoint:
    label: str
    yield_pct: float
    change_bps: float
    as_of: dt.date
    prev_as_of: dt.date


@dataclass(frozen=True)
class Spread:
    label: str        # e.g. "2s10s"
    value_bps: float
    as_of: dt.date


@dataclass
class MarketSnapshot:
    indices: list[IndexQuote] = field(default_factory=list)
    yields: list[YieldPoint] = field(default_factory=list)
    spreads: list[Spread] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def has_data(self) -> bool:
        return bool(self.indices or self.yields)

    def meta(self) -> dict:
        """Compact record for run_meta — what was fetched and what failed."""
        return {
            "indices": len(self.indices),
            "yields": len(self.yields),
            "spreads": len(self.spreads),
            "errors": list(self.errors),
            "equity_as_of": self.indices[0].as_of.isoformat() if self.indices else None,
            "rates_as_of": self.yields[0].as_of.isoformat() if self.yields else None,
        }


# --- equities ----------------------------------------------------------------


def fetch_indices() -> tuple[list[IndexQuote], list[str]]:
    """Daily closes for the configured indices. Never raises."""
    quotes: list[IndexQuote] = []
    errors: list[str] = []
    try:
        import yfinance as yf
    except Exception as exc:
        return [], [f"yfinance unavailable: {type(exc).__name__}"]

    for symbol, label in INDICES:
        try:
            hist = yf.Ticker(symbol).history(period=f"{LOOKBACK_DAYS}d", interval="1d")
            closes = hist["Close"].dropna()
            if len(closes) < 2:
                errors.append(f"{label}: fewer than two closes returned")
                continue
            last, prev = float(closes.iloc[-1]), float(closes.iloc[-2])
            if prev == 0:
                errors.append(f"{label}: previous close was zero")
                continue
            quotes.append(
                IndexQuote(
                    label=label,
                    close=last,
                    change=last - prev,
                    pct_change=(last / prev - 1.0) * 100.0,
                    as_of=closes.index[-1].date(),
                    prev_as_of=closes.index[-2].date(),
                )
            )
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}")
            logger.warning("index fetch failed for %s: %s", symbol, exc)
    return quotes, errors


# --- treasuries --------------------------------------------------------------


def _fred_observations(series_id: str, key: str) -> list[tuple[dt.date, float]]:
    """Most-recent-first observations, missing values dropped."""
    import json
    import urllib.parse
    import urllib.request

    start = (dt.date.today() - dt.timedelta(days=LOOKBACK_DAYS)).isoformat()
    query = urllib.parse.urlencode({
        "series_id": series_id, "api_key": key, "file_type": "json",
        "observation_start": start, "sort_order": "desc", "limit": "10",
    })
    with urllib.request.urlopen(f"{FRED_BASE}/series/observations?{query}",
                                timeout=FETCH_TIMEOUT_S) as resp:
        payload = json.load(resp)
    out: list[tuple[dt.date, float]] = []
    for row in payload.get("observations", []):
        # FRED marks a missing observation with "." — a holiday, not a zero.
        if row.get("value") in (".", "", "NA", "NaN", None):
            continue
        try:
            out.append((dt.date.fromisoformat(row["date"]), float(row["value"])))
        except (ValueError, KeyError):
            continue
    return out


def fetch_yields() -> tuple[list[YieldPoint], list[Spread], list[str]]:
    """Constant-maturity yields and the configured spreads. Never raises."""
    # Through services.secrets, so a mounted /run/secrets file is preferred over
    # an environment variable — the same contract the Supabase keys use. Falls
    # back to the plain env var, which is what the dev runner and the tests use.
    try:
        from services.secrets import secret as read_secret

        key = (read_secret(FRED_KEY_ENV) or "").strip()
    except Exception:
        key = (os.environ.get(FRED_KEY_ENV) or "").strip()
    if not key:
        return [], [], ["FRED_API_KEY is not set"]

    points: dict[str, YieldPoint] = {}
    errors: list[str] = []
    for series_id, label in TENORS:
        try:
            obs = _fred_observations(series_id, key)
            if len(obs) < 2:
                errors.append(f"{label}: fewer than two observations")
                continue
            (d0, v0), (d1, v1) = obs[0], obs[1]
            points[label] = YieldPoint(
                label=label,
                yield_pct=v0,
                change_bps=(v0 - v1) * 100.0,
                as_of=d0,
                prev_as_of=d1,
            )
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}")
            logger.warning("FRED fetch failed for %s: %s", series_id, exc)

    spreads: list[Spread] = []
    for short, long in SPREADS:
        a, b = points.get(short), points.get(long)
        if not a or not b:
            continue
        # Both legs must be the same session, or the "spread" is a difference
        # between two different days and means nothing.
        if a.as_of != b.as_of:
            errors.append(f"{short}s{long}s: legs differ in date ({a.as_of} vs {b.as_of})")
            continue
        spreads.append(Spread(
            label=f"{short.rstrip('y')}s{long.rstrip('y')}s",
            value_bps=(b.yield_pct - a.yield_pct) * 100.0,
            as_of=a.as_of,
        ))

    ordered = [points[label] for _, label in TENORS if label in points]
    return ordered, spreads, errors


# --- entry point -------------------------------------------------------------


def market_snapshot() -> MarketSnapshot | None:
    """Everything for the data section, or None when the feature is off.

    Returns a snapshot even when both sources fail — an empty snapshot with
    populated ``errors`` is how the run reports that it tried, which is more
    useful than a None that cannot be told apart from "disabled".
    """
    if not ENABLED:
        return None
    indices, eq_errors = fetch_indices()
    yields, spreads, rate_errors = fetch_yields()
    return MarketSnapshot(
        indices=indices,
        yields=yields,
        spreads=spreads,
        errors=[*eq_errors, *rate_errors],
    )
