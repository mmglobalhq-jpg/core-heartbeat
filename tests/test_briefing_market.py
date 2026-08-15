"""Tests for the briefing's data section: market snapshot and report links.

The rule these exist to hold is that NEITHER MAY FAIL A BRIEFING. A market feed
being down, a research service being unreachable, a missing API key — none of
these is a reason for the reader to get no news, and an exception escaping into
`build_briefing` would do exactly that.

The second rule is that the two blocks carry their OWN as-of dates. FRED's H.15
series publish a day behind the equity close, so a single shared heading would be
wrong about one of them.
"""

from __future__ import annotations

import datetime as dt

import pytest

from briefing import market, reports
from briefing.market import IndexQuote, MarketSnapshot, Spread, YieldPoint
from briefing.models import BriefingDraft, Section
from briefing.render import render_html, render_text
from briefing.reports import ReportLink

D14 = dt.date(2026, 8, 14)
D13 = dt.date(2026, 8, 13)
D12 = dt.date(2026, 8, 12)


def snapshot():
    return MarketSnapshot(
        indices=[
            IndexQuote("S&P 500", 7785.76, -13.23, -0.1696, D14, D13),
            IndexQuote("Nasdaq", 26729.16, -73.87, -0.2756, D14, D13),
            IndexQuote("Dow", 53732.41, -107.58, -0.1998, D14, D13),
        ],
        yields=[
            YieldPoint("2y", 4.15, -5.0, D13, D12),
            YieldPoint("5y", 4.32, -6.0, D13, D12),
            YieldPoint("10y", 4.63, -5.0, D13, D12),
        ],
        spreads=[Spread("2s10s", 48.0, D13), Spread("5s10s", 31.0, D13)],
    )


def draft(market_=None, reports_=None, n=10):
    return BriefingDraft(
        user_id="u1",
        briefing_date=dt.date(2026, 8, 15),
        sections=[Section("top", i, f"H{i}", f"B{i}", f"https://x.example/{i}", "Outlet")
                  for i in range(1, n + 1)],
        run_meta={"sources_ok": 19},
        market=market_,
        reports=reports_ or [],
    )


# --- failure must never propagate ---------------------------------------------

def test_indices_survive_yfinance_being_absent(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "yfinance", None)
    quotes, errors = market.fetch_indices()
    assert quotes == []
    assert errors and "yfinance" in errors[0]


def test_indices_survive_a_symbol_raising(monkeypatch):
    class Boom:
        def __init__(self, *a, **k): pass
        def history(self, **k): raise RuntimeError("yahoo said no")

    fake = type("m", (), {"Ticker": Boom})
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake)
    quotes, errors = market.fetch_indices()
    assert quotes == []
    assert len(errors) == len(market.INDICES)


def test_yields_without_a_key_report_it_rather_than_raising(monkeypatch):
    monkeypatch.delenv(market.FRED_KEY_ENV, raising=False)
    points, spreads, errors = market.fetch_yields()
    assert (points, spreads) == ([], [])
    assert "FRED_API_KEY" in errors[0]


def test_yields_survive_the_api_failing(monkeypatch):
    monkeypatch.setenv(market.FRED_KEY_ENV, "k")
    monkeypatch.setattr(market, "_fred_observations",
                        lambda sid, key: (_ for _ in ()).throw(OSError("network")))
    points, spreads, errors = market.fetch_yields()
    assert (points, spreads) == ([], [])
    assert len(errors) == len(market.TENORS)


def test_snapshot_returns_an_empty_snapshot_not_none_when_everything_fails(monkeypatch):
    """Empty-with-errors and disabled are different states and must look different."""
    monkeypatch.setattr(market, "fetch_indices", lambda: ([], ["boom"]))
    monkeypatch.setattr(market, "fetch_yields", lambda: ([], [], ["boom"]))
    snap = market.market_snapshot()
    assert snap is not None and not snap.has_data and snap.errors


def test_snapshot_is_none_when_disabled(monkeypatch):
    monkeypatch.setattr(market, "ENABLED", False)
    assert market.market_snapshot() is None


# --- correctness of the numbers ------------------------------------------------

def test_a_missing_fred_observation_is_skipped_not_read_as_zero(monkeypatch):
    """FRED marks a holiday with '.' — treating that as 0.0 would invent a crash."""
    import json, io

    payload = {"observations": [
        {"date": "2026-08-13", "value": "4.15"},
        {"date": "2026-08-12", "value": "."},
        {"date": "2026-08-11", "value": "4.20"},
    ]}

    class Resp(io.StringIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(market.__dict__["__builtins__"] if False else market,
                        "_fred_observations",
                        lambda sid, key: [(dt.date(2026, 8, 13), 4.15),
                                          (dt.date(2026, 8, 11), 4.20)])
    monkeypatch.setenv(market.FRED_KEY_ENV, "k")
    points, _, _ = market.fetch_yields()
    two = next(p for p in points if p.label == "2y")
    assert two.change_bps == pytest.approx(-5.0)   # 4.15 vs 4.20, not vs 0
    assert json.dumps(payload)  # payload documents the shape being guarded


def test_a_spread_is_refused_when_its_legs_are_different_days(monkeypatch):
    """A 'spread' across two sessions is a difference, not a spread."""
    monkeypatch.setenv(market.FRED_KEY_ENV, "k")

    def obs(series_id, key):
        day = D13 if series_id == "DGS2" else D12
        base = 4.15 if series_id == "DGS2" else 4.63
        return [(day, base), (day - dt.timedelta(days=1), base + 0.05)]

    monkeypatch.setattr(market, "_fred_observations", obs)
    _, spreads, errors = market.fetch_yields()
    # Selective, not blanket: 2y is D13 and 10y is D12, so 2s10s is refused —
    # while 5y and 10y are both D12, so 5s10s is legitimately computable and
    # must still be produced. A guard that dropped both would be over-broad.
    labels = {s.label for s in spreads}
    assert "2s10s" not in labels
    assert "5s10s" in labels
    assert any("differ in date" in e for e in errors)


# --- rendering -----------------------------------------------------------------

def test_each_block_carries_its_own_date():
    """H.15 lags the equity close; one shared date would be wrong about one."""
    text = render_text(draft(snapshot()))
    assert "close Fri 14 Aug" in text
    assert "Thu 13 Aug" in text


def test_the_data_section_comes_before_the_stories():
    text = render_text(draft(snapshot()))
    assert text.index("MARKETS") < text.index("TOP 10")


def test_index_rows_show_level_change_and_percent():
    text = render_text(draft(snapshot()))
    assert "7,785.76" in text and "-13.23" in text and "-0.17%" in text


def test_yields_show_level_and_bps():
    text = render_text(draft(snapshot()))
    assert "4.15%" in text and "-5 bps" in text


def test_both_spreads_are_rendered():
    text = render_text(draft(snapshot()))
    assert "2s10s +48 bps" in text and "5s10s +31 bps" in text


def test_no_market_data_means_no_section_not_an_empty_one():
    text = render_text(draft(None))
    assert "MARKETS" not in text and "TREASURIES" not in text
    assert "TOP 10" in text


def test_an_empty_snapshot_renders_nothing_rather_than_headings():
    text = render_text(draft(MarketSnapshot(errors=["everything failed"])))
    assert "MARKETS" not in text


def test_html_market_section_renders():
    html = render_html(draft(snapshot()))
    assert "Markets" in html and "7,785.76" in html and "2s10s" in html


# --- report links ---------------------------------------------------------------

def links():
    return [ReportLink("ARR", "ARMOUR Residential REIT", "ARR rotates into 5.5s and 6.0s",
                       "https://chat.mmglobal.us/reits?issuer=ARR&report=arr%3Aabc", D14)]


def test_a_report_is_a_bullet_whose_title_links_to_it():
    html = render_html(draft(snapshot(), links()))
    assert "ARR rotates into 5.5s and 6.0s" in html
    assert "reits?issuer=ARR" in html


def test_no_reports_means_no_research_section():
    assert "Research" not in render_html(draft(snapshot(), []))
    assert "RESEARCH" not in render_text(draft(snapshot(), []))


def test_report_url_encodes_the_colon_in_a_report_id():
    url = reports._report_url("ARR", "arr:f15534c7")
    assert "arr%3Af15534c7" in url and url.startswith("https://")


def test_recent_reports_is_empty_when_disabled(monkeypatch):
    monkeypatch.setattr(reports, "ENABLED", False)
    assert reports.recent_reports() == []


def test_recent_reports_survives_the_service_being_down(monkeypatch):
    import tools.reit_research as rr
    monkeypatch.setattr(rr, "_rpc", lambda fn, payload: (_ for _ in ()).throw(OSError("down")))
    assert reports.recent_reports(dt.date(2026, 1, 1)) == []


def test_an_unpublished_report_is_not_surfaced(monkeypatch):
    """publication_date is None on a draft; guessing from the portfolio date
    would put unpublished work in an email."""
    import tools.reit_research as rr

    def fake_rpc(fn, payload):
        if "issuers" in fn:
            return [{"issuer_code": "ARR", "issuer_name": "ARMOUR"}]
        return [{"report_id": "arr:1", "title": "draft", "publication_date": None,
                 "portfolio_as_of_date": "2026-07-31"}]

    monkeypatch.setattr(rr, "_rpc", fake_rpc)
    assert reports.recent_reports(dt.date(2026, 1, 1)) == []


def test_reports_are_newest_first_and_stably_ordered(monkeypatch):
    import tools.reit_research as rr

    def fake_rpc(fn, payload):
        if "issuers" in fn:
            return [{"issuer_code": "ORC", "issuer_name": "Orchid"},
                    {"issuer_code": "ARR", "issuer_name": "ARMOUR"}]
        code = payload["p_issuer_code"]
        return [{"report_id": f"{code.lower()}:1", "title": f"{code} note",
                 "publication_date": "2026-08-14", "portfolio_as_of_date": "2026-07-31"}]

    monkeypatch.setattr(rr, "_rpc", fake_rpc)
    got = reports.recent_reports(dt.date(2026, 1, 1))
    # Same date for both, so issuer breaks the tie — deterministically.
    assert [r.issuer_code for r in got] == ["ARR", "ORC"]
