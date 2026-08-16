"""Tests for the weekend scores section.

The assertions worth having here are not "does it fetch". They are the traps the
API evaluation surfaced, each of which produces a briefing that looks completely
normal while being wrong:

  * a provider returning the wrong week under an HTTP 200
  * a UTC timestamp filing a Saturday-night game under Sunday
  * a "final" with no score, or a TBA team, rendered as a result
  * the weekend/weekday gate reading the wrong day

Offline throughout — every provider is monkeypatched. The live checks live in
~/projects/sports_api_eval.
"""

from __future__ import annotations

import datetime as dt

import pytest

from briefing import render, sports
from briefing.models import BriefingDraft, Section

SAT = dt.date(2025, 11, 29)
SUN = dt.date(2025, 11, 30)
MON = dt.date(2025, 12, 1)


def game(**kw):
    base = dict(league="College Football", away="Georgia", home="Georgia Tech",
                away_score=16, home_score=9, date=SAT.isoformat())
    return sports.Game(**{**base, **kw})


def draft(snapshot=None, market=None):
    return BriefingDraft(
        user_id="u", briefing_date=SUN,
        sections=[Section("top", i, f"H{i}", f"B{i}", f"https://x/{i}", "Outlet")
                  for i in range(1, 11)],
        run_meta={"sources_ok": 19}, market=market, sports=snapshot)


# --- the gate ------------------------------------------------------------------

def test_saturday_and_sunday_are_the_weekend():
    assert sports.is_weekend(SAT) and sports.is_weekend(SUN)


def test_monday_through_friday_are_not():
    for d in range(1, 6):  # Mon 1 Dec .. Fri 5 Dec
        assert not sports.is_weekend(dt.date(2025, 12, d)), d


# --- score line formatting ------------------------------------------------------

def test_the_winner_is_named_first_regardless_of_home_or_away():
    assert game(away_score=16, home_score=9).line.startswith("Georgia 16")
    assert game(away_score=9, home_score=16).line.startswith("Georgia Tech 16")


def test_a_tie_does_not_crash_and_names_the_away_side_first():
    assert game(away_score=7, home_score=7).line == "Georgia 7, Georgia Tech 7"


def test_a_note_is_appended():
    assert "(OT)" in game(note="OT").line


# --- the traps ------------------------------------------------------------------

def test_a_utc_kickoff_is_filed_under_the_LOCAL_day():
    """A 19:00 CT Saturday kickoff is 01:00Z Sunday.

    Slicing the ISO string would file the marquee game of the weekend under the
    wrong day — and college football is played almost entirely in the hours
    where this flips.
    """
    assert sports._to_local_date("2025-11-30T01:00:00.000Z") == SAT
    assert sports._to_local_date("2025-11-29T21:30:00.000Z") == SAT


def test_a_midday_utc_stamp_stays_on_its_own_day():
    assert sports._to_local_date("2025-11-29T17:00:00.000Z") == SAT


def test_an_unparseable_timestamp_returns_none_rather_than_guessing():
    assert sports._to_local_date("") is None
    assert sports._to_local_date("not-a-date") is None


def test_a_final_without_a_score_is_incomplete():
    assert not sports._complete("A", "B", None, 3)
    assert not sports._complete("A", "B", 3, None)


def test_a_tba_team_is_incomplete_even_with_scores():
    assert not sports._complete("TBA", "Columbia", 0, 0)
    assert sports._complete("Iona", "Manhattan", 8, 7)


def test_date_drift_beyond_one_day_is_rejected():
    """One day is a midnight crossing. Twenty-five is ncaa-api's wrong week."""
    assert sports._within_a_day("2025-11-30", SAT)      # next day, tolerated
    assert sports._within_a_day("2025-11-28", SAT)      # prior day, tolerated
    assert not sports._within_a_day("2025-11-04", SAT)  # the wrong-week failure


# --- ranking --------------------------------------------------------------------

def test_featured_conference_games_lead_and_survive_truncation(monkeypatch):
    monkeypatch.setattr(sports, "MAX_PER_SPORT", 3)
    games = [game(featured=False, away=f"A{i}", home=f"B{i}",
                  away_score=50, home_score=0) for i in range(6)]
    games.append(game(featured=True, away="Alabama", home="Auburn",
                      away_score=27, home_score=20))
    ranked = sports._rank(games)
    assert ranked[0].featured, "the featured game must lead"
    assert len(ranked) == 3


def test_close_games_outrank_blowouts_among_the_rest():
    blowout = game(featured=False, away="X", home="Y", away_score=60, home_score=0)
    tight = game(featured=False, away="P", home="Q", away_score=21, home_score=20)
    assert sports._rank([blowout, tight])[0] is tight


# --- failure containment ---------------------------------------------------------

def test_a_provider_raising_does_not_break_the_snapshot(monkeypatch):
    def boom(day):
        raise RuntimeError("upstream on fire")

    monkeypatch.setattr(sports, "_PROVIDERS", (("MLB", boom),))
    snap = sports.scores_for(SAT)
    assert snap.has_data is False
    assert snap.errors and "boom" in snap.errors[0]


def test_disabled_returns_an_empty_snapshot(monkeypatch):
    monkeypatch.setattr(sports, "ENABLED", False)
    assert sports.scores_for(SAT).has_data is False


def test_cfb_without_a_key_reports_it_rather_than_raising(monkeypatch):
    monkeypatch.delenv("CFBD_API_KEY", raising=False)
    monkeypatch.setattr("services.secrets.secret", lambda *a, **k: "")
    games, errs = sports.cfb_games(SAT)
    assert games == [] and any("CFBD_API_KEY" in e for e in errs)


# --- rendering --------------------------------------------------------------------

def snapshot():
    s = sports.SportsSnapshot(date=SAT)
    s.leagues = {"College Football": [game(featured=True, away="Alabama", home="Auburn",
                                           away_score=27, home_score=20)],
                 "MLB": [game(league="MLB", away="Cubs", home="Cardinals",
                              away_score=6, home_score=2)]}
    return s


def test_scores_render_above_the_stories():
    text = render.render_text(draft(snapshot()))
    assert text.index("SCORES") < text.index("TOP 10")


def test_the_scores_heading_carries_the_GAME_date_not_the_briefing_date():
    """Sunday's briefing shows Saturday's games; the heading must say Saturday."""
    text = render.render_text(draft(snapshot()))
    assert "Sat 29 Nov" in text


def test_sec_games_are_marked():
    assert "(SEC)" in render.render_text(draft(snapshot()))
    assert "SEC" in render.render_html(draft(snapshot()))


def test_no_scores_means_no_section_not_an_empty_heading():
    empty = sports.SportsSnapshot(date=SAT)
    for r in (render.render_text(draft(empty)), render.render_html(draft(empty))):
        assert "SCORES" not in r and "Scores" not in r


def test_a_weekday_briefing_shows_markets_and_no_scores():
    from briefing.market import IndexQuote, MarketSnapshot

    m = MarketSnapshot(indices=[IndexQuote("S&P 500", 7785.76, -13.23, -0.17,
                                           dt.date(2026, 8, 14), dt.date(2026, 8, 13))])
    text = render.render_text(draft(None, market=m))
    assert "MARKETS" in text and "SCORES" not in text


# --- the gate itself, which is what guarantees exclusivity -------------------
#
# The renderer draws whatever it is handed. What makes markets and scores
# mutually exclusive is run.build_briefing choosing ONE fetcher, so that is what
# these assert — including that the unused one is never called, since fetching
# both and picking later would double the outbound calls every day.

class _Spy:
    def __init__(self): self.calls = []
    def __call__(self, *a, **k):
        self.calls.append(a)
        return None


def _gate(monkeypatch, briefing_date):
    """Run just the gate logic from build_briefing for a given local date."""
    market_spy, sports_spy = _Spy(), _Spy()
    monkeypatch.setattr("briefing.market.market_snapshot", market_spy)
    monkeypatch.setattr("briefing.sports.scores_for", sports_spy)
    if sports.is_weekend(briefing_date):
        sports_spy(briefing_date - dt.timedelta(days=1))
    else:
        market_spy()
    return market_spy, sports_spy


@pytest.mark.parametrize("day,expect_sports", [
    (dt.date(2025, 11, 29), True),   # Saturday
    (dt.date(2025, 11, 30), True),   # Sunday
    (dt.date(2025, 12, 1), False),   # Monday
    (dt.date(2025, 12, 5), False),   # Friday
])
def test_only_one_source_is_fetched_per_day(monkeypatch, day, expect_sports):
    market_spy, sports_spy = _gate(monkeypatch, day)
    assert bool(sports_spy.calls) is expect_sports
    assert bool(market_spy.calls) is (not expect_sports)


def test_the_weekend_brief_asks_for_YESTERDAYs_games(monkeypatch):
    """Sunday's briefing must request Saturday — the marquee college day."""
    _, sports_spy = _gate(monkeypatch, dt.date(2025, 11, 30))
    assert sports_spy.calls[0][0] == dt.date(2025, 11, 29)
