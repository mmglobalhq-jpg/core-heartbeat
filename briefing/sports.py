"""Yesterday's scores, for the weekend edition of the briefing.

WHAT RUNS WHEN
Monday–Friday the briefing opens with market data. Saturday and Sunday it opens
with this instead. The gate is the BRIEFING's local weekday, not the data's:
Sunday's briefing carries Saturday's college football, which is the point.

PROVIDERS, AND WHY THESE ONES
Chosen by measurement in ~/projects/sports_api_eval, not by feature list:

  MLB            statsapi.mlb.com   first-party, no key, 15/15 verified complete
  NFL            nflverse           a CSV in git, not an API — cannot rate-limit,
                                    cannot truncate, cannot serve stale data
  NCAA football  CFBD               free key; conference=SEC is first-class
  NCAA baseball  ncaa-api           the only source that serves it at all

Cross-checks (Highlightly for baseball) live in the evaluation project. This
module deliberately ships the PRIMARY of each sport only: a second live call per
sport doubles the failure surface of a section that must never break a briefing,
and the reconciliation work belongs in evaluation until it earns its place.

THREE TRAPS THIS MODULE IS BUILT AROUND, all found by measurement:

1. **ncaa-api's football DATE path silently returns the wrong week.** Asking for
   2025/11/29 returns games from 4–7 November — the `11` is read as *week 11*
   and the day discarded. HTTP 200, plausible finals, 25 days stale. Football is
   therefore NOT taken from ncaa-api here at all; CFBD is, and every returned
   game is re-checked against the date we asked for.

2. **Providers disagree about what day a game belongs to.** Late games cross
   midnight in UTC. One day of drift is tolerated; more is treated as the
   provider ignoring the request, and the whole section is dropped rather than
   printed with the wrong day's scores.

3. **"Final" without a score, and TBA teams, are common.** ncaa-api baseball
   returned 10 scoreless finals and 5 TBA rows out of 140. Both are dropped —
   an omitted game is a worse briefing, a wrong one is a broken briefing.

NOTHING HERE MAY FAIL A BRIEFING. Every entry point returns partial data or an
empty snapshot; failures land in ``errors`` so the run reports them.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

ENABLED = os.environ.get("BRIEFING_SPORTS", "1") not in ("0", "false", "False", "")
TIMEOUT_S = float(os.environ.get("BRIEFING_SPORTS_TIMEOUT_S", "20"))
UA = "MMGlobalBriefing/1.0 (+https://mmglobal.us; daily brief, personal use)"

CFBD_KEY_ENV = "CFBD_API_KEY"
LOCAL_TZ = os.environ.get("BRIEFING_TIMEZONE", "America/Chicago")

# Per sport, so one blowout league cannot crowd out the rest.
MAX_PER_SPORT = int(os.environ.get("BRIEFING_SPORTS_MAX_PER_LEAGUE", "8"))
# Conference to lead college football with. Its games are listed first and are
# never truncated away in favour of others.
FEATURED_CONFERENCE = os.environ.get("BRIEFING_SPORTS_CONFERENCE", "SEC")

NFLVERSE_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
MLB_URL = "https://statsapi.mlb.com/api/v1/schedule"
CFBD_URL = "https://api.collegefootballdata.com/games"
NCAA_URL = "https://ncaa-api.henrygd.me"


@dataclass(frozen=True)
class Game:
    league: str          # display label: "NFL", "College Football", …
    away: str
    home: str
    away_score: int
    home_score: int
    date: str            # provider's own date, for the drift check
    featured: bool = False   # SEC (or configured conference)
    note: str = ""           # "OT", "F/11", …

    @property
    def line(self) -> str:
        """Winner first — 'Georgia 31, Alabama 28'."""
        if self.away_score >= self.home_score:
            a, b = (self.away, self.away_score), (self.home, self.home_score)
        else:
            a, b = (self.home, self.home_score), (self.away, self.away_score)
        tail = f" ({self.note})" if self.note else ""
        return f"{a[0]} {a[1]}, {b[0]} {b[1]}{tail}"


@dataclass
class SportsSnapshot:
    date: dt.date | None = None
    leagues: dict[str, list[Game]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def has_data(self) -> bool:
        return any(self.leagues.values())

    def meta(self) -> dict:
        return {
            "date": self.date.isoformat() if self.date else None,
            "leagues": {k: len(v) for k, v in self.leagues.items() if v},
            "errors": list(self.errors),
        }


# --- helpers -----------------------------------------------------------------


def is_weekend(day: dt.date) -> bool:
    """Saturday or Sunday in the reader's local calendar."""
    return day.weekday() >= 5


def _get(url: str, headers: dict | None = None):
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        return json.load(r)


def _to_local_date(iso_utc: str) -> dt.date | None:
    """UTC instant -> the reader's local calendar date.

    Slicing the first ten characters would be wrong in the way that matters most:
    a 19:00 CT Saturday kickoff is 00:00Z SUNDAY, so a string slice files the
    marquee game of the weekend under the wrong day. College football is played
    almost entirely in the hours where this flips.
    """
    if not iso_utc:
        return None
    try:
        from zoneinfo import ZoneInfo
        stamp = dt.datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=dt.UTC)
        return stamp.astimezone(ZoneInfo(LOCAL_TZ)).date()
    except Exception:
        try:
            return dt.date.fromisoformat(iso_utc[:10])
        except ValueError:
            return None


def _within_a_day(got: str, want: dt.date) -> bool:
    """Tolerate a midnight crossing; reject a provider ignoring the request."""
    try:
        return abs((dt.date.fromisoformat(got[:10]) - want).days) <= 1
    except (ValueError, TypeError):
        return False


def _complete(away: str, home: str, a, h) -> bool:
    if a is None or h is None:
        return False
    return "tba" not in away.lower() and "tba" not in home.lower()


# --- providers ---------------------------------------------------------------


def mlb_games(day: dt.date) -> tuple[list[Game], list[str]]:
    try:
        q = urllib.parse.urlencode({"sportId": 1, "date": day.isoformat(), "hydrate": "team"})
        payload = _get(f"{MLB_URL}?{q}")
    except Exception as exc:
        return [], [f"MLB: {type(exc).__name__}"]
    out = []
    for d in payload.get("dates", []):
        for g in d.get("games", []):
            if "final" not in (g.get("status", {}).get("detailedState", "")).lower():
                continue
            a, h = g["teams"]["away"], g["teams"]["home"]
            an, hn = a["team"]["name"], h["team"]["name"]
            if not _complete(an, hn, a.get("score"), h.get("score")):
                continue
            if not _within_a_day(d.get("date", ""), day):
                continue
            out.append(Game("MLB", an, hn, a["score"], h["score"], d.get("date", "")))
    return out, []


def nfl_games(day: dt.date) -> tuple[list[Game], list[str]]:
    import csv
    import io
    try:
        req = urllib.request.Request(NFLVERSE_URL, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=TIMEOUT_S * 3) as r:
            rows = list(csv.DictReader(io.StringIO(r.read().decode("utf-8"))))
    except Exception as exc:
        return [], [f"NFL: {type(exc).__name__}"]
    out = []
    want = day.isoformat()
    for row in rows:
        if row.get("gameday") != want:
            continue
        def num(k):
            v = (row.get(k) or "").strip()
            try:
                return int(float(v)) if v else None
            except ValueError:
                return None
        a, h = num("away_score"), num("home_score")
        # No status column: scores present means played. Stated in the module
        # docstring because it is an inference the file does not assert.
        if not _complete(row.get("away_team", ""), row.get("home_team", ""), a, h):
            continue
        label = "NFL" if (row.get("game_type") or "REG") == "REG" else "NFL Playoffs"
        out.append(Game(label, row["away_team"], row["home_team"], a, h, want,
                        note="OT" if (row.get("overtime") or "") == "1" else ""))
    return out, []


def cfb_games(day: dt.date) -> tuple[list[Game], list[str]]:
    """College football via CFBD, addressed by DATE and filtered on the response.

    CFBD is queried by year+week, so the week is derived and then every returned
    game is checked against the day we actually wanted. That second step is not
    belt-and-braces — it is the only thing standing between us and ncaa-api's
    wrong-week failure happening here in a different costume.
    """
    key = ""
    try:
        from services.secrets import secret as read_secret
        key = (read_secret(CFBD_KEY_ENV) or "").strip()
    except Exception:
        key = (os.environ.get(CFBD_KEY_ENV) or "").strip()
    if not key:
        return [], ["college football: no CFBD_API_KEY"]

    # CFBD's season runs Aug–Jan; a January game belongs to the prior season.
    year = day.year if day.month >= 7 else day.year - 1
    out, errors = [], []
    for season_type in ("regular", "postseason"):
        try:
            q = urllib.parse.urlencode({"year": year, "seasonType": season_type})
            rows = _get(f"{CFBD_URL}?{q}", {"Authorization": f"Bearer {key}"})
        except Exception as exc:
            errors.append(f"college football ({season_type}): {type(exc).__name__}")
            continue
        for g in rows or []:
            if not g.get("completed"):
                continue
            # CFBD returns camelCase (awayTeam/awayPoints/startDate). The older
            # snake_case names silently yield None for every field, which reads
            # as "no games" rather than as an error — checked against a known
            # 94-game Saturday to catch it.
            local = _to_local_date(str(g.get("startDate") or ""))
            if local != day:
                continue
            a, h = g.get("awayPoints"), g.get("homePoints")
            an, hn = g.get("awayTeam") or "", g.get("homeTeam") or ""
            if not _complete(an, hn, a, h):
                continue
            featured = FEATURED_CONFERENCE in (
                (g.get("homeConference") or ""), (g.get("awayConference") or ""))
            note = "neutral site" if g.get("neutralSite") else ""
            out.append(Game("College Football", an, hn, a, h, local.isoformat(),
                            featured=featured, note=note))
    return out, errors


def cbb_games(day: dt.date) -> tuple[list[Game], list[str]]:
    """College baseball via ncaa-api. Its DATE path is verified correct for
    baseball (three consecutive dates returned disjoint, correctly-dated sets) —
    unlike football, where the same-shaped path is silently week-based."""
    path = f"/scoreboard/baseball/d1/{day.year}/{day.month:02d}/{day.day:02d}/all-conf"
    try:
        payload = _get(f"{NCAA_URL}{path}")
    except Exception as exc:
        return [], [f"college baseball: {type(exc).__name__}"]
    out = []
    for row in payload.get("games", []):
        g = row.get("game", row)
        if "final" not in str(g.get("gameState", "")).lower():
            continue
        def side(k):
            s = g.get(k) or {}
            names = s.get("names") or {}
            v = s.get("score")
            try:
                v = int(v) if v not in (None, "", "-") else None
            except (TypeError, ValueError):
                v = None
            return (names.get("short") or names.get("full") or "?"), v
        an, a = side("away")
        hn, h = side("home")
        if not _complete(an, hn, a, h):
            continue
        raw = str(g.get("startDate") or "")
        iso = day.isoformat()
        if "/" in raw:
            m, d2, y = raw.split("/")
            iso = f"{y}-{m}-{d2}"
        if not _within_a_day(iso, day):
            continue
        out.append(Game("College Baseball", an, hn, a, h, iso))
    return out, []


# --- entry point -------------------------------------------------------------

_PROVIDERS = (
    ("College Football", cfb_games),
    ("College Baseball", cbb_games),
    ("NFL", nfl_games),
    ("MLB", mlb_games),
)


def _rank(games: list[Game]) -> list[Game]:
    """Featured conference first, then by margin — a one-score game is the one
    worth reading. Truncation drops the least interesting, never a featured game."""
    featured = [g for g in games if g.featured]
    rest = sorted((g for g in games if not g.featured),
                  key=lambda g: abs(g.away_score - g.home_score))
    return (featured + rest)[:MAX_PER_SPORT]


def scores_for(day: dt.date) -> SportsSnapshot:
    """Completed games for ``day``. Never raises; returns an empty snapshot when
    nothing is in season, which is the normal state for much of the year."""
    snap = SportsSnapshot(date=day)
    if not ENABLED:
        return snap
    for _, fn in _PROVIDERS:
        try:
            games, errs = fn(day)
        except Exception as exc:  # a provider contract violation, not an outage
            logger.warning("sports provider %s raised: %s", fn.__name__, type(exc).__name__)
            games, errs = [], [f"{fn.__name__}: {type(exc).__name__}"]
        snap.errors.extend(errs)
        for g in games:
            snap.leagues.setdefault(g.league, []).append(g)
    for league in list(snap.leagues):
        snap.leagues[league] = _rank(snap.leagues[league])
        if not snap.leagues[league]:
            del snap.leagues[league]
    return snap
