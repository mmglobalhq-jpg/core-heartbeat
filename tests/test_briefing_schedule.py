"""Who is due, and when.

The scheduled job fires hourly and asks "who is due?" rather than running at a
fixed time. These tests pin the two properties that makes it safe: it does not
fire early in the user's own timezone, and it does not fire twice.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

from briefing.schedule import due_users, is_due, parse_time

CHICAGO = ZoneInfo("America/Chicago")


def at(hour: int, minute: int = 0, *, zone=CHICAGO) -> dt.datetime:
    return dt.datetime(2026, 8, 7, hour, minute, tzinfo=zone)


def prefs(**over) -> dict:
    return {"user_id": "u1", "deliver_at": "06:30",
            "timezone": "America/Chicago", **over}


class TestParseTime:
    def test_parses_hh_mm(self):
        assert parse_time("07:15") == dt.time(7, 15)

    def test_tolerates_seconds(self):
        # Postgres returns time as "06:30:00".
        assert parse_time("06:30:00") == dt.time(6, 30)

    def test_falls_back_on_nonsense(self):
        assert parse_time("not a time") == dt.time(6, 30)


class TestIsDue:
    def test_not_due_before_the_delivery_time(self):
        assert is_due(prefs(), existing_dates=set(), now=at(5, 0)) is False

    def test_due_after_the_delivery_time(self):
        assert is_due(prefs(), existing_dates=set(), now=at(7, 0)) is True

    def test_not_due_when_one_already_exists_for_the_local_date(self):
        assert is_due(prefs(), existing_dates={"2026-08-07"}, now=at(7, 0)) is False

    def test_timezone_is_the_users_not_the_hosts(self):
        # 07:00 UTC is 02:00 in Chicago — before the delivery time, so a user in
        # Chicago is NOT due even though the host's clock has passed 06:30.
        utc_seven = dt.datetime(2026, 8, 7, 7, 0, tzinfo=dt.UTC)
        assert is_due(prefs(), existing_dates=set(), now=utc_seven) is False

    def test_a_user_in_another_zone_is_evaluated_in_that_zone(self):
        utc_seven = dt.datetime(2026, 8, 7, 7, 0, tzinfo=dt.UTC)
        london = prefs(timezone="Europe/London")  # 08:00 local — past 06:30
        assert is_due(london, existing_dates=set(), now=utc_seven) is True

    def test_unknown_timezone_does_not_raise(self):
        assert is_due(prefs(timezone="Mars/Olympus"),
                      existing_dates=set(), now=at(7, 0)) in (True, False)


class FakeRepo:
    def __init__(self, prefs_rows, existing=None):
        self._prefs = prefs_rows
        self._existing = existing or {}
        self.lookups = []

    def list_enabled_prefs(self):
        return self._prefs

    def get_briefing(self, user_id, date):
        self.lookups.append((user_id, date))
        return self._existing.get((user_id, date.isoformat()))


class TestDueUsers:
    def test_returns_users_past_their_time(self):
        repo = FakeRepo([prefs()])
        assert len(due_users(repo, now=at(7, 0))) == 1

    def test_skips_users_with_a_ready_briefing(self):
        repo = FakeRepo([prefs()], {("u1", "2026-08-07"): {"status": "ready"}})
        assert due_users(repo, now=at(7, 0)) == []

    def test_retries_a_failed_briefing(self):
        # A row that exists but failed is not a briefing. The job depends on other
        # people's web servers, so a failure must get another attempt rather than
        # blocking the day.
        repo = FakeRepo([prefs()], {("u1", "2026-08-07"): {"status": "failed"}})
        assert len(due_users(repo, now=at(7, 0))) == 1

    def test_annotates_the_local_date(self):
        repo = FakeRepo([prefs()])
        assert due_users(repo, now=at(7, 0))[0]["briefing_date"] == dt.date(2026, 8, 7)

    def test_no_enabled_users_is_not_an_error(self):
        assert due_users(FakeRepo([]), now=at(7, 0)) == []


class TestOneBadRowDoesNotBreakEveryone:
    """The failure mode this guards: a single malformed preference row taking
    down the scheduled run for every other user."""

    def test_a_bad_timezone_does_not_hide_other_users(self):
        repo = FakeRepo([
            prefs(user_id="bad", timezone="Mars/Olympus"),
            prefs(user_id="good"),
        ])
        got = {u["user_id"] for u in due_users(repo, now=at(7, 0))}
        assert "good" in got

    def test_a_failing_lookup_does_not_hide_other_users(self):
        class Exploding(FakeRepo):
            def get_briefing(self, user_id, date):
                if user_id == "bad":
                    raise RuntimeError("database hiccup")
                return None

        repo = Exploding([prefs(user_id="bad"), prefs(user_id="good")])
        got = {u["user_id"] for u in due_users(repo, now=at(7, 0))}
        assert got == {"good"}
