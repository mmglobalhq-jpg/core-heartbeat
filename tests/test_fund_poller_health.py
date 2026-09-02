"""Fund-poller health reporting in the gateway.

The pollers are one-shot systemd units, so "is it up?" is meaningless — they
fail by going quiet. These tests pin the properties that make the endpoint
trustworthy: stale data is caught, one poller's outage cannot mask the other's
state, and an unreachable check reports *unknown* rather than *healthy*.
"""

from __future__ import annotations

import datetime as dt

import pytest

from services import fund_pollers


class FakeApi:
    """Stands in for PostgREST, for both query shapes the report issues.

    Health data is now fetched two ways: snapshot/unresolved/needs-review counts in
    one batched request each (``fund_id=in.(...)``, every row carrying its own
    ``fund_id``), and last-success/recent-attempts still per fund
    (``fund_id=eq.X``), because "latest row per fund" has no single-request form.
    This models both faithfully — batched responses carry ``fund_id`` and come back
    ordered, exactly as PostgREST would return them — so the assertions below still
    exercise real grouping logic rather than passing by construction.
    """

    def __init__(self, funds, snapshots=None, successes=None, unresolved=None, recent=None):
        self.funds = funds
        self.snapshots = snapshots or {}
        self.successes = successes or {}
        self.unresolved = unresolved or {}
        self.recent = recent or {}
        self.calls: list[str] = []

    @staticmethod
    def _batched_ids(params):
        """Fund ids from a ``fund_id=in.(a,b,c)`` filter, or None if not batched."""
        raw = params.get("fund_id", "")
        if not raw.startswith("in.("):
            return None
        return [i for i in raw[4:].rstrip(")").split(",") if i]

    def __call__(self, path, **params):
        self.calls.append(path)
        if path == "funds":
            return self.funds

        ids = self._batched_ids(params)

        if path == "fund_snapshots":
            if ids is None:  # legacy per-fund shape
                value = self.snapshots.get(params.get("fund_id", "").removeprefix("eq."))
                return [{"as_of_date": value}] if value else []
            rows = [
                {"fund_id": fid, "as_of_date": self.snapshots[fid]}
                for fid in ids
                if self.snapshots.get(fid)
            ]
            # PostgREST honours order=as_of_date.desc; the report takes the first
            # row seen per fund, so returning these unordered would be a lie.
            return sorted(rows, key=lambda r: r["as_of_date"], reverse=True)

        if path == "poll_attempts":
            if params.get("status", "").startswith("in.(success"):
                value = self.successes.get(params.get("fund_id", "").removeprefix("eq."))
                return [{"attempt_time": value}] if value else []
            if params.get("status") == "eq.material_revision_applied":
                return []
            if params.get("is_resolved") == "is.false":
                if ids is None:
                    fid = params.get("fund_id", "").removeprefix("eq.")
                    return [{"id": str(n)} for n in range(self.unresolved.get(fid, 0))]
                return [
                    {"fund_id": fid}
                    for fid in ids
                    for _ in range(self.unresolved.get(fid, 0))
                ]
            fid = params.get("fund_id", "").removeprefix("eq.")
            return [{"status": s} for s in self.recent.get(fid, [])]

        raise AssertionError(f"unexpected table {path}")


@pytest.fixture
def patch_api(monkeypatch):
    def apply(fake):
        monkeypatch.setattr(fund_pollers, "_get", fake)
        return fake

    return apply


TODAY = dt.date(2026, 8, 4)


def one_jp_fund(**kwargs):
    return FakeApi(funds=[{"id": "f1", "ticker": "JAGG", "is_active": True}], **kwargs)


# --------------------------------------------------------------------------- #
# Streak logic
# --------------------------------------------------------------------------- #
def test_consecutive_failures_stops_at_the_first_healthy_attempt():
    assert (
        fund_pollers._consecutive_failures(
            ["download_failed", "parse_failed", "unchanged", "download_failed"]
        )
        == 2
    )


def test_equivalent_reissue_breaks_the_streak():
    assert fund_pollers._consecutive_failures(["equivalent_reissue", "download_failed"]) == 0


def test_skips_are_transparent_to_the_streak():
    assert (
        fund_pollers._consecutive_failures(
            ["skipped_already_succeeded", "download_failed", "download_failed"]
        )
        == 2
    )


def test_no_attempts_is_no_failures():
    assert fund_pollers._consecutive_failures([]) == 0


# --------------------------------------------------------------------------- #
# Staleness
# --------------------------------------------------------------------------- #
def test_friday_data_on_monday_is_not_stale(patch_api):
    """The false-positive this whole change exists to avoid."""
    patch_api(
        one_jp_fund(
            snapshots={"f1": "2026-07-31"},
            successes={"f1": "2026-08-01T06:00:00+00:00"},
            recent={"f1": ["equivalent_reissue"]},
        )
    )
    report = fund_pollers._poller_report(
        poller="jp", tickers=["JAGG"], stale_after_days=4, today=dt.date(2026, 8, 3)
    )
    assert not report["stale_data"]
    assert report["healthy"]


def test_data_older_than_the_tolerance_is_stale(patch_api):
    patch_api(
        one_jp_fund(
            snapshots={"f1": "2026-07-20"},
            successes={"f1": "2026-07-21T06:00:00+00:00"},
            recent={"f1": ["unchanged"]},
        )
    )
    report = fund_pollers._poller_report(
        poller="jp", tickers=["JAGG"], stale_after_days=4, today=TODAY
    )
    assert report["stale_data"]
    assert not report["healthy"]
    assert report["funds"][0]["as_of_age_days"] == 15


def test_a_fund_with_no_data_at_all_is_stale(patch_api):
    patch_api(one_jp_fund())
    report = fund_pollers._poller_report(
        poller="jp", tickers=["JAGG"], stale_after_days=4, today=TODAY
    )
    assert report["stale_data"], "absent data must not read as fresh"
    assert report["funds"][0]["last_as_of_date"] is None


# --------------------------------------------------------------------------- #
# Unresolved failures
# --------------------------------------------------------------------------- #
def test_unresolved_failures_make_a_fund_unhealthy(patch_api):
    patch_api(
        one_jp_fund(
            snapshots={"f1": "2026-08-03"},
            successes={"f1": "2026-08-04T06:00:00+00:00"},
            unresolved={"f1": 3},
            recent={"f1": ["download_failed"]},
        )
    )
    report = fund_pollers._poller_report(
        poller="jp", tickers=["JAGG"], stale_after_days=4, today=TODAY
    )
    assert report["unresolved_failures"] == 3
    assert not report["healthy"]
    assert report["max_consecutive_failures"] == 1


def test_recovery_shows_as_healthy_again(patch_api):
    """After the failures are resolved, nothing lingers."""
    patch_api(
        one_jp_fund(
            snapshots={"f1": "2026-08-03"},
            successes={"f1": "2026-08-04T06:00:00+00:00"},
            unresolved={"f1": 0},
            recent={"f1": ["success", "download_failed"]},
        )
    )
    report = fund_pollers._poller_report(
        poller="jp", tickers=["JAGG"], stale_after_days=4, today=TODAY
    )
    assert report["healthy"]
    assert report["unresolved_failures"] == 0
    assert report["max_consecutive_failures"] == 0


# --------------------------------------------------------------------------- #
# Reporting every poller independently
# --------------------------------------------------------------------------- #
def test_pollers_are_reported_separately(monkeypatch):
    def fake_report(*, poller, tickers, stale_after_days, today):
        return {"poller": poller, "healthy": poller == "jp", "funds_total": len(tickers)}

    monkeypatch.setattr(fund_pollers, "_poller_report", fake_report)
    report = fund_pollers.fund_poller_health(today=TODAY)
    assert report["jp"]["healthy"] is True
    assert report["allspring"]["healthy"] is False
    assert report["regan"]["healthy"] is False
    assert report["healthy"] is False, "one unhealthy poller degrades the rollup"
    assert report["degraded"] is True
    assert not report["unknown"]


def test_one_poller_failing_to_report_does_not_break_the_other(monkeypatch):
    def fake_report(*, poller, tickers, stale_after_days, today):
        if poller == "allspring":
            raise RuntimeError("PostgREST unreachable")
        return {"poller": poller, "healthy": True, "funds_total": len(tickers)}

    monkeypatch.setattr(fund_pollers, "_poller_report", fake_report)
    report = fund_pollers.fund_poller_health(today=TODAY)
    assert report["jp"]["healthy"] is True
    assert report["regan"]["healthy"] is True, "an unrelated poller must be unaffected"
    assert report["allspring"]["status"] == "unknown"
    assert report["allspring"]["healthy"] is None
    assert report["unknown"] is True
    assert report["healthy"] is False, "an unreachable check is not a passing check"


def test_missing_configuration_is_surfaced_not_swallowed(monkeypatch):
    monkeypatch.delenv(fund_pollers.FUNDS_SUPABASE_URL_ENV, raising=False)
    monkeypatch.delenv(fund_pollers.FUNDS_SERVICE_ROLE_ENV, raising=False)
    report = fund_pollers.fund_poller_health(today=TODAY)
    assert report["jp"]["status"] == "unknown"
    assert report["allspring"]["status"] == "unknown"
    assert report["regan"]["status"] == "unknown"
    assert report["healthy"] is False


def test_report_records_the_outbound_alerting_state(monkeypatch):
    """This assertion has been wrong twice, and passed happily both times.

    "none" stopped being true on 2026-08-05 when the poller units gained
    OnFailure=alert@ drop-ins. "unit_onfailure_only" was wrong the same day it was
    written: it claimed nothing polls this rollup, when platform-watchdog check 5 had
    been GETting it every 30 minutes since 2026-08-10.

    A test that only pins whatever the code currently says cannot catch either error.
    The value describes machinery in another repository, so verifying it means looking
    at `systemctl show <unit> -p OnFailure` and at platform-watchdog.sh.
    """
    monkeypatch.setattr(
        fund_pollers,
        "_poller_report",
        lambda **kw: {"poller": kw["poller"], "healthy": True},
    )
    report = fund_pollers.fund_poller_health(today=TODAY)
    assert report["outbound_alerting"] == "unit_onfailure_and_watchdog_poll"


# --------------------------------------------------------------------------- #
# Regan
# --------------------------------------------------------------------------- #
def test_regan_tickers_match_what_the_poller_writes():
    """These funds carry is_active=false so the JP poller never selects them, which
    means this list is the only thing making them visible to monitoring."""
    assert fund_pollers.REGAN_TICKERS == ["MBSF", "MBSX"]


def _regan_api(**kwargs):
    return FakeApi(
        funds=[
            {"id": "f1", "ticker": "MBSF", "is_active": False},
            {"id": "f2", "ticker": "MBSX", "is_active": False},
        ],
        **kwargs,
    )


def _regan_report(today):
    return fund_pollers._poller_report(
        poller="regan",
        tickers=fund_pollers.REGAN_TICKERS,
        stale_after_days=fund_pollers.REGAN_STALE_AFTER_DAYS,
        today=today,
    )


def test_regan_tolerates_the_one_business_day_lag_between_its_funds(patch_api):
    """MBSF consistently serves the prior business day while MBSX serves the current
    one. A tolerance tuned for a same-day feed would call MBSF stale every weekend."""
    patch_api(
        _regan_api(
            snapshots={"f1": "2026-08-31", "f2": "2026-09-04"},  # MBSF 5 days behind
            successes={
                "f1": "2026-09-04T06:00:00+00:00",
                "f2": "2026-09-04T06:00:00+00:00",
            },
            unresolved={"f1": 0, "f2": 0},
            recent={"f1": ["success"], "f2": ["success"]},
        )
    )
    report = _regan_report(dt.date(2026, 9, 5))
    assert report["healthy"], "a long-weekend lag on MBSF is normal, not stale"
    assert report["funds_stale"] == 0
    assert report["funds_total"] == 2


def test_regan_still_reports_a_genuinely_stalled_fund(patch_api):
    """The tolerance is loose, not absent."""
    patch_api(
        _regan_api(
            snapshots={"f1": "2026-08-20", "f2": "2026-09-04"},
            successes={"f1": None, "f2": "2026-09-04T06:00:00+00:00"},
            unresolved={"f1": 0, "f2": 0},
            recent={"f1": ["success"], "f2": ["success"]},
        )
    )
    report = _regan_report(dt.date(2026, 9, 5))
    assert not report["healthy"]
    assert report["funds_stale"] == 1
    assert [f["ticker"] for f in report["funds"] if f["stale_data"]] == ["MBSF"]


def test_regan_is_inactive_by_design_and_still_monitored(patch_api):
    """Both funds carry is_active=false so the JP poller never selects them. That is
    exactly why they need an explicit ticker list here — nothing derives them."""
    patch_api(
        _regan_api(
            snapshots={"f1": "2026-09-04", "f2": "2026-09-04"},
            successes={"f1": "2026-09-04T06:00:00+00:00", "f2": "2026-09-04T06:00:00+00:00"},
            unresolved={"f1": 0, "f2": 0},
            recent={"f1": ["success"], "f2": ["success"]},
        )
    )
    report = _regan_report(dt.date(2026, 9, 5))
    assert report["funds_total"] == 2
    assert report["healthy"]
