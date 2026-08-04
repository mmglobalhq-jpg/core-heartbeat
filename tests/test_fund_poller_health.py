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
    """Stands in for PostgREST. Keyed by (table, fund_id) where it matters."""

    def __init__(self, funds, snapshots=None, successes=None, unresolved=None, recent=None):
        self.funds = funds
        self.snapshots = snapshots or {}
        self.successes = successes or {}
        self.unresolved = unresolved or {}
        self.recent = recent or {}
        self.calls: list[str] = []

    def __call__(self, path, **params):
        self.calls.append(path)
        if path == "funds":
            return self.funds
        fund_id = params.get("fund_id", "").removeprefix("eq.")
        if path == "fund_snapshots":
            value = self.snapshots.get(fund_id)
            return [{"as_of_date": value}] if value else []
        if path == "poll_attempts":
            if params.get("status", "").startswith("in.(success"):
                value = self.successes.get(fund_id)
                return [{"attempt_time": value}] if value else []
            if params.get("status") == "eq.material_revision_applied":
                return []
            if params.get("is_resolved") == "is.false":
                return [{"id": str(n)} for n in range(self.unresolved.get(fund_id, 0))]
            return [{"status": s} for s in self.recent.get(fund_id, [])]
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
# Reporting the two pollers independently
# --------------------------------------------------------------------------- #
def test_pollers_are_reported_separately(monkeypatch):
    def fake_report(*, poller, tickers, stale_after_days, today):
        return {"poller": poller, "healthy": poller == "jp", "funds_total": len(tickers)}

    monkeypatch.setattr(fund_pollers, "_poller_report", fake_report)
    report = fund_pollers.fund_poller_health(today=TODAY)
    assert report["jp"]["healthy"] is True
    assert report["allspring"]["healthy"] is False
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
    assert report["healthy"] is False


def test_report_records_that_no_outbound_channel_exists(monkeypatch):
    monkeypatch.setattr(
        fund_pollers,
        "_poller_report",
        lambda **kw: {"poller": kw["poller"], "healthy": True},
    )
    assert fund_pollers.fund_poller_health(today=TODAY)["outbound_alerting"] == "none"
