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


# --------------------------------------------------------------------------- #
# Request count — the reason this endpoint was rewritten
# --------------------------------------------------------------------------- #
def test_request_count_does_not_grow_five_per_fund(patch_api):
    """The endpoint issued five sequential requests PER FUND — ~75 round-trips for
    15 funds, measured at 21-30s, long enough to trip a monitoring timeout on a
    health check.

    Snapshots, unresolved failures and needs-review counts are now one batched
    request each regardless of fund count; only the two top-N-per-fund lookups
    remain per fund, and those run concurrently. So the growth rate is 2 per fund
    plus a constant, not 5 per fund.
    """
    funds = [{"id": f"f{n}", "ticker": f"TICK{n}", "is_active": True} for n in range(15)]
    fake = patch_api(FakeApi(
        funds=funds,
        snapshots={f"f{n}": "2026-08-03" for n in range(15)},
        successes={f"f{n}": "2026-08-03T06:00:00+00:00" for n in range(15)},
        recent={f"f{n}": ["success"] for n in range(15)},
    ))
    report = fund_pollers._poller_report(
        poller="jp", tickers=[f"TICK{n}" for n in range(15)],
        stale_after_days=4, today=TODAY,
    )
    assert report["funds_total"] == 15

    # 1 funds + 3 batched + (2 x 15 per-fund) = 34, vs 1 + 75 = 76 before.
    assert len(fake.calls) == 34
    assert fake.calls.count("fund_snapshots") == 1, "snapshots must be one batched call"


def test_batched_counts_are_attributed_to_the_right_fund(patch_api):
    """The batched queries return every fund's rows in one response, so grouping by
    fund_id is now this module's job. Mis-grouping would report one fund's failures
    against another — worse than being slow."""
    patch_api(FakeApi(
        funds=[{"id": "f1", "ticker": "AAA", "is_active": True},
               {"id": "f2", "ticker": "BBB", "is_active": True}],
        snapshots={"f1": "2026-08-03", "f2": "2026-08-03"},
        successes={"f1": "2026-08-03T06:00:00+00:00", "f2": "2026-08-03T06:00:00+00:00"},
        recent={"f1": ["success"], "f2": ["success"]},
        unresolved={"f2": 3},
    ))
    report = fund_pollers._poller_report(
        poller="jp", tickers=["AAA", "BBB"], stale_after_days=4, today=TODAY,
    )
    by_ticker = {f["ticker"]: f for f in report["funds"]}
    assert by_ticker["AAA"]["unresolved_failures"] == 0
    assert by_ticker["BBB"]["unresolved_failures"] == 3
    assert by_ticker["AAA"]["healthy"] is True
    assert by_ticker["BBB"]["healthy"] is False


def test_no_matching_funds_reports_unhealthy_not_a_crash(patch_api):
    """An empty id list would produce a malformed in.() filter, so this returns
    early. Unhealthy, matching the previous `bool(entries) and all(...)`."""
    patch_api(FakeApi(funds=[]))
    report = fund_pollers._poller_report(
        poller="jp", tickers=["NOPE"], stale_after_days=4, today=TODAY,
    )
    assert report["funds_total"] == 0
    assert report["healthy"] is False
    assert report["funds"] == []
