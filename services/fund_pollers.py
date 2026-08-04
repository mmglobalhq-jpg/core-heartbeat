"""Fund-poller health for the platform heartbeat.

Neither fund poller is a long-running service — both are one-shot systemd
units. Nothing "goes down"; a poller fails by quietly not producing data, which
no liveness probe can see. This module therefore checks the *evidence a poller
leaves behind* rather than any process:

- how recent the newest ``as_of_date`` is (stale data),
- when the poller last succeeded (timer freshness),
- how many failures have run consecutively,
- how many failures are still unresolved,
- how many revisions are awaiting review.

Reads the shared funds database through PostgREST with the credentials the
gateway already holds. Read-only: only ``GET`` requests are issued. Returns
counts and dates — never holdings, never credentials.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

FUNDS_SUPABASE_URL_ENV = "FUNDS_SUPABASE_URL"
FUNDS_SERVICE_ROLE_ENV = "FUNDS_SUPABASE_SERVICE_ROLE_KEY"

SECRET_KEY_PREFIX = "sb_secret_"
"""Opaque Supabase secret keys authenticate with ``apikey`` alone; sending an
``Authorization: Bearer`` header alongside one is rejected. Legacy JWT
``service_role`` keys require both. Same split the other services handle."""

HTTP_TIMEOUT_SECONDS = 15

JP_STALE_AFTER_DAYS = 4
"""Friday data is still newest through Monday's polls; +1 day covers a holiday."""

ALLSPRING_STALE_AFTER_DAYS = 70
"""Allspring publishes monthly, in arrears, captured during a mid-month window.
Two months plus slack — anything less alarms every normal cycle."""

_FAILURE_STATUSES = (
    "partial",
    "download_failed",
    "download_invalid",
    "storage_failed",
    "parse_failed",
    "validation_failed",
    "database_failed",
)

_SUCCESS_STATUSES = ("success", "partial", "material_revision_applied")


class FundPollerError(RuntimeError):
    """Raised when poller health cannot be determined."""


def _config() -> tuple[str, str]:
    url = os.environ.get(FUNDS_SUPABASE_URL_ENV)
    key = os.environ.get(FUNDS_SERVICE_ROLE_ENV)
    if not url:
        raise FundPollerError(f"{FUNDS_SUPABASE_URL_ENV} is not set")
    if not key:
        raise FundPollerError(f"{FUNDS_SERVICE_ROLE_ENV} is not set")
    return url.rstrip("/"), key


def _get(path: str, **params: str) -> list[dict[str, Any]]:
    url, key = _config()
    query = urllib.parse.urlencode(params, safe="*.,():")
    request = urllib.request.Request(f"{url}/rest/v1/{path}?{query}", method="GET")
    request.add_header("apikey", key)
    request.add_header("Accept", "application/json")
    if not key.startswith(SECRET_KEY_PREFIX):
        request.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        payload = json.load(response)
    return payload if isinstance(payload, list) else []


def _parse_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value[:10])
    except ValueError:
        return None


def _parse_ts(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _consecutive_failures(statuses: list[str]) -> int:
    """Failures since the last attempt that proved the pipeline works.

    Healthy no-ops break the streak — ``unchanged``/``equivalent_reissue``/
    ``stale_source`` all required a successful download, parse and DB round
    trip. Skips are ignored: they carry no signal either way.
    """
    streak = 0
    for status in statuses:
        if status.startswith("skipped"):
            continue
        if status in _FAILURE_STATUSES:
            streak += 1
            continue
        break
    return streak


def _poller_report(
    *,
    poller: str,
    tickers: list[str],
    stale_after_days: int,
    today: dt.date,
) -> dict[str, Any]:
    funds = _get("funds", select="id,ticker,is_active", ticker=f"in.({','.join(tickers)})")
    by_id = {f["id"]: f["ticker"] for f in funds}
    entries: list[dict[str, Any]] = []

    for fund_id, ticker in sorted(by_id.items(), key=lambda kv: kv[1]):
        snapshots = _get(
            "fund_snapshots",
            select="as_of_date",
            fund_id=f"eq.{fund_id}",
            is_current="is.true",
            snapshot_status="eq.accepted",
            order="as_of_date.desc",
            limit="1",
        )
        last_as_of = _parse_date(snapshots[0]["as_of_date"]) if snapshots else None

        successes = _get(
            "poll_attempts",
            select="attempt_time",
            fund_id=f"eq.{fund_id}",
            status=f"in.({','.join(_SUCCESS_STATUSES)})",
            order="attempt_time.desc",
            limit="1",
        )
        last_success = _parse_ts(successes[0]["attempt_time"]) if successes else None

        unresolved = _get(
            "poll_attempts",
            select="id",
            fund_id=f"eq.{fund_id}",
            is_resolved="is.false",
            status=f"in.({','.join(_FAILURE_STATUSES)})",
            limit="500",
        )
        recent = _get(
            "poll_attempts",
            select="status",
            fund_id=f"eq.{fund_id}",
            order="attempt_time.desc",
            limit="50",
        )
        needs_review = _get(
            "poll_attempts",
            select="id",
            fund_id=f"eq.{fund_id}",
            status="eq.material_revision_applied",
            is_resolved="is.false",
            limit="200",
        )

        age_days = (today - last_as_of).days if last_as_of else None
        stale = age_days is None or age_days > stale_after_days
        streak = _consecutive_failures([r["status"] for r in recent])
        entries.append(
            {
                "ticker": ticker,
                "last_as_of_date": last_as_of.isoformat() if last_as_of else None,
                "as_of_age_days": age_days,
                "stale_data": stale,
                "last_successful_run": last_success.isoformat() if last_success else None,
                "consecutive_failures": streak,
                "unresolved_failures": len(unresolved),
                "needs_review_revisions": len(needs_review),
                "healthy": not stale and not unresolved,
            }
        )

    return {
        "poller": poller,
        "funds_total": len(entries),
        "funds_healthy": sum(1 for e in entries if e["healthy"]),
        "funds_stale": sum(1 for e in entries if e["stale_data"]),
        "unresolved_failures": sum(e["unresolved_failures"] for e in entries),
        "needs_review_revisions": sum(e["needs_review_revisions"] for e in entries),
        "max_consecutive_failures": max((e["consecutive_failures"] for e in entries), default=0),
        "stale_data": any(e["stale_data"] for e in entries),
        "healthy": bool(entries) and all(e["healthy"] for e in entries),
        "funds": entries,
    }


JP_TICKERS = ["JAGG", "JBND", "JCPB", "JCPI", "JFLX", "JMTG", "JPIE", "JPLD", "JSCP"]
ALLSPRING_TICKERS = [
    "AS_CORE_BOND",
    "AS_CORE_PLUS",
    "AS_GOVT_SEC",
    "AS_INCOME_PLUS",
    "AS_SHORT_PLUS",
    "AS_ULTRA_SHORT",
]


def fund_poller_health(*, today: dt.date | None = None) -> dict[str, Any]:
    """Report JP and Allspring poller health separately.

    Each poller is reported independently and a failure in one never masks or
    degrades the other. If a poller cannot be queried at all, it is reported as
    ``"unknown"`` rather than ``"healthy"`` — an unreachable check is not a
    passing check.
    """
    today = today or dt.datetime.now(dt.UTC).date()
    report: dict[str, Any] = {"checked_at": dt.datetime.now(dt.UTC).isoformat()}

    for name, tickers, stale_days in (
        ("jp", JP_TICKERS, JP_STALE_AFTER_DAYS),
        ("allspring", ALLSPRING_TICKERS, ALLSPRING_STALE_AFTER_DAYS),
    ):
        try:
            report[name] = _poller_report(
                poller=name, tickers=tickers, stale_after_days=stale_days, today=today
            )
        except Exception as exc:  # noqa: BLE001 — one poller must not break the other
            logger.warning("fund poller health check failed for %s: %s", name, exc)
            report[name] = {"poller": name, "healthy": None, "status": "unknown", "error": str(exc)}

    statuses = [report[n].get("healthy") for n in ("jp", "allspring")]
    report["healthy"] = all(s is True for s in statuses)
    report["degraded"] = any(s is False for s in statuses)
    report["unknown"] = any(s is None for s in statuses)
    # Nothing is wired to send an alert when this goes unhealthy, so this
    # endpoint is the only signal and must be polled externally. A Resend API key
    # does exist on the host (used by the signup-approval Edge Function) but is
    # not in this service's environment, and no systemd unit has OnFailure=.
    # See system-source-of-truth docs/26-fund-pollers.md §8.
    report["outbound_alerting"] = "none"
    return report
