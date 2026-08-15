"""REIT research reports published since the last briefing.

The research engine writes its own reports on its own schedule, and until now the
briefing had no idea they existed — the owner had to go and look. This surfaces a
bullet per new report, titled with the report's own headline and linked to it.

REUSES THE READER CONTRACT, DOES NOT REIMPLEMENT IT
``tools.reit_research`` already holds the RPC access: the two Supabase server-key
generations, the format-aware headers, the injectable transport and the issuer
name table. All of that is deployed and tested. This module calls it rather than
opening a second path to the same data — a second path is a second thing to get
wrong when the key generation next rotates.

NOTHING HERE MAY FAIL A BRIEFING
Like ``market``, every entry point degrades to an empty list. A research service
being unreachable is not a reason for the reader to get no news.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
from dataclasses import dataclass
from urllib.parse import quote

logger = logging.getLogger(__name__)

ENABLED = os.environ.get("BRIEFING_REIT_REPORTS", "1") not in ("0", "false", "False", "")

# Where the reader UI lives. The /reits page deep-links a report through
# ?issuer=<code>&report=<report_id>, which is what makes a title clickable.
APP_BASE_URL = os.environ.get("BRIEFING_APP_BASE_URL", "https://chat.mmglobal.us").rstrip("/")

# How far back counts as "new". A day would miss a report published while
# yesterday's briefing was already being assembled; three days is forgiving
# without dredging up last week's.
LOOKBACK_DAYS = int(os.environ.get("BRIEFING_REPORT_LOOKBACK_DAYS", "3"))

# Per issuer. Reports are published at most a few times a month, so this only has
# to cover a burst, not a backlog.
PER_ISSUER_LIMIT = 5


@dataclass(frozen=True)
class ReportLink:
    issuer_code: str
    issuer_name: str
    title: str
    url: str
    published_on: dt.date


def _report_url(issuer_code: str, report_id: str) -> str:
    return (
        f"{APP_BASE_URL}/reits"
        f"?issuer={quote(issuer_code, safe='')}&report={quote(report_id, safe='')}"
    )


def recent_reports(since: dt.date | None = None) -> list[ReportLink]:
    """Reports published on or after ``since``, newest first. Never raises."""
    if not ENABLED:
        return []
    cutoff = since or (dt.date.today() - dt.timedelta(days=LOOKBACK_DAYS))
    try:
        from tools.reit_research import (
            _RPC_LIST_ISSUERS,
            _RPC_LIST_REPORTS,
            _issuer_name,
            _rpc,
            _title_for,
        )
    except Exception as exc:
        logger.warning("reit_research unavailable: %s", type(exc).__name__)
        return []

    try:
        issuers = _rpc(_RPC_LIST_ISSUERS, {})
    except Exception as exc:
        logger.warning("could not list REIT issuers: %s", type(exc).__name__)
        return []

    found: list[ReportLink] = []
    for issuer in issuers:
        code = (issuer.get("issuer_code") or "").strip()
        if not code:
            continue
        name = issuer.get("issuer_name") or _issuer_name(code)
        try:
            rows = _rpc(_RPC_LIST_REPORTS, {"p_issuer_code": code, "p_limit": PER_ISSUER_LIMIT})
        except Exception as exc:
            logger.warning("could not list reports for %s: %s", code, type(exc).__name__)
            continue
        for row in rows:
            published = row.get("publication_date")
            report_id = row.get("report_id")
            if not published or not report_id:
                # A report with no publication date is a draft as far as this is
                # concerned. Guessing from portfolio_as_of_date would surface
                # unpublished work in an email.
                continue
            try:
                published_on = dt.date.fromisoformat(str(published))
            except ValueError:
                continue
            if published_on < cutoff:
                continue
            found.append(
                ReportLink(
                    issuer_code=code,
                    issuer_name=name,
                    title=_title_for(name, row.get("title"), row.get("portfolio_as_of_date")),
                    url=_report_url(code, str(report_id)),
                    published_on=published_on,
                )
            )

    # Newest first, then by issuer so a same-day pair is stably ordered — the
    # ranking pipeline has already been bitten once by a non-deterministic sort.
    found.sort(key=lambda r: (-r.published_on.toordinal(), r.issuer_code))
    return found


def reports_meta(reports: list[ReportLink]) -> dict:
    return {
        "count": len(reports),
        "issuers": sorted({r.issuer_code for r in reports}),
        "latest": reports[0].published_on.isoformat() if reports else None,
    }
