"""Deciding who is due a briefing right now.

THE SHAPE OF THE SCHEDULED JOB
The timer fires hourly and asks "who is due?", rather than firing once at a fixed
time. Three reasons:

* users pick their own delivery time and timezone, and one fixed cron time cannot
  serve both;
* a missed hour (host asleep, deploy in progress, generation failed) is picked up
  on the next tick instead of being lost until tomorrow;
* it makes the job idempotent by construction — "due" means *no briefing exists
  for the user's local date yet*, so running it five times in an hour produces at
  most one briefing.

The database's ``UNIQUE (user_id, briefing_date)`` is still the real guarantee.
This is the cheap check that avoids doing the work; the constraint is what makes
a race harmless.
"""

from __future__ import annotations

import datetime as dt
import logging
from zoneinfo import ZoneInfo

from briefing import config

logger = logging.getLogger(__name__)


def zone(timezone: str | None) -> ZoneInfo:
    """Resolve a timezone, falling back rather than raising.

    Every path that converts a time goes through here. The first version guarded
    only the "read the clock now" path, so passing an explicit time with a bad
    zone still raised — and because ``due_users`` loops over everyone, ONE user
    with a malformed timezone would have taken down the whole scheduled run.
    """
    try:
        return ZoneInfo(timezone or config.DEFAULT_TIMEZONE)
    except Exception:  # noqa: BLE001
        logger.warning("unknown timezone %r; using %s", timezone, config.DEFAULT_TIMEZONE)
        return ZoneInfo(config.DEFAULT_TIMEZONE)


def local_now(timezone: str) -> dt.datetime:
    return dt.datetime.now(zone(timezone))


def parse_time(value: str) -> dt.time:
    try:
        hour, minute = (int(p) for p in str(value)[:5].split(":", 1))
        return dt.time(hour, minute)
    except (ValueError, TypeError):
        return dt.time(6, 30)


def is_due(prefs: dict, *, existing_dates: set[str], now: dt.datetime | None = None) -> bool:
    """Should this user get a briefing on this tick?

    Due when the user's local wall clock has passed their delivery time and no
    briefing exists for their local date.

    Note what is deliberately absent: any "did we already try recently" state.
    The only thing consulted is whether the briefing EXISTS. A failed attempt
    therefore retries on the next tick, which is what you want from a job that
    depends on other people's web servers.
    """
    timezone = prefs.get("timezone")
    current = now.astimezone(zone(timezone)) if now else local_now(timezone)
    if current.timetz().replace(tzinfo=None) < parse_time(prefs.get("deliver_at", "06:30")):
        return False
    return current.date().isoformat() not in existing_dates


def due_users(repo, *, now: dt.datetime | None = None) -> list[dict]:
    """Enabled users who are due, each annotated with the local date to generate."""
    due: list[dict] = []
    for prefs in repo.list_enabled_prefs():
        # Contained per user: one malformed preference row, or one lookup that
        # errors, must not decide that nobody else gets a briefing today.
        try:
            current = now.astimezone(zone(prefs.get("timezone"))) if now \
                else local_now(prefs.get("timezone"))
            today = current.date()
            existing = repo.get_briefing(prefs["user_id"], today)
            # A row that exists but failed is not a briefing; let it be retried.
            dates = ({today.isoformat()}
                     if existing and existing.get("status") == "ready" else set())
            if is_due(prefs, existing_dates=dates, now=now):
                due.append({**prefs, "briefing_date": today})
        except Exception:  # noqa: BLE001
            logger.exception("could not evaluate schedule for %s", prefs.get("user_id"))
    return due
