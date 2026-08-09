"""The briefing job: discover, rank, enrich, write, edit, store, deliver.

Run it:

    python -m briefing.run --user <uuid> --dry-run
    python -m briefing.run --user <uuid> --repo postgres --deliver file

ORDER MATTERS. Ranking happens on titles and summaries BEFORE full text is
fetched, so the run makes a handful of article requests rather than a few hundred.
Structure is validated BEFORE anything is persisted, so a malformed briefing fails
in the job rather than in someone's inbox.

SCHEDULED IN PRODUCTION since 2026-08-07: ``core-briefing.timer`` fires hourly and
runs ``--due``, which generates for every enabled user past their own delivery
time in their own timezone. See system-source-of-truth doc 30.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from zoneinfo import ZoneInfo

from briefing import config
from briefing.compose import compose_deep_dive, compose_top, validate_structure
from briefing.dedup import select, topic_boost
from briefing.delivery import sender_for
from briefing.editorial import polish_sections
from briefing.ingest import discover, enrich
from briefing.llm import EscalationBudget, local_available
from briefing.models import BriefingDraft, SourceSpec
from briefing.render import render_all
from briefing.sources import DEFAULT_SOURCES, sources_for

logger = logging.getLogger("briefing.run")


def local_date(timezone: str) -> dt.date:
    """Today, in the reader's zone.

    A briefing is 'for' a local day. Using UTC would roll the date over at 6pm
    Chicago time and produce two briefings on one calendar day, or none — the
    unique constraint is on the LOCAL date, so getting this wrong is not cosmetic.
    """
    try:
        zone = ZoneInfo(timezone)
    except Exception:  # noqa: BLE001
        logger.warning("unknown timezone %r; falling back to %s", timezone, config.DEFAULT_TIMEZONE)
        zone = ZoneInfo(config.DEFAULT_TIMEZONE)
    return dt.datetime.now(zone).date()


def build_briefing(
    user_id: str,
    *,
    sources: tuple[SourceSpec, ...] | None = None,
    topics: list[str] | None = None,
    user_sources: list[dict] | None = None,
    timezone: str = config.DEFAULT_TIMEZONE,
    top_count: int | None = None,
    do_editorial: bool = True,
) -> BriefingDraft:
    """Produce a briefing in memory. No persistence, no delivery.

    ``topics`` steers RANKING, not ingestion — no permissible news-search source
    exists (see ``sources.sources_for``), so a topic promotes matching stories
    within what the default feeds already carry. A topic nothing covers still
    yields nothing. Passing ``sources`` explicitly overrides the default list;
    that is the test seam.
    """
    if sources is None:
        sources = sources_for(topics, user_sources)
    budget = EscalationBudget()
    meta: dict = {"timezone": timezone, "started_at": dt.datetime.now(dt.UTC).isoformat()}

    logger.info("discovering from %d sources", len(sources))
    items, discover_stats = discover(list(sources))
    meta.update(discover_stats)
    if not items:
        raise RuntimeError("no items discovered — every source failed or returned nothing")

    weights = {s.name: s.weight for s in sources}
    count = top_count if top_count is not None else config.TOP_COUNT
    calibration: dict = {}
    top, deep = select(items, weights=weights, top_count=count, topics=topics,
                       calibration=calibration)
    if calibration:
        meta["topic_calibration"] = calibration

    # Record whether each topic actually matched anything. Topics were wired
    # through ranking correctly and still had zero effect for months, and nothing
    # in run_meta would have shown it — the run reported sources, clusters,
    # escalations and editorial rejections, but never this. A topic sitting at 0
    # is now visible in the stored briefing instead of silent.
    if topics:
        meta["topics"] = list(topics)
        meta["topics_matched"] = {
            t: sum(1 for i in items if topic_boost(i, [t]) > 1.0) for t in topics
        }
    if len(top) < count:
        raise RuntimeError(
            f"only {len(top)} distinct stories after clustering; need {count}"
        )
    meta["clusters"] = len(top) + (1 if deep else 0)

    logger.info("enriching %d selected stories", len(top) + (1 if deep else 0))
    chosen = [s.item for s in top] + ([deep.item] if deep and deep not in top else [])
    _, enrich_stats = enrich(chosen)
    meta.update(enrich_stats)

    logger.info("composing")
    sections, top_meta = compose_top(top, budget=budget)
    meta["compose"] = top_meta
    if deep is not None:
        deep_section, deep_meta = compose_deep_dive(deep, budget=budget)
        sections.append(deep_section)
        meta.update(deep_meta)

    if do_editorial:
        logger.info("editorial pass")
        sections, editorial_meta = polish_sections(sections, budget=budget)
        meta["editorial"] = editorial_meta

    # Before persistence, not after.
    validate_structure(sections, top_count=count)

    meta["escalations_used"] = budget.spent
    meta["escalation_limit"] = budget.limit
    meta["finished_at"] = dt.datetime.now(dt.UTC).isoformat()

    return BriefingDraft(
        user_id=user_id,
        briefing_date=local_date(timezone),
        sections=sections,
        run_meta=meta,
    )


def run_due(repo, *, deliver: str, dry_run: bool = False) -> dict:
    """Generate for everyone who is due. The scheduled entry point.

    One user's failure never stops the others: a briefing is per-user work, and a
    feed that breaks one person's run has nothing to do with anyone else's.
    """
    from briefing.delivery import sender_for
    from briefing.schedule import due_users

    summary = {"due": 0, "generated": 0, "failed": 0, "delivered": 0, "errors": []}
    pending = due_users(repo)
    summary["due"] = len(pending)
    if not pending:
        logger.info("no users due")
        return summary

    for prefs in pending:
        user_id = prefs["user_id"]
        try:
            # Loaded per run, not cached: a user can add or remove a source
            # between ticks and the next briefing reflects it. Failure here is
            # non-fatal because custom sources are additive — losing them gives a
            # thinner briefing, not none.
            try:
                user_sources = repo.list_user_sources(user_id)
            except Exception as exc:  # noqa: BLE001 — custom sources are additive
                logger.warning("could not load custom sources for %s (%s); "
                               "using defaults", user_id, exc)
                user_sources = []
            draft = build_briefing(
                user_id,
                topics=prefs.get("topics") or [],
                user_sources=user_sources,
                timezone=prefs.get("timezone") or config.DEFAULT_TIMEZONE,
            )
            if dry_run:
                logger.info("[dry run] would store and deliver for %s", user_id)
                summary["generated"] += 1
                continue

            briefing_id = repo.upsert_briefing(draft)
            repo.replace_sections(briefing_id, draft.sections)
            summary["generated"] += 1
            logger.info("stored briefing %s for %s", briefing_id, user_id)

            # Email only if the user asked for it AND gave an address. Defaulting
            # to "send" on a half-configured preference would mail people who
            # never opted in.
            if prefs.get("deliver_email") and prefs.get("email_to"):
                rendered = render_all(draft)
                result = sender_for(deliver).send(
                    to=prefs["email_to"],
                    subject=rendered["subject"],
                    html=rendered["html"],
                    text=rendered["text"],
                )
                repo.record_delivery(briefing_id, "email", result.status,
                                     result.provider, result.detail)
                if result.status == "sent":
                    summary["delivered"] += 1
                else:
                    logger.warning("delivery for %s: %s", user_id, result)
        except Exception as exc:  # noqa: BLE001
            summary["failed"] += 1
            summary["errors"].append(f"{user_id}: {type(exc).__name__}: {exc}")
            logger.exception("briefing failed for %s", user_id)

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a daily briefing.")
    parser.add_argument("--user", help="user uuid (omit with --due)")
    parser.add_argument("--due", action="store_true",
                        help="generate for every enabled user who is due (scheduled mode)")
    parser.add_argument("--repo", choices=("postgres", "postgrest", "none"), default="none",
                        help="where to persist; 'none' keeps it in memory")
    parser.add_argument("--deliver", choices=("file", "resend", "none"), default="file")
    parser.add_argument("--to", default="", help="recipient (file sender uses it for the filename)")
    parser.add_argument("--timezone", default=config.DEFAULT_TIMEZONE)
    parser.add_argument("--top", type=int, default=config.TOP_COUNT)
    parser.add_argument("--topic", action="append", default=[], dest="topics",
                        help="add a topic search (repeatable); --due reads them per user")
    parser.add_argument("--no-editorial", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="build and render only; never persist or deliver")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    if not local_available():
        logger.warning("local model unreachable at %s — composition will degrade",
                       config.OLLAMA_URL)

    if args.due:
        if args.repo == "none":
            parser.error("--due needs a repository (--repo postgrest or postgres)")
        from briefing.repository import PostgresRepository, PostgrestRepository

        repo = PostgresRepository() if args.repo == "postgres" else PostgrestRepository()
        summary = run_due(repo, deliver=args.deliver, dry_run=args.dry_run)
        print(json.dumps(summary, indent=2, default=str))
        # Non-zero on any failure so systemd's OnFailure= alerting fires. A job
        # that swallows per-user errors and exits 0 is a job nobody hears about.
        return 1 if summary["failed"] else 0

    if not args.user:
        parser.error("--user is required unless --due is given")

    draft = build_briefing(
        args.user,
        topics=args.topics,
        timezone=args.timezone,
        top_count=args.top,
        do_editorial=not args.no_editorial,
    )
    rendered = render_all(draft)

    print(f"\n=== {rendered['subject']} ===\n")
    print(rendered["text"])

    if args.dry_run:
        print("\n[dry run] nothing persisted, nothing delivered")
        print(json.dumps(draft.run_meta, indent=2, default=str))
        return 0

    briefing_id = None
    if args.repo != "none":
        from briefing.repository import PostgresRepository, PostgrestRepository

        repo = PostgresRepository() if args.repo == "postgres" else PostgrestRepository()
        briefing_id = repo.upsert_briefing(draft)
        repo.replace_sections(briefing_id, draft.sections)
        logger.info("stored briefing %s", briefing_id)

    if args.deliver != "none":
        sender = sender_for(args.deliver)
        result = sender.send(
            to=args.to or f"{args.user}@local",
            subject=rendered["subject"],
            html=rendered["html"],
            text=rendered["text"],
        )
        logger.info("delivery: %s", result)
        if briefing_id and args.repo != "none":
            repo.record_delivery(briefing_id, "email" if args.deliver == "resend" else "file",
                                 result.status, result.provider, result.detail)

    print(json.dumps(draft.run_meta, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
