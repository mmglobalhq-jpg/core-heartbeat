"""The briefing job: discover, rank, enrich, write, edit, store, deliver.

Run it:

    python -m briefing.run --user <uuid> --dry-run
    python -m briefing.run --user <uuid> --repo postgres --deliver file

ORDER MATTERS. Ranking happens on titles and summaries BEFORE full text is
fetched, so the run makes a handful of article requests rather than a few hundred.
Structure is validated BEFORE anything is persisted, so a malformed briefing fails
in the job rather than in someone's inbox.

NOTHING HERE IS SCHEDULED. There is no systemd unit and no timer for this in
production; the job is invoked by hand. Wiring it to a schedule is a production
change and is deliberately not part of this build.
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
from briefing.dedup import select
from briefing.delivery import sender_for
from briefing.editorial import polish_sections
from briefing.ingest import discover, enrich
from briefing.llm import EscalationBudget, local_available
from briefing.models import BriefingDraft, SourceSpec
from briefing.render import render_all
from briefing.sources import DEFAULT_SOURCES

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
    sources: tuple[SourceSpec, ...] = DEFAULT_SOURCES,
    timezone: str = config.DEFAULT_TIMEZONE,
    top_count: int | None = None,
    do_editorial: bool = True,
) -> BriefingDraft:
    """Produce a briefing in memory. No persistence, no delivery."""
    budget = EscalationBudget()
    meta: dict = {"timezone": timezone, "started_at": dt.datetime.now(dt.UTC).isoformat()}

    logger.info("discovering from %d sources", len(sources))
    items, discover_stats = discover(list(sources))
    meta.update(discover_stats)
    if not items:
        raise RuntimeError("no items discovered — every source failed or returned nothing")

    weights = {s.name: s.weight for s in sources}
    count = top_count if top_count is not None else config.TOP_COUNT
    top, deep = select(items, weights=weights, top_count=count)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a daily briefing.")
    parser.add_argument("--user", required=True, help="user uuid")
    parser.add_argument("--repo", choices=("postgres", "postgrest", "none"), default="none",
                        help="where to persist; 'none' keeps it in memory")
    parser.add_argument("--deliver", choices=("file", "resend", "none"), default="file")
    parser.add_argument("--to", default="", help="recipient (file sender uses it for the filename)")
    parser.add_argument("--timezone", default=config.DEFAULT_TIMEZONE)
    parser.add_argument("--top", type=int, default=config.TOP_COUNT)
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

    draft = build_briefing(
        args.user,
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
