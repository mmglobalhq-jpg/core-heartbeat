"""Turn ranked stories into the briefing's sections.

STRUCTURE IS NOT NEGOTIABLE
Exactly ``TOP_COUNT`` top items and exactly one Deep Dive. The database enforces
"at most one deep dive" and "no duplicate ranks"; cardinality cannot be expressed
there, so it is asserted here and in the tests. ``validate_structure`` is called
before anything is persisted — a malformed briefing fails loudly during
generation rather than arriving in someone's inbox with four items.

PROVENANCE IS NOT NEGOTIABLE EITHER
A section's URL is attached by the pipeline from the item it was built from; the
model is never asked to supply one and is instructed not to write URLs at all.
Output is then checked against the set of URLs the pipeline chose to read. That
check is what makes injected "visit this link" text harmless — see untrusted.py.
"""

from __future__ import annotations

import logging

from briefing import config
from briefing.llm import EscalationBudget, LLMUnavailable, generate
from briefing.models import ScoredItem, Section, tidy
from briefing.untrusted import UntrustedContentError, assert_allowed_urls, wrap

logger = logging.getLogger(__name__)


SUMMARY_SYSTEM = (
    "You write a short news briefing for one reader. Summarise only what the "
    "supplied material actually says. Never add facts, figures, names or "
    "conclusions that are not in it. If the material is thin, write less rather "
    "than filling the gap. Do not write URLs, links, email addresses or phone "
    "numbers. Plain prose, no markdown, no bullet points, no headings."
)

DEEP_DIVE_SYSTEM = (
    "You write the single longer piece in a daily briefing: three or four "
    "paragraphs explaining why one story matters and what is genuinely uncertain "
    "about it. Ground every claim in the supplied material. Where the material "
    "does not settle a question, say so plainly instead of guessing. Do not write "
    "URLs, links, email addresses or phone numbers. Plain prose, no markdown."
)


class CompositionError(RuntimeError):
    """The briefing could not be assembled to spec."""


def _material(scored: ScoredItem, *, limit: int) -> str:
    """The text handed to the model for one story, already contained.

    Includes corroborating headlines from the cluster so the write-up can reflect
    that several outlets carried it, without fetching all of them.
    """
    item = scored.item
    parts = [f"HEADLINE: {item.title}", f"OUTLET: {item.source_name}"]
    if item.published_at:
        parts.append(f"PUBLISHED: {item.published_at.isoformat()}")
    if item.summary:
        parts.append(f"SUMMARY: {item.summary}")
    if item.body:
        parts.append(f"ARTICLE TEXT:\n{item.body[:limit]}")
    elif item.skipped_reason:
        # Be explicit that the full text was not read, so the model works from
        # the headline and summary instead of inventing detail to fill a gap.
        parts.append(
            f"NOTE: the full article was not retrieved ({item.skipped_reason}). "
            "Only the headline and summary above are available."
        )
    if scored.duplicates:
        others = "; ".join(f"{d.source_name}: {d.title}" for d in scored.duplicates[:5])
        parts.append(f"ALSO REPORTED BY: {others}")
    return "\n".join(parts)


def _allowed_urls(scored: ScoredItem) -> set[str]:
    return {scored.item.url, *(d.url for d in scored.duplicates)}


def _write(scored: ScoredItem, *, system: str, instruction: str, limit: int,
           budget: EscalationBudget | None, prefer_hosted: bool) -> tuple[str, str]:
    """Generate prose for one story and verify it before accepting it."""
    prompt = f"{instruction}\n\n{wrap(_material(scored, limit=limit), label='NEWS MATERIAL')}"
    text, model = generate(prompt, system=system, budget=budget, prefer_hosted=prefer_hosted)
    text = tidy(text)
    if not text:
        raise CompositionError(f"empty write-up for {scored.item.url!r}")

    # The model was told not to write URLs. If one appears anyway it is either a
    # hallucination or content that came out of the fenced block, and both are
    # grounds to reject rather than to publish.
    assert_allowed_urls(text, _allowed_urls(scored))
    return text, model


def compose_top(
    top: list[ScoredItem],
    *,
    budget: EscalationBudget | None = None,
) -> tuple[list[Section], dict]:
    """Write the Top N. One local call per story."""
    sections: list[Section] = []
    meta = {"models": {}, "rejected": 0}
    for rank_index, scored in enumerate(top, start=1):
        try:
            body, model = _write(
                scored,
                system=SUMMARY_SYSTEM,
                instruction=(
                    "Summarise the story below in two or three sentences for a "
                    "reader who has not seen it. Lead with what happened."
                ),
                limit=4_000,
                budget=budget,
                prefer_hosted=False,
            )
        except (UntrustedContentError, CompositionError, LLMUnavailable) as exc:
            # Fall back to the publisher's own summary rather than dropping the
            # slot. The structure requires N items; a story with a weaker write-up
            # still informs the reader, an empty slot does not.
            logger.warning("write-up rejected for %s (%s); using source summary",
                           scored.item.url, exc)
            meta["rejected"] += 1
            body = tidy(scored.item.summary or scored.item.title, limit=400)
            model = "none"
        meta["models"][model] = meta["models"].get(model, 0) + 1
        sections.append(
            Section(
                kind="top",
                rank=rank_index,
                headline=tidy(scored.item.title, limit=160),
                body=body,
                url=scored.item.url,
                source_name=scored.item.source_name,
                published_at=scored.item.published_at,
            )
        )
    return sections, meta


def compose_deep_dive(
    scored: ScoredItem,
    *,
    budget: EscalationBudget | None = None,
) -> tuple[Section, dict]:
    """Write the single Deep Dive, preferring the hosted model.

    This is the one section where the difference between a 7B local model and a
    hosted one is visible to the reader, so it is where the budget is spent.
    """
    try:
        body, model = _write(
            scored,
            system=DEEP_DIVE_SYSTEM,
            instruction=(
                "Write the deep dive on the story below: what happened, why it "
                "matters, and what remains genuinely unclear."
            ),
            limit=8_000,
            budget=budget,
            prefer_hosted=True,
        )
    except (UntrustedContentError, CompositionError, LLMUnavailable) as exc:
        logger.warning("deep dive rejected for %s (%s); using source summary",
                       scored.item.url, exc)
        body = tidy(scored.item.summary or scored.item.title, limit=600)
        model = "none"
    section = Section(
        kind="deep_dive",
        rank=1,
        headline=tidy(scored.item.title, limit=160),
        body=body,
        url=scored.item.url,
        source_name=scored.item.source_name,
        published_at=scored.item.published_at,
    )
    return section, {"deep_dive_model": model}


def validate_structure(sections: list[Section], *, top_count: int | None = None) -> None:
    """Refuse to persist a briefing that is not the shape it claims to be."""
    expected = config.TOP_COUNT if top_count is None else top_count
    top = [s for s in sections if s.kind == "top"]
    deep = [s for s in sections if s.kind == "deep_dive"]

    if len(top) != expected:
        raise CompositionError(f"expected {expected} top items, got {len(top)}")
    if len(deep) != config.DEEP_DIVE_COUNT:
        raise CompositionError(f"expected exactly one deep dive, got {len(deep)}")
    if sorted(s.rank for s in top) != list(range(1, expected + 1)):
        raise CompositionError(f"top ranks must be 1..{expected}, got {sorted(s.rank for s in top)}")
    for section in sections:
        if not section.headline.strip():
            raise CompositionError("section with an empty headline")
        if not section.body.strip():
            raise CompositionError(f"section {section.headline!r} has an empty body")
        # Every claim in a briefing has to be traceable. A section without a URL
        # is an assertion with nothing behind it.
        if not section.url.strip():
            raise CompositionError(f"section {section.headline!r} has no source URL")
