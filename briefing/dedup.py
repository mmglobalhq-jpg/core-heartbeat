"""Collapse the same story reported by several outlets, then rank what is left.

Two stories about one event must not both appear in a Top 5 — that wastes a slot
and makes the briefing look padded. Exact-URL dedup catches the easy case; the
hard case is five outlets covering one event under five different headlines,
which needs semantic comparison.

Embeddings come from the local ``nomic-embed-text`` model. If it is unreachable,
clustering degrades to title-token overlap rather than failing: a briefing with
imperfect dedup is worth far more than no briefing.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import re
from collections import defaultdict

from briefing import config
from briefing.models import RawItem, ScoredItem, url_hash

logger = logging.getLogger(__name__)


def dedupe_exact(items: list[RawItem]) -> list[RawItem]:
    """One item per normalised URL. Keeps the first, which is the freshest
    because ``discover`` sorted by recency."""
    seen: set[str] = set()
    unique: list[RawItem] = []
    for item in items:
        key = item.hash
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


# --- similarity --------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9']+")
_STOP = frozenset(
    "the a an and or but of to in on for with at by from as is are was were be been "
    "it its this that these those has have had will would can could new says say said "
    "after before over under more most about".split()
)


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if w not in _STOP and len(w) > 2}


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# Tokens this common across the batch carry no signal about WHICH story a headline
# is about. Without this filter, headlines that share only boilerplate — a section
# name, a wire-service prefix, a stock phrase — score as identical, and the whole
# day's news collapses into a single cluster. That is the expensive failure: a
# merged cluster loses every story in it but one.
_BOILERPLATE_DF = 0.30
_MIN_DISTINCTIVE_SHARED = 2

_MIN_BATCH_FOR_DF = 10
"""Below this, document frequency says nothing. Over two headlines every shared
token has df = 1.0 and would be discarded as boilerplate — which inverts the
filter into "two stories that share wording are unrelated" and stops the same
story from ever clustering. Small batches use plain overlap instead."""


def document_frequency(titles: list[str]) -> dict[str, float]:
    """Fraction of titles each token appears in.

    Empty for small batches, which makes every token distinctive and reduces
    ``title_similarity`` to plain Jaccard overlap.
    """
    if len(titles) < _MIN_BATCH_FOR_DF:
        return {}
    counts: dict[str, int] = defaultdict(int)
    for title in titles:
        for token in _tokens(title):
            counts[token] += 1
    return {token: n / len(titles) for token, n in counts.items()}


def distinctive_tokens(title: str, df: dict[str, float]) -> set[str]:
    """Tokens that actually distinguish this headline from the rest of the batch."""
    return {t for t in _tokens(title) if df.get(t, 0.0) <= _BOILERPLATE_DF}


def title_similarity(a: str, b: str, df: dict[str, float]) -> float:
    """Overlap on distinctive tokens only.

    Requires a minimum number of shared distinctive tokens as well as a ratio:
    two headlines sharing one rare word are not the same story, however short
    they are.
    """
    ta, tb = distinctive_tokens(a, df), distinctive_tokens(b, df)
    if not ta or not tb:
        return 0.0
    shared = ta & tb
    if len(shared) < _MIN_DISTINCTIVE_SHARED:
        return 0.0
    return len(shared) / len(ta | tb)


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def embed_all(items: list[RawItem]) -> dict[str, list[float]]:
    """Embed each item's title+summary locally. Empty dict if unavailable."""
    from briefing.llm import embed

    vectors: dict[str, list[float]] = {}
    for item in items:
        text = f"{item.title}. {item.summary or ''}".strip()
        try:
            vector = embed(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("embedding unavailable (%s); falling back to token overlap", exc)
            return {}
        if vector:
            vectors[item.hash] = vector
    return vectors


def cluster(items: list[RawItem]) -> list[list[RawItem]]:
    """Group items that describe the same story.

    Greedy single-pass assignment against cluster representatives. Not optimal
    clustering, but the input is a few hundred items and the failure mode of a
    greedy pass — occasionally leaving two near-duplicates apart — is much milder
    than the failure mode of an aggressive one, which is dropping a real story.
    """
    items = dedupe_exact(items)
    vectors = embed_all(items)
    use_embeddings = len(vectors) == len(items) and bool(vectors)
    if not use_embeddings:
        logger.info("clustering on title overlap (embeddings unavailable or partial)")

    # Computed over the whole batch, so "what counts as boilerplate" is relative
    # to today's headlines rather than a hardcoded list that rots.
    df = {} if use_embeddings else document_frequency([i.title for i in items])

    clusters: list[list[RawItem]] = []
    reps: list[RawItem] = []
    for item in items:
        placed = False
        for index, rep in enumerate(reps):
            if use_embeddings:
                score = cosine(vectors[rep.hash], vectors[item.hash])
                threshold = config.DEDUP_THRESHOLD
            else:
                # Token overlap and cosine similarity are on different scales, so
                # the fallback gets its own threshold rather than a fudge factor
                # applied to the embedding one.
                score = title_similarity(rep.title, item.title, df)
                threshold = config.DEDUP_FALLBACK_THRESHOLD
            if score >= threshold:
                clusters[index].append(item)
                placed = True
                break
        if not placed:
            clusters.append([item])
            reps.append(item)
    return clusters


# --- ranking -----------------------------------------------------------------


TOPIC_BOOST = 1.6
"""Multiplier for a story matching one of the user's topics.

Chosen so a topic match reliably beats a same-day general story but cannot beat
strong corroboration outright — a story carried by five outlets still competes.
The point is to steer the list, not to hand it over."""


MIN_TOPIC_FRACTION = 0.5
"""How much of a multi-word topic must appear before it counts at all.

FALLBACK PATH ONLY. When embeddings are available, `topic_relevance` requires a
FULL token match instead (see `_token_boost`), because the semantic path already
covers partial and synonym matches far better and partial token matching brings
false positives with it — "UGA football" scored 1.3x on a World Cup story and a
Ted Lasso newsletter purely on the word "football".

Retained at 0.5 for the offline path, where a weak signal beats none: a two-word
topic fires on either word, a three-word topic needs two of three. So "commercial
real estate" still does NOT fire on a story whose only overlap is "real", which
was the original and correct concern."""

# --- semantic topic relevance ------------------------------------------------
# Calibrated against the live 126-item corpus on 2026-08-09, NOT guessed. The
# numbers that mattered, cosine(topic, headline) with nomic-embed-text:
#
#   "Financial Markets"  top 0.624 (ETF industry), 0.619, 0.570 · median 0.409
#   "Mortgages"          top 0.533 · median 0.387 — but 0.511 was "the Bayeux
#                        Tapestry LOAN", a plausible-sounding wrong answer
#   "UGA football"       top 0.464 (World Cup soccer) · median 0.333
#
# Two lessons are baked into the constants below.
#
# A PERCENTILE ALONE IS WRONG. Every topic has a top-scoring item, so a purely
# relative rule promotes the least-bad match even when the corpus contains
# nothing on the subject — it would have surfaced World Cup soccer as "the most
# UGA-football-like story today". A topic nothing covers must promote NOTHING.
#
# AN ABSOLUTE FLOOR ALONE IS ALSO WRONG, on a day when the whole corpus leans one
# way. Hence both, and the floor sits above UGA's best false match (0.464) and
# below the genuine finance matches.
EMBED_TOPIC_FLOOR = 0.52
"""Below this, a story is not on the topic however well it ranks relatively."""

EMBED_TOPIC_FULL = 0.62
"""At or above this, the full TOPIC_BOOST applies; between the two it scales."""

EMBED_TOPIC_MARGIN = 0.08
"""Also required over the batch median, so a uniformly on-theme day does not
boost everything and thereby boost nothing."""


def _stem(word: str) -> str:
    """Fold a trailing plural, so "mortgages" and "mortgage" are one token.

    Deliberately crude, and applied to BOTH the topic and the article, so the only
    real risk is two genuinely different words folding together — never a missed
    match. A proper stemmer is not worth a dependency for this.

    MEASURED, not assumed: on the live 2026-08-09 corpus of 127 items, "mortgage"
    appeared in 2 stories and "mortgages" in 0, so the topic "Mortgages" scored
    exactly zero against content that was plainly about it.
    """
    if len(word) <= 3 or word.endswith("ss"):
        return word                       # business, press, class, gas
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"            # companies -> company
    for suffix in ("ches", "shes", "sses", "xes", "zes"):
        if word.endswith(suffix) and len(word) > len(suffix):
            return word[:-2]              # watches -> watch, taxes -> tax
    if word.endswith("s"):
        return word[:-1]                  # markets -> market
    return word


def _topic_tokens(text: str) -> set[str]:
    """Tokens for topic matching.

    Two characters minimum, not three. The clustering tokenizer drops anything
    shorter than three characters, which is right for comparing headlines and
    wrong here: it reduces the topic "AI" to the empty set, and an empty set is a
    subset of everything — so "AI" would have matched every story in the briefing
    rather than none.

    Stopwords are removed BEFORE stemming, so the stopword list stays readable.
    """
    return {_stem(w) for w in _WORD.findall((text or "").lower())
            if w not in _STOP and len(w) > 1}


def topic_fraction(item: RawItem, topics: list[str] | None) -> float:
    """The best fraction of any one topic's words present in this story.

    Matching is on whole words in the headline and summary. Substring matching
    would fire "AI" on "said", "maintain" and "Dubai" — not a subtle failure, but
    most of the corpus.
    """
    if not topics:
        return 0.0
    haystack = _topic_tokens(f"{item.title} {item.summary or ''}")
    if not haystack:
        return 0.0
    best = 0.0
    for raw in topics:
        wanted = _topic_tokens(raw or "")
        # `wanted` must be non-empty: a topic of pure stopwords would otherwise
        # divide by zero, and previously boosted every story via the subset rule.
        if not wanted:
            continue
        best = max(best, len(wanted & haystack) / len(wanted))
    return best


def topic_boost(item: RawItem, topics: list[str] | None) -> float:
    """How much this story matches what the user asked for.

    WHY THIS IS NOT ALL-OR-NOTHING ANY MORE (changed 2026-08-09):
    it used to require EVERY significant word of a topic to appear. The reasoning
    was sound — don't fire "commercial real estate" on every story containing
    "real" — but nothing had ever measured it against real headlines, and against
    the live corpus it made the whole feature inert. For the owner's actual
    topics, over 127 items that day:

        "Financial Markets"  0/127 — 1 story had "financial", 1 had "markets",
                                     and NO story had both
        "Mortgages"          0/127 — plural; now fixed by `_stem`
        "UGA football"       0/127 — genuinely uncovered by any default feed

    Two words co-occurring in one short RSS summary is close to unsatisfiable, so
    the rule was not cautious, it was off. A partial match now earns a
    proportionally smaller boost, which says what we actually mean: more overlap,
    more relevance. Below `MIN_TOPIC_FRACTION` it still counts for nothing.

    This is the same failure shape as the dedup threshold in doc 30 §6a #4 — a
    justified-sounding constant that no one had measured where it mattered.
    """
    fraction = topic_fraction(item, topics)
    if fraction < MIN_TOPIC_FRACTION:
        return 1.0
    return 1.0 + (TOPIC_BOOST - 1.0) * fraction


def _exact_token_boost(item: RawItem, topics: list[str] | None) -> float:
    """Full-token-match boost, used when the semantic path is available.

    Precise rather than generous: every significant word of some topic must be
    present. It exists alongside the embedding score because the two fail in
    opposite directions and the union is better than either. Measured example —
    "Trump revives effort to fire Fed's Lisa Cook" contains "mortgage" outright,
    so tokens give it the full boost, while its cosine to "Mortgages" was only
    0.525 and would have earned it almost nothing.
    """
    return TOPIC_BOOST if topic_fraction(item, topics) >= 1.0 else 1.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def topic_relevance(items: list[RawItem], topics: list[str] | None,
                    *, vectors: dict[str, list[float]] | None = None
                    ) -> dict[str, float]:
    """Boost multiplier per item hash, semantic where possible.

    Returns ``{}`` when there is nothing to say, which callers read as "use the
    token fallback". Degrades exactly like ``cluster``: if the embedding model is
    unreachable or embeds only part of the batch, this returns ``{}`` and ranking
    falls back to token matching rather than half-applying a semantic score.

    Deterministic: embeddings are deterministic for a given model, and the batch
    median is computed from the same inputs. This matters — ranking
    nondeterminism has already caused two identical inputs to produce different
    briefings once (§7 #5 of the briefing doc), and that must not come back.
    """
    if not topics or not items:
        return {}
    if vectors is None:
        vectors = embed_all(items)
    if len(vectors) != len(items) or not vectors:
        logger.info("topic relevance on tokens only (embeddings unavailable or partial)")
        return {}

    from briefing.llm import embed

    boosts: dict[str, float] = {}
    for raw in topics:
        text = (raw or "").strip()
        if not text:
            continue
        try:
            topic_vector = embed(text)
        except Exception as exc:  # noqa: BLE001 - any failure means fall back
            logger.warning("could not embed topic %r (%s); skipping it", text, exc)
            continue
        if not topic_vector:
            continue
        scores = {i.hash: cosine(topic_vector, vectors[i.hash]) for i in items}
        cutoff = max(EMBED_TOPIC_FLOOR, _median(list(scores.values())) + EMBED_TOPIC_MARGIN)
        span = max(EMBED_TOPIC_FULL - cutoff, 1e-6)
        for item_hash, score in scores.items():
            if score < cutoff:
                continue
            share = min(1.0, (score - cutoff) / span)
            boosts[item_hash] = max(boosts.get(item_hash, 1.0),
                                    1.0 + (TOPIC_BOOST - 1.0) * share)
    return boosts


def score_cluster(group: list[RawItem], *, weights: dict[str, float] | None = None,
                  topics: list[str] | None = None,
                  now: dt.datetime | None = None,
                  boosts: dict[str, float] | None = None) -> float:
    """How much does this story deserve a slot?

    Three signals: corroboration (how many outlets carried it), freshness (an
    exponential decay), and source weight. Corroboration is damped with a log so
    that a story carried by twenty aggregators cannot bury a significant story
    carried by two.

    ``now`` is passed in rather than read here. Reading the clock per item made
    two equally-fresh stories differ in the last decimal place of their score,
    which silently defeated the tie-break in ``rank`` — the ordering was then
    decided by float noise, so the same inputs produced different briefings.
    """
    weights = weights or {}
    lead = group[0]
    now = now or dt.datetime.now(dt.UTC)
    corroboration = 1.0 + math.log1p(len(group) - 1)
    age = min(item.age_hours(now) for item in group)
    freshness = 0.5 ** (age / config.RECENCY_HALF_LIFE_H)
    weight = max(weights.get(item.source_name, 1.0) for item in group)
    # `boosts` is the semantic score computed once over the whole batch. Where it
    # exists it is combined with an EXACT token match rather than replacing it:
    # embeddings catch what tokens miss ("ETF industry" for "Financial Markets",
    # cosine 0.624, no shared word) and tokens catch what embeddings underrate
    # (a story literally containing "mortgage" that scored only 0.525). Falling
    # back to the generous partial-token rule only when there is no semantic
    # score at all keeps the offline path useful without importing its false
    # positives into the normal path.
    if boosts:
        boost = max(max(boosts.get(item.hash, 1.0), _exact_token_boost(item, topics))
                    for item in group)
    else:
        boost = max(topic_boost(item, topics) for item in group)
    return (corroboration * (0.35 + 0.65 * freshness) * weight * boost
            * (1.0 if lead.readable else 0.5))


def rank(items: list[RawItem], *, weights: dict[str, float] | None = None,
         topics: list[str] | None = None,
         now: dt.datetime | None = None) -> list[ScoredItem]:
    """Cluster, score, and order. Deterministic for identical input.

    One clock reading for the whole run, so equally-fresh stories score
    identically and the ``url_hash`` tie-break actually decides their order.
    Without both halves of that, the same inputs produce two different briefings.

    Topic relevance is computed ONCE for the whole batch, not per cluster: the
    cutoff is relative to the day's median, so it is only meaningful over the
    full set.
    """
    now = now or dt.datetime.now(dt.UTC)
    boosts = topic_relevance(items, topics)
    scored = [
        ScoredItem(
            item=group[0],
            score=score_cluster(group, weights=weights, topics=topics, now=now,
                                boosts=boosts),
            cluster_id=url_hash(group[0].url),
            duplicates=group[1:],
        )
        for group in cluster(items)
    ]
    scored.sort(key=lambda s: (-s.score, url_hash(s.item.url)))
    return scored


def diversify(ranked: list[ScoredItem], *, count: int,
              max_per_source: int | None = None) -> list[ScoredItem]:
    """Take the best ``count`` stories without letting one outlet take them all.

    Two passes. The first respects the per-outlet cap; the second backfills from
    whatever was passed over, in score order, if there were not enough distinct
    outlets to fill the list. Structure wins over diversity — a briefing with
    four items would be a worse outcome than one with three items from the same
    paper.
    """
    cap = config.MAX_PER_SOURCE if max_per_source is None else max_per_source
    chosen: list[ScoredItem] = []
    passed_over: list[ScoredItem] = []
    used: dict[str, int] = defaultdict(int)

    for scored in ranked:
        source = scored.item.source_name or "unknown"
        if len(chosen) < count and used[source] < cap:
            chosen.append(scored)
            used[source] += 1
        else:
            passed_over.append(scored)

    for scored in passed_over:
        if len(chosen) >= count:
            break
        chosen.append(scored)

    return chosen[:count]


def select(items: list[RawItem], *, weights: dict[str, float] | None = None,
           top_count: int | None = None, max_per_source: int | None = None,
           topics: list[str] | None = None,
           now: dt.datetime | None = None) -> tuple[list[ScoredItem], ScoredItem | None]:
    """Pick the Top N and the Deep Dive.

    The Deep Dive is the best-corroborated story *below* the Top N, not the
    top-scoring one. Leading with the biggest story and then dwelling on it again
    reads as one story told twice; taking the deep dive from just outside the
    headline set gives the briefing a second dimension. When there is nothing
    below the cut, it falls back to the top story rather than omitting the
    section, because the structure is fixed.
    """
    ranked = rank(items, weights=weights, topics=topics, now=now)
    count = top_count if top_count is not None else config.TOP_COUNT
    top = diversify(ranked, count=count, max_per_source=max_per_source)
    chosen = {id(s) for s in top}
    remainder = [s for s in ranked if id(s) not in chosen]
    if remainder:
        deep = max(remainder, key=lambda s: (len(s.duplicates), s.score))
    else:
        deep = top[0] if top else None
    return top, deep


def cluster_sizes(items: list[RawItem]) -> dict[str, int]:
    sizes: dict[str, int] = defaultdict(int)
    for group in cluster(items):
        sizes[group[0].title] = len(group)
    return dict(sizes)
