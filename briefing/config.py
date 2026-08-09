"""Configuration for the briefing pipeline.

Every value is read from the environment with a working default, so the pipeline
runs in a development checkout with nothing configured. Names are prefixed
``BRIEFING_`` so they are greppable and so doc 08 can account for them.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

# --- structure ---------------------------------------------------------------

TOP_COUNT = int(os.environ.get("BRIEFING_TOP_COUNT", "5"))
"""Always this many. A briefing that sometimes has four items trains the reader
to wonder whether something was dropped."""

DEEP_DIVE_COUNT = 1
"""Structural, not configurable. 'Exactly one Deep Dive' is part of what the
product is; making it a knob would let a bad config produce a shapeless email."""

# --- ingestion ---------------------------------------------------------------

MAX_ITEMS_PER_SOURCE = int(os.environ.get("BRIEFING_MAX_ITEMS_PER_SOURCE", "25"))
MAX_TOTAL_ITEMS = int(os.environ.get("BRIEFING_MAX_TOTAL_ITEMS", "200"))
ITEM_MAX_AGE_HOURS = int(os.environ.get("BRIEFING_ITEM_MAX_AGE_HOURS", "36"))
"""Wider than 24h on purpose: a story published late yesterday evening belongs in
this morning's briefing, and a strict midnight cut would drop it."""

FETCH_CONCURRENCY = int(os.environ.get("BRIEFING_FETCH_CONCURRENCY", "4"))
PER_HOST_DELAY_S = float(os.environ.get("BRIEFING_PER_HOST_DELAY_S", "1.0"))
"""Politeness delay between requests to the same host. This job has no deadline
pressure; being a well-behaved client costs nothing."""

USER_AGENT = os.environ.get(
    "BRIEFING_USER_AGENT",
    "core-heartbeat-briefing/1.0 (+self-hosted personal briefing; contact via site owner)",
)
"""Identifies the crawler honestly. Never impersonate a browser to defeat bot
detection — a host that does not want automated traffic is entitled to refuse it."""

RESPECT_ROBOTS = os.environ.get("BRIEFING_RESPECT_ROBOTS", "1") != "0"

# --- ranking -----------------------------------------------------------------

DEDUP_THRESHOLD = float(os.environ.get("BRIEFING_DEDUP_THRESHOLD", "0.62"))
"""Cosine similarity above which two stories are the same story.

MEASURED, NOT GUESSED. The first production briefing carried the July jobs report
twice — slots 1 and 5, two outlets, one event — because this was 0.86, which is
ABOVE the score real duplicates get. It could never merge anything.

Scored with nomic-embed-text on that briefing's own headlines:

    same story, two outlets ............ 0.792
    jobs report vs defence pact ........ 0.383
    jobs report vs senate nomination ... 0.332
    defence pact vs senate nomination .. 0.398

So duplicates land near 0.79 and unrelated pairs near 0.33-0.40, leaving an empty
band between. 0.62 sits in the middle: 0.17 of margin before it misses a
duplicate, 0.22 before it merges two distinct stories. Token overlap on that same
pair was 0.14 — useless, which is why the fallback has its own threshold rather
than sharing this one.

The original 0.86 came with a comment justifying it as deliberately cautious.
That reasoning was sound and the number was still wrong: nothing had measured
where real duplicates actually score."""

MAX_PER_SOURCE = int(os.environ.get("BRIEFING_MAX_PER_SOURCE", "2"))
"""Ceiling on Top-N slots one outlet may hold.

Pure score ranking gave a real run five BBC items out of five: the outlet with
the most items in the pool wins every slot, because nothing in the score pushes
back. That is one outlet's front page, not a briefing. Relaxed rather than
enforced when there are not enough distinct outlets to fill the list — the fixed
structure outranks the diversity preference."""

DEDUP_FALLBACK_THRESHOLD = float(os.environ.get("BRIEFING_DEDUP_FALLBACK_THRESHOLD", "0.5"))
"""Threshold for the no-embeddings path, which scores overlap of *distinctive*
title tokens. Separate from DEDUP_THRESHOLD because token overlap and cosine
similarity are not the same scale — sharing one threshold made the fallback merge
every headline in the batch into one cluster."""

RECENCY_HALF_LIFE_H = float(os.environ.get("BRIEFING_RECENCY_HALF_LIFE_H", "12"))

TOPIC_RESERVED_SLOTS = int(os.environ.get("BRIEFING_TOPIC_RESERVED_SLOTS", "1"))
"""Top-N slots guaranteed to a topic the list does not already cover.

Ranking is dominated by corroboration, and a niche interest is niche precisely
because one outlet covers it — a 1-outlet story needs ~2.9x to reach fifth place,
while TOPIC_BOOST is capped at 2.4 on purpose. A reserved slot promotes one story
without inflating what everything else scores. `0` disables it. See
`dedup.reserve_topic_slots` for why the slot prefers an UNREPRESENTED topic."""

TOPIC_BOOST = float(os.environ.get("BRIEFING_TOPIC_BOOST", "2.4"))
"""Score multiplier for a story matching a user's topic. See `dedup.TOPIC_BOOST`
for the measurement this was set from, and why it is provisional. Overridable by
environment so it can be retuned from `run_meta.topic_calibration` without a
rebuild."""

# --- models ------------------------------------------------------------------

LOCAL_MODEL = os.environ.get("BRIEFING_LOCAL_MODEL", "qwen2.5:7b")
EMBED_MODEL = os.environ.get("BRIEFING_EMBED_MODEL", "nomic-embed-text")


def _ollama_base(raw: str) -> str:
    """Reduce OLLAMA_URL to scheme://host:port.

    THIS PLATFORM SETS OLLAMA_URL TO A FULL ENDPOINT, not a base — compose sets
    ``http://host.docker.internal:11434/api/generate``, matching what
    orchestrator.py posts to directly. This module needs several paths
    (/api/generate, /api/embeddings, /api/tags), so it appends its own.

    Treating the configured value as a base produced
    ``/api/generate/api/tags`` — a 404 that read as "Ollama is unreachable". The
    first production run degraded every write-up to the publisher's summary
    because of it, while reporting only a warning.
    """
    parts = urlsplit(raw.strip())
    if parts.scheme and parts.netloc:
        return f"{parts.scheme}://{parts.netloc}"
    return raw.strip().rstrip("/")


OLLAMA_URL = _ollama_base(os.environ.get("OLLAMA_URL") or "http://127.0.0.1:11434")

ESCALATION_MODEL = os.environ.get("BRIEFING_ESCALATION_MODEL", "gemini-2.5-flash")
MAX_ESCALATIONS = int(os.environ.get("BRIEFING_MAX_ESCALATIONS", "3"))
"""Hard ceiling on hosted-model calls per briefing. The Deep Dive and the
editorial pass each want one; the third is headroom for a single retry. Exceeding
this raises rather than silently spending."""

LLM_TIMEOUT_S = float(os.environ.get("BRIEFING_LLM_TIMEOUT_S", "120"))

# --- delivery ----------------------------------------------------------------

DEFAULT_TIMEZONE = os.environ.get("BRIEFING_TIMEZONE", "America/Chicago")
"""Matches the platform default set on 2026-08-05."""

LEAD_MINUTES = int(os.environ.get("BRIEFING_LEAD_MINUTES", "6"))
"""Start generating this many minutes BEFORE the user's delivery time.

`deliver_at` is what the user reads as "when the briefing arrives", but the job
only ever asked whether that time had already *passed* — so the work started at
the target and the email landed however long generation took after it. Measured
on 2026-08-08: `deliver_at` 06:30, delivered 07:13.

Generation took 3m56s and 6m06s on the two real runs before this was written, so
6 minutes lands the email on the target rather than after it. Combined with a
5-minute timer tick the arrival window is [deliver_at, deliver_at + ~5min]: the
earliest a tick can fire is `deliver_at - LEAD`, which finishes no sooner than
the target, so a briefing is never delivered EARLY.

Raising this above the tick interval is what makes early delivery possible —
that is the knob to be careful with, not the lower bound."""

EMAIL_PROVIDER = os.environ.get("BRIEFING_EMAIL_PROVIDER", "file")
"""'file' writes rendered output to disk; 'resend' sends for real. Defaults to
'file' so that a misconfigured environment cannot email anyone by accident."""

EMAIL_OUTPUT_DIR = os.environ.get("BRIEFING_EMAIL_OUTPUT_DIR", "/tmp/briefing-out")
EMAIL_FROM = os.environ.get("BRIEFING_EMAIL_FROM", "briefing@localhost")
