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

DEDUP_THRESHOLD = float(os.environ.get("BRIEFING_DEDUP_THRESHOLD", "0.86"))
"""Cosine similarity above which two stories are the same story. Tuned high:
merging two distinct stories loses one entirely, while failing to merge shows a
near-duplicate, which is the cheaper mistake."""

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

EMAIL_PROVIDER = os.environ.get("BRIEFING_EMAIL_PROVIDER", "file")
"""'file' writes rendered output to disk; 'resend' sends for real. Defaults to
'file' so that a misconfigured environment cannot email anyone by accident."""

EMAIL_OUTPUT_DIR = os.environ.get("BRIEFING_EMAIL_OUTPUT_DIR", "/tmp/briefing-out")
EMAIL_FROM = os.environ.get("BRIEFING_EMAIL_FROM", "briefing@localhost")
