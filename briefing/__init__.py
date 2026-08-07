"""Daily Briefing: one canonical briefing per user per day.

A scheduled job, not a request handler. It ingests public web content, clusters
and ranks it, writes up the Top 5 plus exactly one Deep Dive, and delivers the
result.

The three rules that shape most of the code in here:

1. **Fetched content is data, never instruction.** Everything retrieved from the
   web passes through ``untrusted`` before it reaches a model, and every URL in
   the finished briefing must trace back to a source the pipeline chose to read.
   See ``untrusted.py``.

2. **The briefing is an immutable snapshot.** Sections carry their own copy of
   the headline, body and URL. Source pages get edited and retracted; a briefing
   that changed afterwards would be a record of nothing.

3. **Local model first.** Classification, dedup and summarisation run on the
   local Ollama model. Escalation to a hosted model is bounded per briefing and
   counted, so the cost of a briefing is knowable rather than emergent.
"""

from briefing.models import (  # noqa: F401
    BriefingDraft,
    RawItem,
    ScoredItem,
    Section,
    SourceSpec,
)

__all__ = ["BriefingDraft", "RawItem", "ScoredItem", "Section", "SourceSpec"]
