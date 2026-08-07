"""Backend endpoint for validating a source a user wants to add.

WHY THIS LIVES HERE AND NOT IN THE FRONTEND
Deciding whether a URL can be used means checking robots.txt with the right user
agent, fetching through the SSRF-guarded client, detecting paywalls and anti-bot
walls, and parsing feeds. All of that already exists in Python and none of it
exists in the Next app. Reimplementing it there would produce a second, subtly
different set of rules — and the rule that matters is "what will the pipeline
actually be allowed to do", which only this code can answer.

So the frontend asks, and stores the answer. The pipeline re-checks robots at
fetch time anyway: permission can be withdrawn after a source is added, and a
stored row is a preference, not a standing entitlement.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from auth import SANDBOX_USER_ID, resolve_user_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/briefing", tags=["briefing"])


class SourceCheckRequest(BaseModel):
    url: str = Field(..., max_length=2048)


@router.post("/check-source")
def check_source(payload: SourceCheckRequest, user_id: str = Depends(resolve_user_id)) -> dict:
    """Can this URL be used as a briefing source?

    Returns the resolved feed when one is found — pasting a bare domain should
    just work — or a plain-language reason it cannot be used. Never raises for a
    bad URL: "no" is a normal answer here, not an error.
    """
    if user_id == SANDBOX_USER_ID:
        return {"ok": False, "reason": "Sign in to add sources."}

    from briefing.discovery import discover

    try:
        candidate = discover(payload.url)
    except Exception as exc:  # noqa: BLE001 — a bad URL must not 500
        logger.warning("source discovery failed for %r: %s", payload.url, exc)
        return {"ok": False, "reason": "Could not check that address."}
    return candidate.as_dict()
