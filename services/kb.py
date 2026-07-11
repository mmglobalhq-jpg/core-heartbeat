"""Knowledge-base (Graph-RAG) gateway helpers.

Server-to-server calls from the core-heartbeat gateway to the internal Graph-RAG
service for INGEST + document management (the query path is the orchestrator tool,
tools/graphrag.py). The gateway is the trust boundary: it has already verified the
caller's JWT (resolve_user_id) and — for global writes — checks the profiles.is_admin
flag here before telling the KB service to stamp owner=global.

owner is either a user_id (private) or the literal "global" (admin-ingested), sent as
the X-User-Id header the KB service scopes on.
"""
from __future__ import annotations

import base64
import os

import httpx

GRAPHRAG_URL_ENV = "GRAPHRAG_SERVICE_URL"
GRAPHRAG_KEY_ENV = "GRAPHRAG_API_KEY"
SUPABASE_URL_ENV = "SUPABASE_URL"
SERVICE_ROLE_ENV = "SUPABASE_SERVICE_ROLE_KEY"

TIMEOUT_S = 30.0
GLOBAL_OWNER = "global"


class KbError(Exception):
    """Raised when the KB service / profiles lookup is unreachable or misconfigured."""


def _kb_base() -> str:
    url = os.environ.get(GRAPHRAG_URL_ENV)
    if not url:
        raise KbError(f"{GRAPHRAG_URL_ENV} is not set")
    return url.rstrip("/")


def _kb_headers(owner: str) -> dict[str, str]:
    key = os.environ.get(GRAPHRAG_KEY_ENV)
    if not key:
        raise KbError(f"{GRAPHRAG_KEY_ENV} is not set")
    return {
        "Authorization": f"Bearer {key}",
        "X-User-Id": owner,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


async def is_admin(user_id: str) -> bool:
    """True iff profiles.is_admin is set for this user (service-role read; fail-closed)."""
    url = os.environ.get(SUPABASE_URL_ENV)
    key = os.environ.get(SERVICE_ROLE_ENV)
    if not (url and key):
        return False
    headers = {"apikey": key, "Authorization": f"Bearer {key}", "Accept": "application/json"}
    async with httpx.AsyncClient(base_url=url.rstrip("/"), headers=headers, timeout=10.0) as c:
        r = await c.get("/rest/v1/profiles", params={"id": f"eq.{user_id}", "select": "is_admin"})
        r.raise_for_status()
        rows = r.json()
    return bool(rows and rows[0].get("is_admin"))


async def ingest(owner: str, filename: str, content: bytes) -> dict:
    """Forward document bytes to the KB service's async ingest. Returns {job_id, status}."""
    payload = {"file": base64.b64encode(content).decode("ascii"), "filename": filename}
    async with httpx.AsyncClient(base_url=_kb_base(), headers=_kb_headers(owner), timeout=TIMEOUT_S) as c:
        r = await c.post("/api/ingest", json=payload)
        r.raise_for_status()
        return r.json()


async def get_job(job_id: str, owner: str) -> dict:
    async with httpx.AsyncClient(base_url=_kb_base(), headers=_kb_headers(owner), timeout=TIMEOUT_S) as c:
        r = await c.get(f"/api/jobs/{job_id}")
        r.raise_for_status()
        return r.json()


async def list_documents(owner: str) -> dict:
    async with httpx.AsyncClient(base_url=_kb_base(), headers=_kb_headers(owner), timeout=TIMEOUT_S) as c:
        r = await c.get("/api/documents")
        r.raise_for_status()
        return r.json()
