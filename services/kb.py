"""Knowledge-base (Graph-RAG) gateway helpers.

Server-to-server calls from the core-heartbeat gateway to the internal Graph-RAG
service: INGEST and document management for the library, and RETRIEVAL for the
Knowledge chat agent (knowledge_chat.py). The gateway is the trust boundary: it has already verified the
caller's JWT (resolve_user_id) and — for global writes — checks the profiles.is_admin
flag here before telling the KB service to stamp owner=global.

owner is either a user_id (private) or the literal "global" (admin-ingested), sent as
the X-User-Id header the KB service scopes on.
"""
from __future__ import annotations

import asyncio
import base64
import os

import httpx
from services.secrets import secret as read_secret

GRAPHRAG_URL_ENV = "GRAPHRAG_SERVICE_URL"
GRAPHRAG_KEY_ENV = "GRAPHRAG_API_KEY"
SUPABASE_URL_ENV = "SUPABASE_URL"
SERVICE_ROLE_ENV = "SUPABASE_SERVICE_ROLE_KEY"

TIMEOUT_S = 30.0
GLOBAL_OWNER = "global"
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_S = 0.4


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
    key = read_secret(SERVICE_ROLE_ENV)
    if not (url and key):
        return False
    headers = {"apikey": key, "Authorization": f"Bearer {key}", "Accept": "application/json"}
    async with httpx.AsyncClient(base_url=url.rstrip("/"), headers=headers, timeout=10.0) as c:
        r = await c.get("/rest/v1/profiles", params={"id": f"eq.{user_id}", "select": "is_admin"})
        r.raise_for_status()
        rows = r.json()
    return bool(rows and rows[0].get("is_admin"))


async def ingest(owner: str, filename: str, content: bytes, replaces_document_id: str | None = None) -> dict:
    """Forward document bytes to the KB service's async ingest. Returns {job_id, status}.

    ``replaces_document_id`` makes it a replacement: the service deletes that document
    (same scope only) after the new version has fully ingested.
    """
    payload = {"file": base64.b64encode(content).decode("ascii"), "filename": filename}
    if replaces_document_id:
        payload["replaces_document_id"] = replaces_document_id
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


async def delete_document(doc_id: str, owner: str) -> bool:
    """Delete a doc in the given scope. False if it wasn't found in that scope (404)."""
    async with httpx.AsyncClient(base_url=_kb_base(), headers=_kb_headers(owner), timeout=TIMEOUT_S) as c:
        r = await c.delete(f"/api/documents/{doc_id}")
        if r.status_code == 404:
            return False
        r.raise_for_status()
        return True


# --- retrieval (Knowledge chat) ------------------------------------------------------


class KbNotFound(KbError):
    """The service answered 404 — e.g. a document reference matched nothing."""

    def __init__(self, payload: dict):
        super().__init__("not found")
        self.payload = payload


async def _request(method: str, path: str, owner: str, json_body: dict | None = None) -> dict:
    """One KB call, retrying transport failures and 5xx. A 404 raises KbNotFound with
    the service's body (it names unresolved documents and alternatives)."""
    last: Exception | None = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            async with httpx.AsyncClient(base_url=_kb_base(), headers=_kb_headers(owner), timeout=TIMEOUT_S) as c:
                r = await c.request(method, path, json=json_body)
        except httpx.TransportError as exc:
            last = exc
        else:
            if r.status_code == 404:
                try:
                    body = r.json()
                except ValueError:
                    body = {}
                raise KbNotFound(body)
            if r.status_code < 500:
                r.raise_for_status()
                return r.json()
            last = httpx.HTTPStatusError(f"server error {r.status_code}", request=r.request, response=r)
        if attempt + 1 < RETRY_ATTEMPTS:
            await asyncio.sleep(RETRY_BACKOFF_S * (attempt + 1))
    assert last is not None
    raise last


async def search(
    owner: str,
    query: str,
    *,
    top_k: int = 8,
    document_titles: list[str] | None = None,
    include_parent_context: bool = True,
) -> dict:
    """Ranked passages for ``query`` (own + global), optionally scoped to documents
    named the way a person would name them. Raises KbNotFound if a name matches nothing."""
    options: dict = {"top_k": top_k, "include_parent_context": include_parent_context}
    if document_titles:
        options["document_titles"] = document_titles
    return await _request("POST", "/api/query", owner, {"query": query, "options": options})


async def read_document(owner: str, document: str, *, focus: str | None = None, max_chunks: int = 12) -> dict:
    """One named document: its ingest-time summary plus passages sampled across it (or
    the passages most relevant to ``focus``). Raises KbNotFound if nothing matches."""
    body: dict = {"document_title": document, "max_chunks": max_chunks}
    if focus:
        body["focus"] = focus
    return await _request("POST", "/api/summarize", owner, body)
