"""Storage I/O for uploaded documents (the ``user-docs`` bucket).

Reuses the Supabase-Storage S3 client from :mod:`services.storage_sync` (privileged
credentials, so it bypasses the browser-facing RLS). Layout mirrors the frontend
upload path: ``<user_id>/<doc_id>/original`` (raw bytes) and ``.../extracted.txt``.
All functions are blocking (boto3 is synchronous) — callers offload with
``asyncio.to_thread``.
"""

import os

import httpx

from services.storage_sync import build_s3_client
from services.secrets import secret as read_secret

DOCS_BUCKET = os.environ.get("DOCS_BUCKET", "user-docs")

SUPABASE_URL_ENV = "SUPABASE_URL"
SERVICE_ROLE_ENV = "SUPABASE_SERVICE_ROLE_KEY"
METADATA_TIMEOUT_S = 15.0


def _key(user_id: str, doc_id: str, name: str) -> str:
    return f"{user_id}/{doc_id}/{name}"


def fetch_content_types(user_id: str, doc_ids: list[str]) -> dict[str, str]:
    """Map ``doc_id -> content_type`` for the caller's attached documents.

    Read from the ``documents`` TABLE rather than sniffing bytes on purpose: the
    alternative is downloading every attachment just to discover which ones are
    images, which would pull whole PDFs across the network for nothing.

    Scoped by ``user_id`` as well as id, so a forged document id cannot reveal
    another user's metadata. Best-effort — any failure returns ``{}``, which makes
    the caller treat the turn as image-free and fall back to today's text path.
    """
    if not doc_ids:
        return {}
    url = (os.environ.get(SUPABASE_URL_ENV) or "").rstrip("/")
    key = read_secret(SERVICE_ROLE_ENV)
    if not url or not key:
        return {}
    try:
        with httpx.Client(timeout=METADATA_TIMEOUT_S) as c:
            r = c.get(
                f"{url}/rest/v1/documents",
                params={
                    "select": "id,content_type",
                    "user_id": f"eq.{user_id}",
                    "id": f"in.({','.join(doc_ids)})",
                },
                headers={"apikey": key, "Authorization": f"Bearer {key}"},
            )
        r.raise_for_status()
        return {
            row["id"]: (row.get("content_type") or "")
            for row in r.json()
            if row.get("id")
        }
    except Exception:
        return {}


def fetch_original(user_id: str, doc_id: str) -> bytes:
    """Read the uploaded original bytes. Raises if the object is missing."""
    client = build_s3_client()
    obj = client.get_object(Bucket=DOCS_BUCKET, Key=_key(user_id, doc_id, "original"))
    return obj["Body"].read()


def store_extracted(user_id: str, doc_id: str, text: str) -> None:
    """Persist the extracted plain text alongside the original."""
    client = build_s3_client()
    client.put_object(
        Bucket=DOCS_BUCKET,
        Key=_key(user_id, doc_id, "extracted.txt"),
        Body=text.encode("utf-8"),
        ContentType="text/plain; charset=utf-8",
    )


def fetch_extracted(user_id: str, doc_id: str) -> str:
    """Read the extracted text for a doc, or "" if absent/unreadable."""
    client = build_s3_client()
    try:
        obj = client.get_object(
            Bucket=DOCS_BUCKET, Key=_key(user_id, doc_id, "extracted.txt")
        )
        return obj["Body"].read().decode("utf-8", errors="replace")
    except Exception:
        return ""
