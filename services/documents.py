"""Storage I/O for uploaded documents (the ``user-docs`` bucket).

Reuses the Supabase-Storage S3 client from :mod:`services.storage_sync` (privileged
credentials, so it bypasses the browser-facing RLS). Layout mirrors the frontend
upload path: ``<user_id>/<doc_id>/original`` (raw bytes) and ``.../extracted.txt``.
All functions are blocking (boto3 is synchronous) — callers offload with
``asyncio.to_thread``.
"""

import os

from services.storage_sync import build_s3_client

DOCS_BUCKET = os.environ.get("DOCS_BUCKET", "user-docs")


def _key(user_id: str, doc_id: str, name: str) -> str:
    return f"{user_id}/{doc_id}/{name}"


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
