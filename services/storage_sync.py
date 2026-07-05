"""Per-user Markdown vault sync for core-heartbeat (Phase 2 of multi-user memory).

Before a run streams, the orchestrator localizes the caller's context by pulling
their vault — the ``.md`` notes stored under ``<user_id>/`` in the Supabase
Storage ``user-vaults`` bucket — down to ``/tmp/vaults/<user_id>/``. The
LangGraph supervisor and workers then read from that local directory instead of
reaching across the network mid-run.

Two paths, selected by the resolved identity:

  * A real Supabase user id → download over the S3-compatible Storage endpoint
    with boto3. boto3 is synchronous, so the blocking list/download work runs in
    a worker thread (``asyncio.to_thread``) to keep ``sync_user_vault`` awaitable
    and non-blocking for the event loop.
  * :data:`~auth.SANDBOX_USER_ID` → read from a local mock folder instead of
    hitting S3, so isolated/offline testing and local dev never need credentials
    or network. See :func:`_sync_from_mock`.

Design notes mirroring the rest of the codebase:
  * All configuration is read from the environment at call time (env-overridable
    without a rebuild), exactly like the Ollama/provider settings in
    ``orchestrator``.
  * ``boto3`` is imported lazily inside :func:`build_s3_client` (the OpenAI /
    Anthropic lazy-import pattern), so importing this module — and the whole
    sandbox path — works with boto3 absent. :func:`build_s3_client` is also the
    monkeypatch seam tests use to avoid real network (cf. ``build_ollama_client``).
  * Each sync is a clean snapshot: the destination is cleared first so a deleted
    remote note does not linger locally.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

from auth import SANDBOX_USER_ID

# --- configuration (all env-overridable, read at call time) -----------------

# Supabase Storage bucket holding per-user vaults; keyed by a "<user_id>/" prefix.
VAULT_BUCKET_ENV = "SUPABASE_VAULT_BUCKET"
DEFAULT_VAULT_BUCKET = "user-vaults"

# S3-compatible endpoint + static credentials (Dashboard -> Storage -> S3 keys).
S3_ENDPOINT_ENV = "SUPABASE_S3_ENDPOINT"
S3_ACCESS_KEY_ID_ENV = "SUPABASE_S3_ACCESS_KEY_ID"
S3_SECRET_ACCESS_KEY_ENV = "SUPABASE_S3_SECRET_ACCESS_KEY"
# Region is required by the S3 client; default matches the Core project region.
S3_REGION_ENV = "SUPABASE_S3_REGION"
DEFAULT_S3_REGION = "us-east-2"

# Where synced vaults land locally, and where the sandbox mock reads from.
VAULT_SYNC_ROOT_ENV = "VAULT_SYNC_ROOT"
DEFAULT_VAULT_SYNC_ROOT = "/tmp/vaults"
VAULT_MOCK_ROOT_ENV = "VAULT_MOCK_ROOT"
# Default mock root is repo-relative (this file lives in <repo>/services/), so it
# resolves the same regardless of the process working directory.
DEFAULT_VAULT_MOCK_ROOT = str(Path(__file__).resolve().parent.parent / "mock_vaults")

VAULT_SUFFIX = ".md"
# Locally-generated files that are NOT remote notes and must survive a sync's
# clean-snapshot wipe (e.g. the memory_extractor's cross-session user profile).
# Without this, each streaming turn's pre-run sync would delete the profile before
# the Supervisor could read it back.
PRESERVED_LOCAL_FILES = ("user_preferences.md",)


def _vault_bucket() -> str:
    return os.environ.get(VAULT_BUCKET_ENV) or DEFAULT_VAULT_BUCKET


def _sync_root() -> str:
    return os.environ.get(VAULT_SYNC_ROOT_ENV) or DEFAULT_VAULT_SYNC_ROOT


def _mock_root() -> str:
    return os.environ.get(VAULT_MOCK_ROOT_ENV) or DEFAULT_VAULT_MOCK_ROOT


def _s3_region() -> str:
    return os.environ.get(S3_REGION_ENV) or DEFAULT_S3_REGION


def _reset_dest(user_id: str) -> str:
    """Return (and freshly (re)create) the local destination for ``user_id``.

    The directory is cleared first so each sync is a clean snapshot — a note
    deleted remotely does not survive locally — EXCEPT for locally-generated files
    in :data:`PRESERVED_LOCAL_FILES` (the user profile), which are carried across
    the wipe so cross-session memory is not destroyed on every pre-run sync.
    """
    dest = os.path.join(_sync_root(), user_id)
    preserved: dict[str, bytes] = {}
    for name in PRESERVED_LOCAL_FILES:
        path = os.path.join(dest, name)
        if os.path.isfile(path):
            with open(path, "rb") as fh:
                preserved[name] = fh.read()
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest, exist_ok=True)
    for name, data in preserved.items():
        with open(os.path.join(dest, name), "wb") as fh:
            fh.write(data)
    return dest


# --- S3 path (real users) ---------------------------------------------------

def build_s3_client():
    """Construct the boto3 S3 client bound to the Supabase Storage endpoint.

    boto3 is imported lazily here (like the OpenAI/Anthropic SDKs in
    ``orchestrator``) so importing this module never requires boto3, and this
    function is the single seam tests monkeypatch to avoid real network (cf.
    ``orchestrator.build_ollama_client``).
    """
    import boto3  # lazy: keep the module importable / sandbox path boto3-free

    return boto3.client(
        "s3",
        endpoint_url=os.environ.get(S3_ENDPOINT_ENV),
        aws_access_key_id=os.environ.get(S3_ACCESS_KEY_ID_ENV),
        aws_secret_access_key=os.environ.get(S3_SECRET_ACCESS_KEY_ENV),
        region_name=_s3_region(),
    )


def _iter_object_keys(client, bucket: str, prefix: str):
    """Yield every object key under ``prefix``, following pagination.

    Uses a manual ContinuationToken loop rather than a paginator so the fake
    client in tests only needs to implement ``list_objects_v2``.
    """
    token: str | None = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = client.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            key = obj.get("Key")
            if key:
                yield key
        if not resp.get("IsTruncated"):
            return
        token = resp.get("NextContinuationToken")


def _sync_from_s3(user_id: str, dest: str) -> str:
    """Download every ``.md`` object under ``<user_id>/`` into ``dest`` (blocking).

    Runs in a worker thread (see :func:`sync_user_vault`); boto3 is synchronous.
    Non-Markdown objects and the bare prefix "folder" placeholder are skipped,
    and nested key structure under the prefix is preserved on disk.
    """
    client = build_s3_client()
    bucket = _vault_bucket()
    prefix = f"{user_id}/"
    for key in _iter_object_keys(client, bucket, prefix):
        if not key.endswith(VAULT_SUFFIX):
            continue
        rel = key[len(prefix):] if key.startswith(prefix) else key
        if not rel:  # the prefix itself (a "directory" marker), not a file
            continue
        local_path = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        client.download_file(bucket, key, local_path)
    return dest


def upload_user_file(user_id: str, filename: str) -> None:
    """Upload one local vault file back to the ``user-vaults`` bucket (blocking).

    Write-back for durability: mirrors ``/tmp/vaults/<user_id>/<filename>`` to the
    S3 key ``<user_id>/<filename>`` so a preference profile survives a container
    restart (a fresh container's :func:`sync_user_vault` then downloads it back).
    Runs in a worker thread (boto3 is synchronous). No-op for the sandbox user
    (offline mock, no S3) and when the local file is absent.
    """
    if user_id == SANDBOX_USER_ID:
        return
    local_path = os.path.join(_sync_root(), user_id, filename)
    if not os.path.isfile(local_path):
        return
    client = build_s3_client()
    client.upload_file(local_path, _vault_bucket(), f"{user_id}/{filename}")


# --- sandbox path (local mock, no S3) ---------------------------------------

def _sync_from_mock(user_id: str, dest: str) -> str:
    """Copy the user's ``.md`` files from the local mock folder into ``dest``.

    Reads ``<mock_root>/<user_id>/`` and mirrors its Markdown tree, so isolated
    testing and offline dev exercise the same downstream code path as a real S3
    sync without any credentials or network. A missing mock folder is not an
    error — it yields an empty (already-created) vault.
    """
    src = Path(_mock_root()) / user_id
    if not src.is_dir():
        return dest
    for path in src.rglob(f"*{VAULT_SUFFIX}"):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        local_path = os.path.join(dest, str(rel))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        shutil.copyfile(path, local_path)
    return dest


# --- public API -------------------------------------------------------------

async def sync_user_vault(user_id: str) -> str:
    """Localize ``user_id``'s Markdown vault and return the local directory path.

    Downloads (real user) or copies (sandbox user) every ``.md`` note for the
    caller into ``<VAULT_SYNC_ROOT>/<user_id>/`` (default ``/tmp/vaults/<user_id>/``)
    and returns that path. The sandbox user reads a local mock folder instead of
    hitting S3, so offline/isolated runs need no credentials. The blocking boto3
    work runs in a worker thread so this stays non-blocking on the event loop.
    """
    dest = _reset_dest(user_id)
    if user_id == SANDBOX_USER_ID:
        return _sync_from_mock(user_id, dest)
    return await asyncio.to_thread(_sync_from_s3, user_id, dest)
