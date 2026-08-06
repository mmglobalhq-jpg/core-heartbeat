"""Storage for write actions proposed to the user and awaiting their "yes".

The confirmation handshake spans two HTTP requests: one turn proposes a batch of
calendar writes, the next releases it. Holding that plan in process memory made two
failure modes possible, and both were hit:

* a gateway restart between the turns lost the plan, and the user — who had just
  been shown a list and said yes — was told it no longer existed;
* the payload carried no chat id, so plans were keyed by user alone and two
  conversations shared one slot. Approving in one could release the other's writes.

So a plan is persisted against ``(user_id, chat_id)`` in the core Supabase project,
alongside the chat it belongs to.

**Failures fall back to memory rather than breaking the turn.** If the table is
missing (migration not yet applied) or Supabase is unreachable, this degrades to
the previous in-process behaviour instead of refusing to propose anything. That
keeps deploy ordering safe — the code can ship before the migration — at the cost
of the durability guarantee until the table exists.

Rows describe pending *writes* to a user's calendar, so the table is service-role
only with RLS forced (migration 0007). Nothing here is reachable from a browser.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import threading
import time

import httpx

logger = logging.getLogger(__name__)

SUPABASE_URL_ENV = "SUPABASE_URL"
SERVICE_ROLE_ENV = "SUPABASE_SERVICE_ROLE_KEY"
TABLE = "pending_plans"
REQUEST_TIMEOUT_S = 10.0

# How long an unconfirmed proposal stays valid. Long enough to read a schedule and
# think about it; short enough that a forgotten plan cannot be released by an
# unrelated "yes" hours later.
PLAN_TTL_S = int(os.environ.get("PENDING_PLAN_TTL_S", "1800"))

# Fallback when the table or the network is unavailable. Same semantics as the
# durable path, minus the durability.
_memory: dict[tuple[str, str], tuple[float, list[dict]]] = {}
_memory_lock = threading.Lock()

# Test seam: unit tests set an httpx.MockTransport here. None -> real network.
_transport: httpx.BaseTransport | None = None


def _config() -> tuple[str, str] | None:
    url = (os.environ.get(SUPABASE_URL_ENV) or "").rstrip("/")
    key = os.environ.get(SERVICE_ROLE_ENV)
    return (url, key) if url and key else None


def _client(url: str, key: str) -> httpx.Client:
    return httpx.Client(
        base_url=f"{url}/rest/v1",
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        timeout=REQUEST_TIMEOUT_S,
        transport=_transport,
    )


def _key(user_id: str, chat_id: str | None) -> tuple[str, str]:
    """A plan belongs to a conversation. Without a chat id every conversation for a
    user collapses onto one slot, which is the bug this module exists to fix — so
    the fallback key is explicit rather than silently shared."""
    return (user_id, chat_id or "_nochat")


# --- in-memory fallback -----------------------------------------------------


def _mem_save(user_id: str, chat_id: str | None, calls: list[dict]) -> None:
    with _memory_lock:
        _memory[_key(user_id, chat_id)] = (time.monotonic() + PLAN_TTL_S, list(calls))


def _mem_take(user_id: str, chat_id: str | None) -> list[dict] | None:
    with _memory_lock:
        entry = _memory.pop(_key(user_id, chat_id), None)
    if entry is None:
        return None
    expires_at, calls = entry
    return calls if time.monotonic() < expires_at else None


def _mem_clear(user_id: str, chat_id: str | None) -> None:
    with _memory_lock:
        _memory.pop(_key(user_id, chat_id), None)


# --- public API -------------------------------------------------------------


def save(user_id: str, chat_id: str | None, calls: list[dict]) -> str:
    """Persist a proposed plan. Returns "db" or "memory" (which backend took it)."""
    _mem_save(user_id, chat_id, calls)  # always, so a DB failure still has a copy
    cfg = _config()
    if not cfg:
        return "memory"
    url, key = cfg
    expires = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=PLAN_TTL_S)
    row = {
        "user_id": user_id,
        "chat_id": chat_id or "_nochat",
        "calls": calls,
        "expires_at": expires.isoformat(),
    }
    try:
        with _client(url, key) as c:
            r = c.post(
                f"/{TABLE}",
                json=row,
                headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
            )
        r.raise_for_status()
        return "db"
    except Exception as exc:
        logger.warning("pending plan not persisted (%s); using memory", exc)
        return "memory"


def take(user_id: str, chat_id: str | None) -> tuple[list[dict] | None, str]:
    """Pop the plan awaiting approval. Returns ``(calls_or_None, source)``.

    Single-use: the row is deleted as it is read, so a second "yes" cannot replay
    the same writes. The in-memory copy is cleared either way.
    """
    cfg = _config()
    if cfg:
        url, key = cfg
        try:
            with _client(url, key) as c:
                r = c.delete(
                    f"/{TABLE}",
                    params={
                        "user_id": f"eq.{user_id}",
                        "chat_id": f"eq.{chat_id or '_nochat'}",
                    },
                    headers={"Prefer": "return=representation"},
                )
            r.raise_for_status()
            rows = r.json() if r.content else []
            _mem_clear(user_id, chat_id)
            if rows:
                row = rows[0]
                if _expired(row.get("expires_at")):
                    return None, "db-expired"
                calls = row.get("calls") or []
                return (list(calls) or None), "db"
            return None, "db-miss"
        except Exception as exc:
            logger.warning("pending plan lookup failed (%s); trying memory", exc)
    return _mem_take(user_id, chat_id), "memory"


def clear(user_id: str, chat_id: str | None) -> None:
    """Discard any proposal — the user asked for something else instead."""
    _mem_clear(user_id, chat_id)
    cfg = _config()
    if not cfg:
        return
    url, key = cfg
    try:
        with _client(url, key) as c:
            c.delete(
                f"/{TABLE}",
                params={
                    "user_id": f"eq.{user_id}",
                    "chat_id": f"eq.{chat_id or '_nochat'}",
                },
                headers={"Prefer": "return=minimal"},
            )
    except Exception as exc:  # a stale row expires on its own; never fail the turn
        logger.warning("pending plan clear failed (%s)", exc)


def _expired(value: str | None) -> bool:
    if not value:
        return False
    try:
        when = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return False
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    return dt.datetime.now(dt.timezone.utc) >= when
