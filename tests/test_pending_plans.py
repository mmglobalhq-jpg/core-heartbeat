"""Persistence for plans awaiting confirmation.

Two production failures motivated moving this out of process memory:

* a gateway restart between the proposal and the user's "yes" lost the plan, and
  someone who had just been shown a list and approved it was told it no longer
  existed;
* the payload carried no chat id, so plans were keyed by user alone. Two
  conversations shared one slot, meaning a "yes" in one chat could release the
  calendar writes proposed in another.

The rows describe pending WRITES to a real calendar, so the tests below care about
isolation and single-use as much as about round-tripping.
"""

import datetime as dt
import json

import httpx
import pytest

from services import pending_plans


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    pending_plans._memory.clear()
    pending_plans._transport = None
    yield
    pending_plans._memory.clear()
    pending_plans._transport = None


CALLS = [{"name": "create_calendar_event", "args": {"summary": "Game 1"}}]


# --- fallback (no Supabase configured) --------------------------------------


def test_round_trips_without_any_backend_configured():
    """Deploy ordering: the code must work before migration 0007 is applied."""
    assert pending_plans.save("u1", "chat-a", CALLS) == "memory"
    calls, source = pending_plans.take("u1", "chat-a")
    assert calls == CALLS and source == "memory"


def test_a_plan_is_single_use():
    pending_plans.save("u1", "chat-a", CALLS)
    assert pending_plans.take("u1", "chat-a")[0] == CALLS
    assert pending_plans.take("u1", "chat-a")[0] is None


def test_two_chats_do_not_share_a_slot():
    """The isolation bug: approving in one conversation must not release another's
    writes."""
    pending_plans.save("u1", "chat-a", CALLS)
    other = [{"name": "delete_calendar_event", "args": {"event_id": "x"}}]
    pending_plans.save("u1", "chat-b", other)

    assert pending_plans.take("u1", "chat-a")[0] == CALLS
    assert pending_plans.take("u1", "chat-b")[0] == other


def test_two_users_do_not_share_a_slot():
    pending_plans.save("u1", "chat-a", CALLS)
    assert pending_plans.take("u2", "chat-a")[0] is None
    assert pending_plans.take("u1", "chat-a")[0] == CALLS


def test_clear_discards_the_plan():
    pending_plans.save("u1", "chat-a", CALLS)
    pending_plans.clear("u1", "chat-a")
    assert pending_plans.take("u1", "chat-a")[0] is None


def test_a_missing_chat_id_does_not_collide_with_a_real_one():
    pending_plans.save("u1", None, CALLS)
    assert pending_plans.take("u1", "chat-a")[0] is None
    assert pending_plans.take("u1", None)[0] == CALLS


# --- Supabase-backed path ---------------------------------------------------


def _configured(monkeypatch, handler):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-key")
    pending_plans._transport = httpx.MockTransport(handler)


def test_save_posts_the_row_and_reports_the_db_backend(monkeypatch):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["prefer"] = request.headers.get("Prefer", "")
        return httpx.Response(201, json=[])

    _configured(monkeypatch, handler)
    assert pending_plans.save("u1", "chat-a", CALLS) == "db"
    assert "pending_plans" in seen["url"]
    assert seen["body"]["chat_id"] == "chat-a"
    assert seen["body"]["calls"] == CALLS
    # Re-proposing in the same chat must replace, not collide on the primary key.
    assert "merge-duplicates" in seen["prefer"]


def test_take_deletes_and_returns_the_row(monkeypatch):
    future = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=5)).isoformat()

    def handler(request):
        assert request.method == "DELETE", "take must consume the row"
        return httpx.Response(200, json=[{"calls": CALLS, "expires_at": future}])

    _configured(monkeypatch, handler)
    calls, source = pending_plans.take("u1", "chat-a")
    assert calls == CALLS and source == "db"


def test_an_expired_row_is_not_released(monkeypatch):
    past = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=1)).isoformat()
    _configured(monkeypatch, lambda r: httpx.Response(200, json=[{"calls": CALLS, "expires_at": past}]))
    calls, source = pending_plans.take("u1", "chat-a")
    assert calls is None and source == "db-expired"


def test_a_missing_table_falls_back_to_memory_instead_of_failing(monkeypatch):
    """If migration 0007 hasn't been applied, proposing must still work — degraded
    to in-process durability, not broken."""
    _configured(monkeypatch, lambda r: httpx.Response(404, json={"message": "relation does not exist"}))
    assert pending_plans.save("u1", "chat-a", CALLS) == "memory"
    calls, source = pending_plans.take("u1", "chat-a")
    assert calls == CALLS and source == "memory"


def test_a_network_failure_falls_back_to_memory(monkeypatch):
    def boom(request):
        raise httpx.ConnectError("unreachable")

    _configured(monkeypatch, boom)
    assert pending_plans.save("u1", "chat-a", CALLS) == "memory"
    assert pending_plans.take("u1", "chat-a")[0] == CALLS


def test_a_db_miss_reports_a_miss_rather_than_a_stale_memory_hit(monkeypatch):
    """When the DB is authoritative and says there is no plan, the memory copy must
    not resurrect one another process already consumed."""
    pending_plans._mem_save("u1", "chat-a", CALLS)
    _configured(monkeypatch, lambda r: httpx.Response(200, json=[]))
    calls, source = pending_plans.take("u1", "chat-a")
    assert calls is None and source == "db-miss"
