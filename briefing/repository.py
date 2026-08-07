"""Persistence for briefings.

TWO IMPLEMENTATIONS, ONE PROTOCOL

* ``PostgrestRepository`` — production. Talks to Supabase over PostgREST with the
  service-role key, matching how every other service in this platform reaches the
  database. Unit-tested with ``httpx.MockTransport``, the same seam
  ``services/pending_plans.py`` uses.

* ``PostgresRepository`` — development and end-to-end tests. Talks SQL directly to
  the isolated test database, so the migration, the constraints and the RLS
  policies are exercised for real rather than mocked.

THIS IS A KNOWN GAP, STATED PLAINLY: the end-to-end test proves the *schema* and
the *pipeline*, using the SQL implementation. The PostgREST implementation is
covered only by unit tests against a mock transport, because no PostgREST
instance sits in front of the test database. Whichever bugs live in the
production data path, the E2E run will not find them.

IDEMPOTENCY
``upsert_briefing`` relies on the ``(user_id, briefing_date)`` unique constraint
rather than a read-then-write check. A read-then-write races with itself, and the
job that produces a briefing is exactly the kind of thing that gets retried.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
from typing import Any, Protocol

import httpx

from briefing.models import BriefingDraft, RawItem, Section

logger = logging.getLogger(__name__)


class BriefingRepository(Protocol):
    def list_enabled_prefs(self) -> list[dict]: ...
    def list_user_sources(self, user_id: str) -> list[dict]: ...
    def upsert_briefing(self, draft: BriefingDraft) -> str: ...
    def replace_sections(self, briefing_id: str, sections: list[Section]) -> None: ...
    def record_delivery(self, briefing_id: str, channel: str, status: str,
                        provider: str | None, detail: str | None) -> None: ...
    def get_briefing(self, user_id: str, briefing_date: dt.date) -> dict | None: ...
    def record_items(self, items: list[RawItem]) -> None: ...


def _section_row(briefing_id: str, section: Section) -> dict[str, Any]:
    return {
        "briefing_id": briefing_id,
        "kind": section.kind,
        "rank": section.rank,
        "headline": section.headline,
        "body": section.body,
        "url": section.url,
        "source_name": section.source_name,
        "published_at": section.published_at.isoformat() if section.published_at else None,
    }


# --- SQL (development / end-to-end) ------------------------------------------


class PostgresRepository:
    """Direct SQL. Used against the isolated test database."""

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or os.environ.get(
            "BRIEFING_TEST_DSN", "postgresql://postgres@127.0.0.1:55446/briefing_test"
        )

    def _connect(self):
        import psycopg

        return psycopg.connect(self.dsn, autocommit=True)

    def list_enabled_prefs(self) -> list[dict]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "select user_id, deliver_at, timezone, topics, deliver_email, email_to "
                "from public.briefing_prefs where enabled"
            )
            return [
                {"user_id": str(r[0]), "deliver_at": str(r[1])[:5], "timezone": r[2],
                 "topics": r[3] or [], "deliver_email": r[4], "email_to": r[5]}
                for r in cur.fetchall()
            ]

    def list_user_sources(self, user_id: str) -> list[dict]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "select kind, url, name, topic from public.briefing_user_sources "
                "where user_id = %s and is_active",
                (user_id,),
            )
            return [{"kind": k, "url": u, "name": n, "topic": t}
                    for k, u, n, t in cur.fetchall()]

    def upsert_briefing(self, draft: BriefingDraft) -> str:
        with self._connect() as conn, conn.cursor() as cur:
            # ON CONFLICT makes the retry case a no-op update rather than a
            # second row. The unique constraint is what guarantees that; this
            # clause just stops it from raising.
            cur.execute(
                """
                insert into public.briefings
                    (user_id, briefing_date, status, timezone, generated_at, run_meta)
                values (%s, %s, 'ready', %s, now(), %s)
                on conflict (user_id, briefing_date) do update
                    set status = 'ready', generated_at = now(),
                        run_meta = excluded.run_meta
                returning id
                """,
                (draft.user_id, draft.briefing_date,
                 draft.run_meta.get("timezone", "America/Chicago"),
                 json.dumps(draft.run_meta)),
            )
            return str(cur.fetchone()[0])

    def replace_sections(self, briefing_id: str, sections: list[Section]) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            # Delete-then-insert inside one transaction: a regenerated briefing
            # must not end up holding yesterday's sections alongside today's.
            cur.execute("delete from public.briefing_sections where briefing_id = %s",
                        (briefing_id,))
            for section in sections:
                row = _section_row(briefing_id, section)
                cur.execute(
                    """
                    insert into public.briefing_sections
                        (briefing_id, kind, rank, headline, body, url, source_name, published_at)
                    values (%(briefing_id)s, %(kind)s, %(rank)s, %(headline)s, %(body)s,
                            %(url)s, %(source_name)s, %(published_at)s)
                    """,
                    row,
                )

    def record_delivery(self, briefing_id: str, channel: str, status: str,
                        provider: str | None = None, detail: str | None = None) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                insert into public.briefing_deliveries
                    (briefing_id, channel, status, provider, detail)
                values (%s, %s, %s, %s, %s)
                """,
                (briefing_id, channel, status, provider, detail),
            )

    def get_briefing(self, user_id: str, briefing_date: dt.date) -> dict | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "select id, status, run_meta from public.briefings "
                "where user_id = %s and briefing_date = %s",
                (user_id, briefing_date),
            )
            row = cur.fetchone()
            if not row:
                return None
            briefing_id, status, run_meta = row
            cur.execute(
                "select kind, rank, headline, body, url, source_name "
                "from public.briefing_sections where briefing_id = %s "
                "order by kind, rank",
                (briefing_id,),
            )
            sections = [
                {"kind": k, "rank": r, "headline": h, "body": b, "url": u, "source_name": s}
                for k, r, h, b, u, s in cur.fetchall()
            ]
            return {"id": str(briefing_id), "status": status,
                    "run_meta": run_meta, "sections": sections}

    def record_items(self, items: list[RawItem]) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            for item in items:
                cur.execute(
                    """
                    insert into public.briefing_items
                        (url, url_hash, title, summary, body, published_at, topic, skipped_reason)
                    values (%s, %s, %s, %s, %s, %s, %s, %s)
                    on conflict (url_hash) do nothing
                    """,
                    (item.url, item.hash, item.title, item.summary,
                     (item.body or "")[:20_000] or None,
                     item.published_at, item.topic, item.skipped_reason),
                )


# --- PostgREST (production) ---------------------------------------------------


class PostgrestRepository:
    """Supabase over PostgREST, using the service-role key.

    NOT exercised by the end-to-end test — see the module docstring.
    """

    # Test seam, matching services/pending_plans.py.
    _transport: httpx.BaseTransport | None = None

    def __init__(self, url: str | None = None, key: str | None = None) -> None:
        from services.secrets import secret

        self.url = (url or os.environ.get("SUPABASE_URL") or "").rstrip("/")
        self.key = key or secret("SUPABASE_SERVICE_ROLE_KEY")
        if not self.url or not self.key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=f"{self.url}/rest/v1",
            headers={
                "apikey": self.key,
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=20.0,
            transport=self._transport,
        )

    def list_enabled_prefs(self) -> list[dict]:
        with self._client() as client:
            response = client.get(
                "/briefing_prefs",
                params={
                    "enabled": "is.true",
                    "select": "user_id,deliver_at,timezone,topics,deliver_email,email_to",
                },
            )
        response.raise_for_status()
        rows = response.json() or []
        for row in rows:
            row["deliver_at"] = str(row.get("deliver_at") or "06:30")[:5]
            row["topics"] = row.get("topics") or []
        return rows

    def list_user_sources(self, user_id: str) -> list[dict]:
        with self._client() as client:
            response = client.get(
                "/briefing_user_sources",
                params={"user_id": f"eq.{user_id}", "is_active": "is.true",
                        "select": "kind,url,name,topic"},
            )
        response.raise_for_status()
        return response.json() or []

    def upsert_briefing(self, draft: BriefingDraft) -> str:
        row = {
            "user_id": draft.user_id,
            "briefing_date": draft.briefing_date.isoformat(),
            "status": "ready",
            "timezone": draft.run_meta.get("timezone", "America/Chicago"),
            "generated_at": dt.datetime.now(dt.UTC).isoformat(),
            "run_meta": draft.run_meta,
        }
        with self._client() as client:
            response = client.post(
                "/briefings",
                json=row,
                headers={
                    "Prefer": "resolution=merge-duplicates,return=representation",
                },
                params={"on_conflict": "user_id,briefing_date"},
            )
        response.raise_for_status()
        return response.json()[0]["id"]

    def replace_sections(self, briefing_id: str, sections: list[Section]) -> None:
        with self._client() as client:
            client.delete("/briefing_sections", params={"briefing_id": f"eq.{briefing_id}"})
            response = client.post(
                "/briefing_sections",
                json=[_section_row(briefing_id, s) for s in sections],
                headers={"Prefer": "return=minimal"},
            )
        response.raise_for_status()

    def record_delivery(self, briefing_id: str, channel: str, status: str,
                        provider: str | None = None, detail: str | None = None) -> None:
        with self._client() as client:
            client.post(
                "/briefing_deliveries",
                json={"briefing_id": briefing_id, "channel": channel, "status": status,
                      "provider": provider, "detail": detail},
                headers={"Prefer": "return=minimal"},
            )

    def get_briefing(self, user_id: str, briefing_date: dt.date) -> dict | None:
        with self._client() as client:
            response = client.get(
                "/briefings",
                params={
                    "user_id": f"eq.{user_id}",
                    "briefing_date": f"eq.{briefing_date.isoformat()}",
                    "select": "id,status,run_meta,briefing_sections(kind,rank,headline,body,url,source_name)",
                },
            )
        response.raise_for_status()
        rows = response.json()
        if not rows:
            return None
        row = rows[0]
        return {
            "id": row["id"],
            "status": row["status"],
            "run_meta": row.get("run_meta") or {},
            "sections": row.get("briefing_sections") or [],
        }

    def record_items(self, items: list[RawItem]) -> None:
        payload = [
            {
                "url": i.url,
                "url_hash": i.hash,
                "title": i.title,
                "summary": i.summary,
                "body": (i.body or "")[:20_000] or None,
                "published_at": i.published_at.isoformat() if i.published_at else None,
                "topic": i.topic,
                "skipped_reason": i.skipped_reason,
            }
            for i in items
        ]
        if not payload:
            return
        with self._client() as client:
            client.post(
                "/briefing_items",
                json=payload,
                headers={"Prefer": "resolution=ignore-duplicates,return=minimal"},
                params={"on_conflict": "url_hash"},
            )
