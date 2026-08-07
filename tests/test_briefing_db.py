"""Integration tests against a real Postgres, including cross-user isolation.

These run against the isolated container created by
``tests/fixtures/rebuild_briefing_testdb.sh`` and are SKIPPED when it is not
running, so the normal suite stays offline-friendly.

They exist because the unit tests mock the database, and the guarantees that
matter most here — one briefing per user per day, and a user never seeing another
user's briefing — are enforced by the database itself. Mocking them would test
the mock.
"""

from __future__ import annotations

import datetime as dt
import os
import uuid

import pytest

from briefing.models import BriefingDraft, Section
from briefing.repository import PostgresRepository

DSN = os.environ.get("BRIEFING_TEST_DSN", "postgresql://postgres@127.0.0.1:55446/briefing_test")


def _reachable() -> bool:
    try:
        import psycopg

        with psycopg.connect(DSN, connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute("select 1 from public.briefings limit 1")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _reachable(),
    reason="isolated briefing test DB not running "
           "(tests/fixtures/rebuild_briefing_testdb.sh)",
)


@pytest.fixture
def repo():
    return PostgresRepository(DSN)


@pytest.fixture
def user(repo):
    """A fresh user per test, so tests cannot pass by reading each other's rows."""
    import psycopg

    user_id = str(uuid.uuid4())
    with psycopg.connect(DSN, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("insert into auth.users (id, email) values (%s, %s)",
                    (user_id, f"{user_id}@example.test"))
    yield user_id
    with psycopg.connect(DSN, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("delete from auth.users where id = %s", (user_id,))


def sections(n=5):
    out = [Section("top", i, f"Headline {i}", f"Body {i}.", f"https://x.example/{i}",
                   source_name="Outlet") for i in range(1, n + 1)]
    out.append(Section("deep_dive", 1, "Deep", "Deep body.", "https://x.example/deep"))
    return out


def draft(user_id, day=None):
    return BriefingDraft(
        user_id=user_id,
        briefing_date=day or dt.date(2026, 8, 6),
        sections=sections(),
        run_meta={"timezone": "America/Chicago", "sources_ok": 3},
    )


class TestPersistence:
    def test_round_trip(self, repo, user):
        d = draft(user)
        briefing_id = repo.upsert_briefing(d)
        repo.replace_sections(briefing_id, d.sections)

        stored = repo.get_briefing(user, d.briefing_date)
        assert stored["status"] == "ready"
        assert len(stored["sections"]) == 6
        assert sum(1 for s in stored["sections"] if s["kind"] == "deep_dive") == 1

    def test_generating_twice_produces_one_briefing(self, repo, user):
        """The canonical-briefing guarantee, exercised the way it will actually
        be violated: the same job running twice."""
        d = draft(user)
        first = repo.upsert_briefing(d)
        second = repo.upsert_briefing(d)
        assert first == second

        import psycopg

        with psycopg.connect(DSN) as conn, conn.cursor() as cur:
            cur.execute("select count(*) from public.briefings where user_id = %s", (user,))
            assert cur.fetchone()[0] == 1

    def test_regeneration_replaces_sections_rather_than_appending(self, repo, user):
        d = draft(user)
        briefing_id = repo.upsert_briefing(d)
        repo.replace_sections(briefing_id, d.sections)
        repo.replace_sections(briefing_id, d.sections)

        stored = repo.get_briefing(user, d.briefing_date)
        assert len(stored["sections"]) == 6  # not 12

    def test_a_second_deep_dive_is_refused_by_the_database(self, repo, user):
        import psycopg

        d = draft(user)
        briefing_id = repo.upsert_briefing(d)
        repo.replace_sections(briefing_id, d.sections)
        with pytest.raises(psycopg.errors.UniqueViolation):
            with psycopg.connect(DSN, autocommit=True) as conn, conn.cursor() as cur:
                cur.execute(
                    "insert into public.briefing_sections "
                    "(briefing_id, kind, rank, headline, body, url) "
                    "values (%s, 'deep_dive', 2, 'Second', 'body', 'https://x.example/2')",
                    (briefing_id,),
                )

    def test_delivery_is_recorded(self, repo, user):
        d = draft(user)
        briefing_id = repo.upsert_briefing(d)
        repo.record_delivery(briefing_id, "file", "sent", "file", "/tmp/x.html")
        import psycopg

        with psycopg.connect(DSN) as conn, conn.cursor() as cur:
            cur.execute("select channel, status from public.briefing_deliveries "
                        "where briefing_id = %s", (briefing_id,))
            assert cur.fetchone() == ("file", "sent")


class TestCrossUserIsolation:
    """The security property. Two real users, one database, the authenticated role."""

    @pytest.fixture
    def two_users(self, repo):
        import psycopg

        ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        with psycopg.connect(DSN, autocommit=True) as conn, conn.cursor() as cur:
            for uid in ids:
                cur.execute("insert into auth.users (id, email) values (%s, %s)",
                            (uid, f"{uid}@example.test"))
        for uid in ids:
            d = draft(uid)
            bid = repo.upsert_briefing(d)
            repo.replace_sections(bid, d.sections)
        yield ids
        with psycopg.connect(DSN, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute("delete from auth.users where id = any(%s)", (ids,))

    def _as_user(self, uid, sql, params=()):
        import psycopg

        with psycopg.connect(DSN, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute("set role authenticated")
            # The same GUC PostgREST sets from a verified JWT, so this exercises
            # the production policy rather than a test-only lookalike.
            cur.execute("select set_config('request.jwt.claim.sub', %s, false)", (uid,))
            cur.execute(sql, params)
            return cur.fetchall()

    def test_each_user_sees_only_their_own_briefing(self, two_users):
        a, b = two_users
        rows_a = self._as_user(a, "select user_id from public.briefings")
        rows_b = self._as_user(b, "select user_id from public.briefings")
        assert [str(r[0]) for r in rows_a] == [a]
        assert [str(r[0]) for r in rows_b] == [b]

    def test_naming_another_users_id_returns_nothing(self, two_users):
        a, b = two_users
        rows = self._as_user(a, "select id from public.briefings where user_id = %s", (b,))
        assert rows == []

    def test_sections_do_not_leak_across_users(self, two_users):
        a, b = two_users
        rows = self._as_user(
            a,
            "select s.headline from public.briefing_sections s "
            "join public.briefings br on br.id = s.briefing_id where br.user_id = %s",
            (b,),
        )
        assert rows == []

    def test_deliveries_do_not_leak_across_users(self, two_users, repo):
        a, b = two_users
        stored_b = repo.get_briefing(b, dt.date(2026, 8, 6))
        repo.record_delivery(stored_b["id"], "file", "sent", "file", "/tmp/b.html")
        rows = self._as_user(a, "select id from public.briefing_deliveries")
        assert rows == []

    def test_a_user_cannot_write_a_briefing_for_themselves(self, two_users):
        import psycopg

        a, _ = two_users
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            self._as_user(
                a,
                "insert into public.briefings (user_id, briefing_date) values (%s, %s)",
                (a, dt.date(2026, 8, 9)),
            )

    def test_a_user_cannot_read_the_ingestion_working_set(self, two_users):
        import psycopg

        a, _ = two_users
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            self._as_user(a, "select * from public.briefing_items")
