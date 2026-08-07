# Daily Briefing database fixtures

Schema-level tests for the briefing tables. These run against a **dedicated,
ephemeral Postgres container** and never touch production Supabase.

```
tests/fixtures/rebuild_briefing_testdb.sh
```

Rebuilds the database from `core-chat/supabase/migrations/0008_daily_briefing.sql`
and asserts 11 isolation and invariant checks. Exits non-zero if any check fails
*or* if fewer than 11 checks report — a check that errors produces no `PASS` line,
and silence must not read as success.

| File | Purpose |
|---|---|
| `supabase_shim.sql` | Minimum Supabase surface (`auth.users`, `auth.uid()`, the three roles) so Supabase-targeted migrations can run on stock Postgres |
| `briefing_rls_check.sql` | The 11 checks: cross-user isolation both directions, anonymous read, server-only tables, user-cannot-author, one-briefing-per-day, one-deep-dive, rank constraints |
| `rebuild_briefing_testdb.sh` | Recreate + migrate + verify |

## Why a separate container

Postgres roles are **cluster-wide**. Running these migrations inside the shared
`orc-testpg` would mean altering that project's `service_role` to add `BYPASSRLS`.
So this uses its own container (`briefing-testpg`, loopback `:55446`,
labelled `lifecycle=ephemeral-test`). Disposable — `docker rm -f briefing-testpg`
costs nothing, the rebuild script recreates it.

## These checks are known to fail when the schema is wrong

Verified by deliberately breaking the schema, not by assumption:

- replacing `briefings_select_own` with `using (true)` → checks 1, 2, 4, 5 fail
- dropping `briefing_sections_one_deep_dive` → check 9 fails

If you change a policy or constraint, break it on purpose once and confirm the
suite still notices. A test that cannot go red is decoration.
