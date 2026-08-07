-- Does the briefing schema actually isolate users, and are the "canonical
-- briefing" invariants really enforced by the database?
--
-- Every check below is written to FAIL LOUDLY. A check that silently returns no
-- rows would look identical to a pass, which is the failure mode this file
-- exists to avoid: the platform has already shipped a green suite over a broken
-- feature more than once.
--
-- Run against the isolated test database only. See tests/fixtures/README.md.

\set ON_ERROR_STOP on
\set QUIET on
\pset tuples_only on
\pset format unaligned

-- --- fixtures, as the service role ------------------------------------------

set role service_role;

-- Clear fixtures first so this file can be run repeatedly against the same
-- database. Without it a second run dies on a duplicate key while seeding, and
-- reports zero passes — which reads exactly like "every check failed" and sent
-- me chasing a policy bug that did not exist. Cascades to briefings/sections.
delete from auth.users where email in ('a@example.test', 'b@example.test');

insert into auth.users (id, email) values
  ('aaaaaaaa-0000-0000-0000-000000000001', 'a@example.test'),
  ('bbbbbbbb-0000-0000-0000-000000000002', 'b@example.test');

insert into public.briefings (id, user_id, briefing_date, status) values
  ('11111111-0000-0000-0000-000000000001', 'aaaaaaaa-0000-0000-0000-000000000001', date '2026-08-06', 'ready'),
  ('22222222-0000-0000-0000-000000000002', 'bbbbbbbb-0000-0000-0000-000000000002', date '2026-08-06', 'ready');

insert into public.briefing_sections (briefing_id, kind, rank, headline, body, url) values
  ('11111111-0000-0000-0000-000000000001', 'top', 1, 'A-only headline', 'body', 'https://example.test/a'),
  ('22222222-0000-0000-0000-000000000002', 'top', 1, 'B-only headline', 'body', 'https://example.test/b');

reset role;


-- --- 1. cross-user isolation: A sees exactly A's briefing -------------------

set role authenticated;
set request.jwt.claim.sub = 'aaaaaaaa-0000-0000-0000-000000000001';

select case when count(*) = 1 and bool_and(user_id = 'aaaaaaaa-0000-0000-0000-000000000001')
            then 'PASS  1  user A sees only own briefing'
            else 'FAIL  1  user A saw ' || count(*) || ' briefings' end
from public.briefings;

-- The cross-user read is the one that matters. Naming B's id explicitly means a
-- policy that leaks everything cannot pass by accident.
select case when count(*) = 0
            then 'PASS  2  user A cannot read user B briefing by id'
            else 'FAIL  2  user A READ USER B BRIEFING' end
from public.briefings where id = '22222222-0000-0000-0000-000000000002';

select case when count(*) = 0
            then 'PASS  3  user A cannot read user B sections'
            else 'FAIL  3  user A READ USER B SECTIONS' end
from public.briefing_sections where headline = 'B-only headline';

-- --- 2. the same, from B's side (a one-sided test can pass on a broken join) -

set request.jwt.claim.sub = 'bbbbbbbb-0000-0000-0000-000000000002';

select case when count(*) = 1 and bool_and(user_id = 'bbbbbbbb-0000-0000-0000-000000000002')
            then 'PASS  4  user B sees only own briefing'
            else 'FAIL  4  user B saw ' || count(*) || ' briefings' end
from public.briefings;

-- --- 3. an unauthenticated session sees nothing -----------------------------

reset request.jwt.claim.sub;

select case when count(*) = 0
            then 'PASS  5  no jwt claim reads nothing'
            else 'FAIL  5  ANONYMOUS SESSION READ ' || count(*) || ' BRIEFINGS' end
from public.briefings;

-- --- 4. the ingestion working set is server-only ----------------------------

do $$
begin
  perform 1 from public.briefing_items;
  raise exception 'FAIL  6  authenticated could read briefing_items';
exception
  when insufficient_privilege then raise notice 'PASS  6  briefing_items denied to authenticated';
end $$;

-- --- 5. a user cannot author a briefing -------------------------------------

set request.jwt.claim.sub = 'aaaaaaaa-0000-0000-0000-000000000001';

do $$
begin
  insert into public.briefings (user_id, briefing_date)
  values ('aaaaaaaa-0000-0000-0000-000000000001', date '2026-08-07');
  raise exception 'FAIL  7  authenticated INSERTED a briefing';
exception
  when insufficient_privilege then raise notice 'PASS  7  briefing insert denied to authenticated';
end $$;

reset role;


-- --- 6. one canonical briefing per user per day -----------------------------

set role service_role;

do $$
begin
  insert into public.briefings (user_id, briefing_date)
  values ('aaaaaaaa-0000-0000-0000-000000000001', date '2026-08-06');
  raise exception 'FAIL  8  SECOND BRIEFING CREATED FOR THE SAME DAY';
exception
  when unique_violation then raise notice 'PASS  8  duplicate briefing for a day rejected';
end $$;

-- --- 7. at most one deep dive ------------------------------------------------

insert into public.briefing_sections (briefing_id, kind, rank, headline, body, url)
values ('11111111-0000-0000-0000-000000000001', 'deep_dive', 1, 'Deep', 'body', 'https://example.test/d');

do $$
begin
  insert into public.briefing_sections (briefing_id, kind, rank, headline, body, url)
  values ('11111111-0000-0000-0000-000000000001', 'deep_dive', 2, 'Deep 2', 'body', 'https://example.test/d2');
  raise exception 'FAIL  9  SECOND DEEP DIVE ACCEPTED';
exception
  when unique_violation then raise notice 'PASS  9  second deep dive rejected';
end $$;

-- --- 8. no duplicate top ranks, and rank stays in 1..5 ----------------------

do $$
begin
  insert into public.briefing_sections (briefing_id, kind, rank, headline, body, url)
  values ('11111111-0000-0000-0000-000000000001', 'top', 1, 'Dup rank', 'body', 'https://example.test/x');
  raise exception 'FAIL 10  DUPLICATE TOP RANK ACCEPTED';
exception
  when unique_violation then raise notice 'PASS 10  duplicate top rank rejected';
end $$;

do $$
begin
  insert into public.briefing_sections (briefing_id, kind, rank, headline, body, url)
  values ('11111111-0000-0000-0000-000000000001', 'top', 6, 'Rank six', 'body', 'https://example.test/y');
  raise exception 'FAIL 11  RANK 6 ACCEPTED';
exception
  when check_violation then raise notice 'PASS 11  rank outside 1..5 rejected';
end $$;

reset role;


-- --- 9. user-added sources: owned, isolated, user-writable -------------------

set role service_role;
insert into public.briefing_user_sources (user_id, kind, url, name) values
  ('aaaaaaaa-0000-0000-0000-000000000001', 'rss', 'https://a.example/feed', 'A feed'),
  ('bbbbbbbb-0000-0000-0000-000000000002', 'rss', 'https://b.example/feed', 'B feed');
reset role;

set role authenticated;
set request.jwt.claim.sub = 'aaaaaaaa-0000-0000-0000-000000000001';

select case when count(*) = 1 then 'PASS 12  user A sees only own sources'
            else 'FAIL 12  user A saw ' || count(*) || ' sources' end
from public.briefing_user_sources;

select case when count(*) = 0
            then 'PASS 13  user A cannot see user B sources'
            else 'FAIL 13  USER A READ USER B SOURCES' end
from public.briefing_user_sources where url = 'https://b.example/feed';

-- Unlike briefings, a user OWNS their reading list and may write it.
do $$
begin
  insert into public.briefing_user_sources (user_id, kind, url, name)
  values ('aaaaaaaa-0000-0000-0000-000000000001', 'rss', 'https://new.example/f', 'New');
  raise notice 'PASS 14  user can add their own source';
exception when others then
  raise exception 'FAIL 14  user could NOT add own source: %', sqlerrm;
end $$;

-- But not one belonging to somebody else.
do $$
begin
  insert into public.briefing_user_sources (user_id, kind, url, name)
  values ('bbbbbbbb-0000-0000-0000-000000000002', 'rss', 'https://evil.example/f', 'X');
  raise exception 'FAIL 15  USER A INSERTED A SOURCE FOR USER B';
exception
  when insufficient_privilege then raise notice 'PASS 15  cross-user source insert denied';
  when others then
    if sqlerrm like '%row-level security%' then
      raise notice 'PASS 15  cross-user source insert denied by RLS';
    else raise; end if;
end $$;

reset role;
