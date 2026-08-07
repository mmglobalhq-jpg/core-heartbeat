-- Minimum Supabase surface needed to run the Core project's migrations against a
-- plain Postgres instance.
--
-- WHY THIS EXISTS
-- The briefing migration is written for Supabase: it references auth.users,
-- auth.uid(), and the anon/authenticated/service_role roles. None of those exist
-- in stock Postgres, so the migration could not be exercised anywhere except
-- production — which is exactly the thing that must not be touched. This shim
-- makes the isolated test database accept it.
--
-- It is a TEST FIXTURE, not a Supabase reimplementation. It reproduces the parts
-- the migration depends on and nothing else. In particular auth.uid() reads the
-- same GUC that PostgREST sets, so RLS policies behave the same way under test as
-- they do in production.

create schema if not exists auth;

create table if not exists auth.users (
  id    uuid primary key default gen_random_uuid(),
  email text unique
);

-- PostgREST sets request.jwt.claim.sub from the verified JWT. Reading the same
-- GUC means the RLS policies under test are the production policies, not
-- test-only lookalikes. `true` on current_setting suppresses the error when the
-- GUC is unset, so an unauthenticated session gets NULL and matches nothing.
create or replace function auth.uid() returns uuid
  language sql stable
as $$
  select nullif(current_setting('request.jwt.claim.sub', true), '')::uuid
$$;

do $$
begin
  if not exists (select 1 from pg_roles where rolname = 'anon') then
    create role anon nologin;
  end if;
  if not exists (select 1 from pg_roles where rolname = 'authenticated') then
    create role authenticated nologin;
  end if;
  -- BYPASSRLS mirrors Supabase: the service role is how the briefing generator
  -- writes rows that no user is permitted to author.
  if not exists (select 1 from pg_roles where rolname = 'service_role') then
    create role service_role nologin bypassrls;
  end if;
end $$;

-- Roles are CLUSTER-wide, not database-wide. A create-if-not-exists is therefore
-- not enough: a service_role left behind by some other database in the same
-- cluster can exist without BYPASSRLS, and every service-role insert then fails
-- an RLS check it was never meant to be subject to. Assert the attribute rather
-- than assuming creation set it.
alter role service_role bypassrls;

grant usage on schema public, auth to anon, authenticated, service_role;
grant select on auth.users to authenticated;
-- Write access is fixture-only: the tests seed users here. In Supabase, auth.users
-- is managed by GoTrue and no application role writes it.
grant select, insert, update, delete on auth.users to service_role;
