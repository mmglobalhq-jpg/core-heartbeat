#!/usr/bin/env bash
# Rebuild the isolated Daily Briefing test database from scratch and verify the
# schema's isolation and canonical-briefing invariants.
#
# ISOLATION
# This uses its OWN Postgres container (briefing-testpg, loopback :55446), not
# the shared orc-testpg. Postgres roles are cluster-wide, so running these
# migrations in a shared cluster would mean altering another project's
# service_role. Nothing here touches production Supabase, and nothing here
# touches another project's test data.
#
# Usage: tests/fixtures/rebuild_briefing_testdb.sh
set -euo pipefail

CONTAINER=briefing-testpg
PORT=55446
DB=briefing_test
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIGRATIONS="$HERE/../../../core-chat/supabase/migrations"

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "==> starting $CONTAINER on 127.0.0.1:$PORT"
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  docker run -d --name "$CONTAINER" \
    -e POSTGRES_HOST_AUTH_METHOD=trust \
    -p "127.0.0.1:$PORT:5432" \
    --label project=daily-briefing --label lifecycle=ephemeral-test \
    postgres:16 >/dev/null
  for _ in $(seq 1 30); do
    docker exec "$CONTAINER" pg_isready -U postgres -q 2>/dev/null && break
    sleep 1
  done
fi

psql_db() { docker exec -i "$CONTAINER" psql -U postgres -d "$DB" "$@"; }

echo "==> recreating $DB"
docker exec "$CONTAINER" psql -U postgres -q -c "drop database if exists $DB" >/dev/null
docker exec "$CONTAINER" psql -U postgres -q -c "create database $DB" >/dev/null

echo "==> applying Supabase shim"
psql_db -v ON_ERROR_STOP=1 -q < "$HERE/supabase_shim.sql"

echo "==> applying briefing migration"
psql_db -v ON_ERROR_STOP=1 -q < "$MIGRATIONS/0008_daily_briefing.sql"

echo "==> verifying isolation and invariants"
out=$(psql_db < "$HERE/briefing_rls_check.sql" 2>&1 | grep -E 'PASS|FAIL|ERROR' | sed -E 's/^NOTICE:  //')
echo "$out" | sed 's/^/    /'

# A check that errored produced no PASS line, which on its own looks like silence
# rather than failure. Count explicitly and require every check to have reported.
expected=11
passed=$(grep -c '^PASS' <<<"$out" || true)
if grep -qE '^(FAIL|ERROR)' <<<"$out" || [ "$passed" -ne "$expected" ]; then
  echo
  echo "!!  $passed/$expected checks passed — schema NOT verified"
  exit 1
fi

echo
echo "    $passed/$expected checks passed"
