# REIT research report tools (core-heartbeat)

Four **read-only** tools let the chat orchestrator answer questions using the REIT
research reports produced by the ARR research engine. They read the reports through the
engine's **normalized, server-only reader contract** — versioned Supabase RPC functions
— using a dedicated service-role key. They never name or query the engine's
issuer-specific tables, never create/edit/supersede/trigger a report, and never run the
research pipeline or mutate any row/Storage object.

## Reader-contract dependency

Depends on the ARR research engine **migration 0005** (`reit_research_*_v1` RPCs). The
tools call only these functions (names are constants, never derived from user input):

| RPC | Purpose |
|---|---|
| `reit_research_list_issuers_v1()` | issuers with ≥1 completed/current report |
| `reit_research_list_reports_v1(p_issuer_code, p_limit)` | completed/current summaries |
| `reit_research_get_report_v1(p_report_id)` | one completed/current report + Markdown |

The RPCs own all schema knowledge, completed/current filtering, ordering, and
namespacing. `tools/reit_research.py` contains **no** `reit_arr_*` / `reit_orc_*` table
name (a unit test asserts their absence).

## Tools (`tools/reit_research.py`)

| Tool | Args | Returns |
|---|---|---|
| `list_reit_issuers` | — | Covered REITs: name, code, completed-report count, latest date |
| `list_reit_reports` | `reit_symbol`, optional `limit` | Report metadata (namespaced id, title, portfolio/publication date, version), newest first — no bodies |
| `get_reit_report` | `report_id` | One completed report with labeled fields + full Markdown body |
| `get_latest_reit_report` | `reit_symbol` | The newest completed report for a REIT (full body) |

### Issuer aliases

Normalized to an issuer **code** only (never used to build a table/RPC/SQL name):

- `ARR`, `ARMOUR`, `ARMOUR Residential REIT` → **ARR**
- `ORC`, `Orchid`, `Orchid Island`, `Orchid Island Capital`, `Orchid Island Capital, Inc.` → **ORC**

The issuer list is data-driven (it comes from `list_issuers`); a bare future symbol
(uppercase alphanumeric) is accepted as-is and passed to the RPC, which returns nothing
for an unknown code. A model-supplied symbol that isn't a known alias or a safe bare
symbol is rejected — no filter/SQL fragment can be smuggled through.

### Report ids: namespaced + legacy

List operations return **namespaced** ids: `arr:<uuid>` / `orc:<uuid>`. `get_reit_report`
accepts a namespaced id or — transitionally — a **bare UUID**, which resolves to a
**legacy ARR** report only (never ORC), so an id colliding across issuers stays
unambiguous. Malformed ids are rejected locally with `error: unrecognized report id …`
before any call.

Every tool returns a plain string and **never raises**: missing credentials or any
service failure degrade to a concise `error: ...` so the orchestration graph keeps
running. Report bodies are capped at `REITS_REPORT_MAX_CHARS` (default 50000) and
truncation is always marked explicitly — never silent.

## Orchestrator wiring

- `models.py` — `ToolArgs` carries `reit_symbol` / `report_id` / `limit`;
  `RoutingDecision.tool_name` includes the four names.
- `orchestrator.py` — `ROUTING_JSON_SCHEMA` mirrors them (enum + `tool_args` properties);
  `tool_execution` dispatches `REIT_TOOL_REGISTRY`; the Supervisor prompt describes the
  REIT family and both issuers' aliases; a REIT tool result routes on to `local_llm`.
- **Completed / current filtering** happens server-side in the RPCs: only reports whose
  logical status is completed and whose `current_version_id` resolves to a completed
  version are returned (ORC additionally requires a persisted snapshot). Draft, failed,
  superseded, `needs_review`, and non-current versions are never served.

## Supervisor routing

Clear REIT questions route to the dedicated tools, **not** generic knowledge-base
retrieval: `get_latest_reit_report` for "latest/current/most recent", `list_reit_reports`
for "what reports exist" / an ambiguous period, `get_reit_report` when an id is known.
The deterministic forced-KB "retrieve-first" guard is exempted for text that clearly
references a known issuer (`looks_like_reit_reference` — now `arr|armour|orc|orchid`), so
a REIT question is not diverted into `query_knowledge_base` first. The exemption is narrow
(issuer names, not the generic word "REIT"), so forced KB is unchanged for unrelated
questions. A tool result is raw data — the graph routes to `local_llm` to compose.

## Environment variables (server-side; never expose the service-role key)

```
REITS_SUPABASE_URL=              # ARR research engine's Supabase project URL
REITS_SUPABASE_SERVICE_ROLE_KEY= # service-role key (the only role granted EXECUTE on the RPCs)
REITS_REPORT_MAX_CHARS=50000     # optional; cap on a returned report body (truncation is marked)
# REITS_SUPABASE_STORAGE_BUCKET  # not required — the report body is returned by the RPC
```

The reader RPCs grant `EXECUTE` only to the service role (PUBLIC/anon/authenticated
revoked), and the underlying tables keep forced RLS, so only the service-role key can
read reports. This may be the same Supabase project as `SUPABASE_URL`, but the REIT tools
use these dedicated vars and an isolated client. Inject these via the deployment's
compose/systemd env (not baked into the image).

### Auth header contract (`apikey` only)

`REITS_SUPABASE_SERVICE_ROLE_KEY` holds **either** a legacy JWT service-role key **or** a
current `sb_secret_*` key. `sb_secret_*` keys are **opaque, not JWTs**. This client sends
the key in **`apikey` only**:

```
apikey: <key>
```

It must **NOT** be duplicated into `Authorization: Bearer <key>` — a raw request that does
so may be rejected as an invalid JWT once the value is an opaque secret key. `apikey`
alone resolves the role for both key generations, so this form works before, during, and
after rotation. `tests/test_reit_research_tool.py` pins it (`apikey` present,
`Authorization` absent, opaque value passed through unparsed).

Note the asymmetry with Core Chat: its `lib/supabaseReits.ts` uses
`createClient(url, key)`, the documented server-side migration pattern, and that SDK sends
**both** `apikey` and `Authorization: Bearer`. That is supported for the SDK path and is
pinned by `core-chat/lib/__tests__/supabaseReits.headers.test.ts`. The two paths therefore
need **separate** live verification — a passing `apikey`-only probe does not prove the SDK
path works.

### Rotating to an `sb_secret_*` key

Legacy and new keys stay active simultaneously, so rotate create → install → verify →
revoke. **No key may appear in a command line, terminal output, log, or shell history** —
read it from a protected temporary file (mode `0600`, deleted afterward) or a secure
interactive prompt.

1. **Create** the new `sb_secret_*` key in the project's API settings. Leave the legacy
   service-role key active.
2. **Pre-rotation live probe — `apikey` only.** Validates the backend's raw-HTTP contract
   before anything is installed:
   ```
   POST <REITS_SUPABASE_URL>/rest/v1/rpc/reit_research_list_issuers_v1
   apikey: <new sb_secret key>
   content-type: application/json
   body: {}
   ```
   **Expect HTTP 200 with a JSON array.** Send the key by reading it from the protected
   file — never inline it as a shell argument.
   **Prohibited:** `Authorization: Bearer <sb_secret key>`. Do not add that header to this
   probe; it is the failure mode this contract exists to avoid.
3. **Install** the new value in the deployment env and recreate the consumers. The key
   lives in `core-heartbeat/.env`, which **both** `backend` and `frontend` load — recreate
   both together. Runtime-only: **no image rebuild required**.
4. **Verify both paths independently:**
   - *backend* — ask the assistant to list covered REITs; the issuer list must render.
   - *Core Chat* — load `/reits`, then open a report (exercises
     `reit_research_get_report_v1` through the SDK, which sends the Bearer header).
   - `docker compose logs --since 5m backend frontend` — grep for `401`, `403`, `PGRST`,
     `JWT`. Never grep for key values.
5. **Revoke** the legacy service-role key **only after both paths pass**, then re-run
   step 4 to prove nothing was still relying on the old key.

**Stop conditions:** the probe returns anything other than 200; any REITS RPC returns
`401`, `403`, or a `PGRST` role error; the issuer list or report fetch returns empty where
it previously returned rows; any key value appears in output or logs. On any of these,
halt and keep the legacy key active — do not revoke.

**Rollback:** the legacy key remains valid until explicitly revoked, so restoring the
previous env value and recreating both containers is a complete rollback. Once revoked it
cannot be restored — never revoke before step 4 passes.

## How a future REIT appears

Once the engine's reader contract lists a new issuer code (i.e. it has completed/current
reports), it appears automatically. Code touch-points are optional: a friendly display
name in `_ISSUER_NAMES` (the RPC already returns one), aliases in `_ALIASES`, and a token
in `_REFERENCE_RE` for the forced-KB exemption.

## Tests

`./venv/bin/pytest` (install dev deps: `./venv/bin/pip install -r requirements-dev.txt`).
REIT coverage: `tests/test_reit_research_tool.py` (MockTransport over the RPC contract —
listing, detail, latest, ARR+ORC aliases, namespaced ids, legacy bare UUID, colliding
UUIDs, malformed ids, limits, output cap + truncation, secret hygiene, never-raises),
`tests/test_reit_routing.py` (routing incl. ORC/Orchid, forced-KB exemption, REIT-result
→ local_llm, schema parity), and `tests/test_reit_tool_loop.py` (end-to-end dispatch +
`tool_call` event). No production Supabase is contacted.

## Troubleshooting

- **`error: REITS_SUPABASE_URL is not set`** — REIT credentials missing; set
  `REITS_SUPABASE_URL` / `REITS_SUPABASE_SERVICE_ROLE_KEY`.
- **`error: REIT research service returned 404`** — the engine's migration 0005 RPCs are
  not present on the target project (apply the reader contract there).
- **`error: unrecognized report id …`** — pass a namespaced id (`arr:<uuid>` /
  `orc:<uuid>`) or a bare ARR UUID.
- **"No REITs with completed reports are available."** — the contract returned no issuers
  (no completed/current reports, or wrong project).
- **A report body ends with a truncation marker** — raise `REITS_REPORT_MAX_CHARS` or ask
  for a specific section.
