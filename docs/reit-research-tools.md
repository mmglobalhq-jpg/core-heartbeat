# REIT research report tools (core-heartbeat)

Four **read-only** tools let the chat orchestrator answer questions using the REIT
research reports produced by the ARR research engine. They read the engine's
`reit_arr_*` Supabase tables via PostgREST with a dedicated service-role key and
never create, edit, supersede, or trigger a report, and never run the research
pipeline or mutate any row/Storage object.

## Tools (`tools/reit_research.py`)

| Tool | Args | Returns |
|---|---|---|
| `list_reit_issuers` | — | Covered REITs: symbol, name, completed-report count, latest date |
| `list_reit_reports` | `reit_symbol`, optional `limit` | Report metadata (id, title, portfolio/publication date, version), newest first — no bodies |
| `get_reit_report` | `report_id` | One completed report with labeled fields + full Markdown body |
| `get_latest_reit_report` | `reit_symbol` | The newest completed report for a REIT (full body) |

`ARR`, `ARMOUR`, and `ARMOUR Residential REIT` all normalize to the canonical symbol
`ARR`. The issuer catalog is data-driven: any `issuer_code` with completed reports is
listable, and a bare future symbol (uppercase alphanumeric) is accepted as-is, so a
new REIT needs no code change. A model-supplied symbol that isn't a known alias or a
safe bare symbol is rejected — no filter/SQL fragment can be smuggled through.

Every tool returns a plain string and **never raises**: missing credentials or any
service failure degrade to a concise `error: ...` so the orchestration graph keeps
running. Report bodies are capped at `REITS_REPORT_MAX_CHARS` (default 50000) and
truncation is always marked explicitly — never silent.

## Orchestrator wiring

- `models.py` — `ToolArgs` gains `reit_symbol` / `report_id` / `limit`;
  `RoutingDecision.tool_name` gains the four names.
- `orchestrator.py` — `ROUTING_JSON_SCHEMA` mirrors them (enum + `tool_args`
  properties) for OpenAI/Anthropic parity; `tool_execution` dispatches
  `REIT_TOOL_REGISTRY`; the Supervisor prompt describes the REIT family; a REIT tool
  result routes on to `local_llm` to compose the answer.
- **Completed / current filtering** matches the engine: a report is served only when
  `reit_arr_reports.status='completed'` and its `current_version_id` resolves to a
  `report_versions` row with `status='completed'`. Superseded / draft / `needs_review`
  reports are never returned. Titles use the stored `headline`, else a deterministic
  fallback from issuer + reporting period.

## Supervisor routing

Clear REIT questions route to the dedicated tools, **not** generic knowledge-base
retrieval: `get_latest_reit_report` for "latest/current/most recent",
`list_reit_reports` for "what reports exist" / an ambiguous period, `get_reit_report`
when an id is known. The deterministic forced-KB "retrieve-first" guard is exempted
for text that clearly references a known issuer (`looks_like_reit_reference`), so a
REIT question is not diverted into `query_knowledge_base` first. The exemption is
narrow (issuer names, not the generic word "REIT"), so forced KB is unchanged for
unrelated substantive questions. A tool result is raw data — the graph routes to
`local_llm` to compose the answer.

## Environment variables (server-side; never expose the service-role key)

```
REITS_SUPABASE_URL=              # ARR research engine's Supabase project URL
REITS_SUPABASE_SERVICE_ROLE_KEY= # service-role key (bypasses the reit_arr_* forced RLS)
REITS_REPORT_MAX_CHARS=50000     # optional; cap on a returned report body (truncation is marked)
# REITS_SUPABASE_STORAGE_BUCKET  # not required — the report body lives in the database
```

The `reit_arr_*` tables have forced RLS with browser roles revoked, so only the
service-role key can read them. This may be the same Supabase project as `SUPABASE_URL`,
but the REIT tools use these dedicated vars and an isolated client. No Storage bucket
is required because the report body (Markdown) is stored in the database. See
`.env.example`. Inject these via the deployment's compose/systemd env (not baked into
the image).

## How a future REIT appears

Add completed reports for a new `issuer_code` in the research database and it becomes
listable automatically. Only a friendly display name (in `_ISSUER_NAMES`) and, if you
want the forced-KB exemption to cover its alias, a token in `_REFERENCE_RE` are code
touch-points; neither is required for the tools to function.

## Tests

`./venv/bin/pytest` (install dev deps: `./venv/bin/pip install -r requirements-dev.txt`).
REIT coverage: `tests/test_reit_research_tool.py` (MockTransport unit tests — listing,
detail, latest, aliases, invalid input, limits, output cap + truncation marker,
superseded/draft exclusion, secret hygiene, never-raises), `tests/test_reit_routing.py`
(routing to REIT tools, forced-KB exemption, REIT-result → local_llm, schema parity),
and `tests/test_reit_tool_loop.py` (end-to-end dispatch + `tool_call` event). No
production Supabase is contacted.

## Troubleshooting

- **`error: REITS_SUPABASE_URL is not set`** — the REIT credentials are missing; set
  `REITS_SUPABASE_URL` / `REITS_SUPABASE_SERVICE_ROLE_KEY`.
- **"No REITs with completed reports are available."** — the research database has no
  `status='completed'` reports (or the credentials point at the wrong project).
- **"No completed report found with id …"** — the id is unknown or its current version
  is superseded/not completed (superseded revisions are never served).
- **A report body ends with a truncation marker** — raise `REITS_REPORT_MAX_CHARS` or
  ask for a specific section.
