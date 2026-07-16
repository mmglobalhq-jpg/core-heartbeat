# Graph-RAG Knowledge Base — Full Build Plan

Status: **planned, not started** (as of 2026-07-10). Spans 3 repos: `graph-rag-service` (fork of
`mmglobalhq-jpg/Graph-RAG-Service-Tool-Public`), `core-heartbeat`, `core-chat`.

## Locked decisions
1. **Scope = per-user private + admin-gated GLOBAL tier.** Scope column `owner_id uuid` NULLABLE,
   `NULL` = global. Retrieval filter `where owner_id = p_owner_id or owner_id is null`. Admin
   "Make available to everyone" toggle on upload. Global-write enforced **server-side in
   core-heartbeat** (admin role from JWT/profile), never trust a client header. v1: no cross-scope
   entity merge (dedup within each scope).
2. **Own Supabase project** for KB graph tables (avoids `documents` name collision with core-chat;
   isolates pgvector load).
3. **Embeddings via Ollama `nomic-embed-text`** (768-dim native, free). Swap
   `lib/ingestion/embedder.ts` (currently OpenAI text-embedding-3-small@768). Depends on the
   `OLLAMA_HOST=0.0.0.0` drop-in.
4. **Tool returns ranked chunks + citations** (`retrieve_only` mode on `/api/query`, skip
   `generate()`); orchestrator composes the final reply.

**Trust boundary:** browser → core-chat (Supabase JWT) → core-heartbeat (*verifies JWT, checks
admin*) → KB service (`API_SECRET_KEY`, trusts `X-User-Id`/scope). KB service is internal-only.

Legend: **[dep: x]** = depends on task x.

---

## Phase 0 — Provisioning (first)
- **0.1** Create the KB Supabase project (separate from app DB). Enable `vector` + `pg_trgm`.
  Record `SUPABASE_URL`/`SUPABASE_SERVICE_KEY`. Create private bucket `ingest-staging`.
- **0.2** Confirm Ollama reachability for embeddings: `ollama pull nomic-embed-text`, reachable
  from a container via the existing `OLLAMA_HOST=0.0.0.0` drop-in.
- **0.3** Mint `API_SECRET_KEY` (service-to-service). Decide internal URL (e.g. `http://graph-rag:3000`).
- **0.4** Fork `Graph-RAG-Service-Tool-Public` → private `mmglobalhq-jpg/graph-rag-service`, clone to
  `~/projects/graph-rag-service`.

## Phase 1 — Fork the KB service (`graph-rag-service`)
- **1.1** Schema: add `owner_id uuid` (nullable, NULL=global) to `documents, chunks, entities, edges,
  communities, jobs` in `lib/supabase/schema.sql`. Add indexes on `chunks(owner_id)`,
  `entities(owner_id)`. Apply to 0.1 project. **[dep: 0.1]**
- **1.2** Scope the 5 SQL fns (`search_chunks_vector`, `search_entities_vector`,
  `find_nearby_entities`, `search_communities_vector`, `expand_entity_graph`): add `p_owner_id uuid`
  param + `and (owner_id = p_owner_id or owner_id is null)`. For `expand_entity_graph`, filter seeds
  AND the edge walk so traversal can't cross into another user's private nodes. **[dep: 1.1]**
- **1.3** Thread `owner_id` through ingest: `app/api/ingest/route.ts` read `X-User-Id` → `processJob`;
  `lib/jobs/processor.ts` stamp `owner_id` on all inserts (~L67-168); `lib/jobs/status.ts createJob`
  stamp job row. **[dep: 1.1]**
- **1.4** Scope dedup + linking per owner: `processor.ts:55` existing-entities query filters by
  owner (user's own + global; no cross-scope merge); `lib/ingestion/linker.ts linkEntities`
  constrains `find_nearby_entities` to same owner. **[dep: 1.2, 1.3]**
- **1.5** Thread user_id + `retrieve_only` in `app/api/query/route.ts`: read `X-User-Id`; accept
  `options.retrieve_only`; pass owner into search/expand/community. When retrieve_only, skip
  `generate()`, return `{chunks:[{id,document_id,content,score}], sources}`. **[dep: 1.2]**
- **1.6** Swap `lib/ingestion/embedder.ts` to POST `${OLLAMA_BASE_URL}/api/embeddings`
  `{model:"nomic-embed-text", prompt}`, keep 768-dim guard, drop OpenAI embed dep. **[dep: 0.2]**
- **1.7** New `GET /api/documents` (Bearer + `X-User-Id`) → owner's docs + global docs
  (`owner_id=user or is null`): `id,title,created_at,owner_id`. Powers mgmt UI + admin delete. **[dep: 1.1]**
- **1.8** Tests: cross-user isolation; global visible to all; retrieve_only returns chunks; Ollama
  embed 768-dim. Write the isolation test alongside 1.4, not after. **[dep: 1.5, 1.6]**
- **1.9** Dockerfiles for Next.js app + Python sidecar (`scripts/run_sidecar.sh`); health via
  `/api/health`. **[dep: 1.5]**

## Phase 2 — core-heartbeat: orchestrator tool + ingest routes
- **2.1** `tools/graphrag.py` httpx client (`_transport` seam +
  retrying `_request`). Base URL `GRAPHRAG_SERVICE_URL`; `Authorization: Bearer <GRAPHRAG_API_KEY>`;
  all calls send `X-User-Id`. **[dep: 0.3]**
- **2.2** Tool `query_knowledge_base(user_id, args)` → POST `/api/query`
  `{query, options:{retrieve_only:true, top_k}}`. Format chunks into compact `[doc:<id>]`-cited
  context. Add `_DISPATCH`, `GRAPHRAG_TOOL_REGISTRY = frozenset(_DISPATCH)`, never-raises
  `run_graphrag_tool(name, user_id, args)`. **[dep: 2.1]**
- **2.3** Register tool in 3 model-facing spots: JSON enum+args `orchestrator.py:95-116`; Pydantic
  `RoutingDecision.tool_name` Literal + `ToolArgs` `models.py:111-156`; supervisor catalog prose
  `orchestrator.py:355-384`. **[dep: 2.2]**
- **2.4** Dispatch branch in `tool_execution` (`orchestrator.py:995-1027`):
  `elif name in GRAPHRAG_TOOL_REGISTRY:` → `run_graphrag_tool(name, state.get("user_id",
  SANDBOX_USER_ID), args)`, wrap as `[tool:...]` msg + `TOOL_CALL_EVENT`. **[dep: 2.2]**
- **2.5** KB routes on gateway (`router.py` + `services/kb.py`), JWT-guarded via
  `Depends(resolve_user_id)`:
  - `POST /kb/ingest {doc_id, scope}` — fetch bytes `services/documents.fetch_original(user_id,
    doc_id)`, base64, forward to KB `/api/ingest` + `X-User-Id`. If `scope=global`, verify caller is
    admin (role from JWT/profile); non-admins forced to own owner_id. Return `job_id`.
  - `GET /kb/jobs/{id}` — proxy KB `/api/jobs/[id]`.
  - `GET /kb/documents` — proxy KB `/api/documents` + `X-User-Id`. **[dep: 2.1]**
- **2.6** Env `GRAPHRAG_SERVICE_URL`, `GRAPHRAG_API_KEY` in `.env.example` (read at call time,
  env-var pattern). Unit-test tool formatter + admin gating. **[dep: 2.5]**

## Phase 3 — core-chat: KB upload UI + management page
- **3.1** `app/api/kb/*` proxies (`ingest`, `jobs/[id]`, `documents`) via `lib/backendProxy.ts`
  (`backendHeaders` forwards Bearer + CF-Access) → core-heartbeat `/kb/*`. **[dep: 2.5]**
- **3.2** Reuse upload plumbing: `createDocument`/`uploadOriginal` (`lib/documents.ts`) land bytes in
  `user-docs` with `chat_id=null`, then call `/api/kb/ingest {doc_id, scope}`; poll `/api/kb/jobs`. **[dep: 3.1]**
- **3.3** Apps sidebar "Knowledge" `<Link href="/kb">` (`components/layout/Sidebar.tsx:177-190`);
  `app/kb/page.tsx` (list from `/api/kb/documents`, dropzone,
  status/delete). **[dep: 3.1]**
- **3.4** Admin-only "Make available to everyone" toggle (reuse profiles/role check from
  `/settings/admin`) → sets `scope:"global"`; enforced server-side in 2.5. **[dep: 3.3]**

## Phase 4 — Deploy & verify
- **4.1** Add `graph-rag` + `graph-rag-sidecar` to root `docker-compose.yml` in `core-deploy`.
  Long-lived containers (ingest is fire-and-forget — never serverless). **[dep: 1.9, 2.6]**
- **4.2** Env wiring: `GRAPHRAG_SERVICE_URL`/`GRAPHRAG_API_KEY` in core-heartbeat; KB service's
  Supabase + Ollama + `API_SECRET_KEY` in its env. **[dep: 4.1]**
- **4.3** E2E smoke: (1) admin global upload visible to 2nd user; (2) user A private doc NOT visible
  to user B (isolation); (3) chat query routes to `query_knowledge_base`, cites KB doc; (4) the
  "chicken recipe → later dinner chat" scenario. **[dep: 4.2, Phase 3]**

## Critical path
```
0.1 ─▶ 1.1 ─▶ 1.2 ─▶ 1.5 ─┐
0.2 ─▶ 1.6 ────────────────┼─▶ 1.8 ─▶ 1.9 ─▶ 4.1 ─▶ 4.2 ─▶ 4.3
       1.3 ─▶ 1.4 ─────────┘
0.3 ─▶ 2.1 ─▶ 2.2 ─▶ 2.3/2.4 ──▶ 2.5 ─▶ 3.1 ─▶ 3.2/3.3 ─▶ 3.4
```
Phase 1 and Phase 2.1-2.4 run in parallel (tool testable against a stub). Phase 3 needs 2.5.
**Riskiest:** 1.2/1.4 (scoped recursive traversal + per-owner dedup — leak hides here) and 4.3
isolation testing. Write isolation test (1.8) alongside 1.4.

## Deferred sub-decisions (don't block P1)
- Delete/update semantics for global docs (admin-only delete via `/kb/documents`?).
- Always-on KB context vs. supervisor-routed tool — this plan does supervisor-routed (tool).

## Service facts (reference)
Endpoints: `POST /api/ingest`(async→job_id), `GET /api/jobs/[id]`, `POST /api/query`, `GET /api/health`.
Auth: single shared `API_SECRET_KEY` (`lib/auth.ts:6`). Pipeline: parse→chunk(parent~800/child~200
tok)→embed(768)→extract entities/rels(OpenRouter qwen3-4b)→fuzzy dedup→graph write→semantic link.
Gen model OpenRouter claude-sonnet-4. Single clean "initial public release" commit.
