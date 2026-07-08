"""Fixed-income fund-holdings query tools for the LangGraph supervisor.

Read-only tools over the shared Supabase *fund-tracker* tables — the ones the
``fixed-income-tracker`` worker populates nightly (funds, holdings_snapshots,
position_changes, the ``v_*`` views and the ``compare_snapshots`` /
``get_manager_changes`` RPCs). They let the orchestrator answer questions like
"what does JMTG hold now?", "what has JPMorgan been buying?", "which funds own
CUSIP X?" and "what is JPMorgan's MBS exposure?".

Contrast with tools/user_vault.py: those are per-user and enforce strict
filesystem isolation from a state-resolved ``user_id``. Fund data is GLOBAL
market data — there is no per-user boundary — so these tools carry NO identity.
They query PostgREST/RPC directly with the project's service-role key (the fund
tables are RLS-protected; the anon key reads nothing). The service-role key is
server-side only, read from the environment at call time, and never logged.

Shape mirrors user_vault.py so the orchestrator wires them the same way: pure
query helpers (network I/O, unit-tested via an injectable httpx transport), a
``_DISPATCH`` name->callable table, and a :func:`run_fund_tool` entrypoint that
degrades to an ``error: ...`` string rather than raising — a bad tool call never
crashes the graph ("degrade, never crash", per the rest of the codebase).
"""

from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import httpx

SUPABASE_URL_ENV = "SUPABASE_URL"
SERVICE_ROLE_ENV = "SUPABASE_SERVICE_ROLE_KEY"

REQUEST_TIMEOUT_S = 15.0
# Cap rows fed back into the (small) local model's context / the SSE log.
MAX_ROWS = 40
# Some tools hit heavy DISTINCT-ON views (v_latest_holdings) whose first cold hit
# can trip a statement timeout (HTTP 500). Retry transient 5xx / transport errors
# a few times with short linear backoff before giving up.
MAX_ATTEMPTS = 3
RETRY_BACKOFF_S = 0.4
# Standard change windows -> lookback in days (matches the worker's WINDOWS).
WINDOW_DAYS = {"1D": 1, "7D": 7, "30D": 30, "1Y": 365}

# Test seam: unit tests set this to an ``httpx.MockTransport`` so the pure query
# helpers can be exercised without a live Supabase. None -> real network.
_transport: httpx.BaseTransport | None = None


class FundDataError(Exception):
    """Raised when the fund-data backend is unreachable or misconfigured."""


# --- HTTP layer (PostgREST + RPC over the service-role key) ------------------

def _base_url() -> str:
    url = os.environ.get(SUPABASE_URL_ENV)
    if not url:
        raise FundDataError(f"{SUPABASE_URL_ENV} is not set")
    return url.rstrip("/")


def _headers() -> dict[str, str]:
    key = os.environ.get(SERVICE_ROLE_ENV)
    if not key:
        raise FundDataError(f"{SERVICE_ROLE_ENV} is not set")
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
    }


def _client() -> httpx.Client:
    return httpx.Client(
        base_url=_base_url(),
        headers=_headers(),
        timeout=REQUEST_TIMEOUT_S,
        transport=_transport,
    )


def _request(method: str, path: str, *, params=None, json_body=None,
             extra_headers=None) -> httpx.Response:
    """Issue a PostgREST request, retrying transient 5xx / transport failures."""
    last_exc: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            with _client() as c:
                r = c.request(method, f"/rest/v1/{path}", params=params,
                              json=json_body, headers=extra_headers)
        except httpx.TransportError as exc:  # connect/read/network transport errors
            last_exc = exc
            if attempt + 1 < MAX_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_S * (attempt + 1))
                continue
            raise
        if r.status_code >= 500 and attempt + 1 < MAX_ATTEMPTS:
            last_exc = httpx.HTTPStatusError(
                f"server error {r.status_code}", request=r.request, response=r
            )
            time.sleep(RETRY_BACKOFF_S * (attempt + 1))
            continue
        r.raise_for_status()
        return r
    raise last_exc  # pragma: no cover - loop always returns or raises above


def _pg_get(path: str, params: dict[str, str]) -> list[dict]:
    """GET a PostgREST table/view; return the decoded rows."""
    data = _request("GET", path, params=params).json()
    return data if isinstance(data, list) else [data]


def _pg_rpc(fn: str, payload: dict) -> list[dict]:
    """POST to a PostgREST RPC; return the decoded rows."""
    data = _request("POST", f"rpc/{fn}", json_body=payload).json()
    return data if isinstance(data, list) else [data]


def _pg_count(path: str, params: dict[str, str]) -> int:
    """Exact row count via PostgREST's ``Prefer: count=exact`` Content-Range."""
    r = _request("GET", path, params={**params, "select": "cusip", "limit": "1"},
                 extra_headers={"Prefer": "count=exact"})
    cr = r.headers.get("content-range", "")
    tail = cr.split("/")[-1] if "/" in cr else ""
    return int(tail) if tail.isdigit() else len(r.json())


# --- formatting helpers ------------------------------------------------------

def _norm(s: str) -> str:
    """Lowercase alphanumerics only — so 'JPMorgan' == 'J.P. Morgan'."""
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _par(x) -> str:
    """Compact money: $1.23B / $45.6M / $789K / $12."""
    if x is None:
        return "$?"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "$?"
    a, sign = abs(v), "-" if v < 0 else ""
    if a >= 1e9:
        return f"{sign}${a / 1e9:.2f}B"
    if a >= 1e6:
        return f"{sign}${a / 1e6:.1f}M"
    if a >= 1e3:
        return f"{sign}${a / 1e3:.0f}K"
    return f"{sign}${a:.0f}"


def _pct(x) -> str:
    try:
        return f"{float(x):.2f}%"
    except (TypeError, ValueError):
        return "—%"


def _g(row: dict, key: str, default: str = "—") -> str:
    v = row.get(key)
    return default if v is None or v == "" else str(v)


def _desc(row: dict) -> str:
    name = row.get("issuer") or row.get("security_name") or row.get("cusip") or "?"
    return f"{name} ({_g(row, 'security_type')})"


# --- resolvers ---------------------------------------------------------------

def _resolve_fund(ticker: str) -> dict | None:
    rows = _pg_get("funds", {
        "ticker": f"eq.{(ticker or '').strip().upper()}",
        "select": "id,ticker,fund_name",
        "limit": "1",
    })
    return rows[0] if rows else None


def _resolve_manager(name: str) -> dict | None:
    rows = _pg_get("fund_managers", {"select": "id,canonical_name,aliases"})
    n = _norm(name)
    if not n:
        return None
    for m in rows:
        for cand in [m.get("canonical_name", "")] + list(m.get("aliases") or []):
            cn = _norm(cand)
            if cn and (n == cn or n in cn or cn in n):
                return m
    return None


def _in_filter(ids: list[str]) -> str:
    return "in.(" + ",".join(ids) + ")"


def _latest_date_for_funds(fund_ids: list[str]) -> str | None:
    rows = _pg_get("holdings_snapshots", {
        "fund_id": _in_filter(fund_ids), "select": "as_of_date",
        "order": "as_of_date.desc", "limit": "1",
    })
    return rows[0]["as_of_date"] if rows else None


def _fund_dates(fund_id: str) -> list[str]:
    """Distinct snapshot dates for one fund, newest first (get_fund_snapshot_dates)."""
    rows = _pg_rpc("get_fund_snapshot_dates", {"p_fund_id": fund_id})
    return [r["as_of_date"] for r in rows]


# Each get_fund_snapshot_dates call is ~0.6s (a distinct-date scan over the fund's
# history + network RTT); a manager has up to ~19 funds. Fetch them concurrently
# so get_manager_changes stays interactive instead of ~19 * 0.6s serial.
_MAX_PARALLEL = 8


def _fund_dates_map(fund_ids: list[str]) -> dict[str, list[str]]:
    """Concurrently fetch each fund's snapshot dates; a failed lookup -> []."""
    out: dict[str, list[str]] = {}
    if not fund_ids:
        return out
    with ThreadPoolExecutor(max_workers=min(_MAX_PARALLEL, len(fund_ids))) as ex:
        futures = {ex.submit(_fund_dates, fid): fid for fid in fund_ids}
        for fut, fid in futures.items():
            try:
                out[fid] = fut.result()
            except Exception:
                out[fid] = []
    return out


def _window_from_date(dates: list[str], to_date: str, days: int) -> str | None:
    """Pick a fund's OWN snapshot date to compare against for a window.

    ``dates`` is newest-first ISO strings. Returns the most recent snapshot
    strictly before ``to_date`` and on/before (to_date - days); if the window
    predates the fund's history, falls back to its earliest snapshot. None if the
    fund has no earlier date at all.

    Per-fund anchoring (rather than one manager-wide date) is essential:
    compare_snapshots matches as_of_date EXACTLY, so comparing a fund against a
    date it never had would report every position as spuriously ADDED.
    """
    target = (date.fromisoformat(to_date[:10]) - timedelta(days=days)).isoformat()
    earlier = [d for d in dates if d < to_date]        # dates is desc, so this stays desc
    if not earlier:
        return None
    on_or_before = [d for d in earlier if d <= target]
    return on_or_before[0] if on_or_before else earlier[-1]


def _gap_days(from_date: str, to_date: str) -> int:
    return (date.fromisoformat(to_date[:10]) - date.fromisoformat(from_date[:10])).days


def _within_window(from_date: str, to_date: str, days: int) -> bool:
    """Keep a fund only if its comparison span is reasonably close to the window.

    A fund whose nearest prior snapshot is months old should not be reported under
    a '7D'/'30D' window (its churn since then is not this window's activity). Allow
    some slack for weekends/holidays / near-daily gaps."""
    return _gap_days(from_date, to_date) <= days * 2 + 5


# --- the tools (each returns a compact, model-ready string) ------------------

def get_fund_holdings(ticker: str) -> str:
    """Top current holdings of one fund by par (e.g. 'what does JMTG hold now?').

    Queries the fund's latest snapshot date directly (indexed on fund_id,
    as_of_date) rather than the heavy DISTINCT-ON v_latest_holdings view, which
    scans the fund's whole history. The (fund_id, as_of_date, dedup_id) unique
    constraint means a single date's rows are already de-duplicated per security.
    """
    f = _resolve_fund(ticker)
    if not f:
        return f"No tracked fund found for ticker {(ticker or '').strip().upper()!r}."
    latest = _latest_date_for_funds([f["id"]])
    if not latest:
        return f"{f['ticker']} ({f['fund_name']}): no holdings on record."
    at_date = {"fund_id": f"eq.{f['id']}", "as_of_date": f"eq.{latest}"}
    rows = _pg_get("holdings_snapshots", {
        **at_date,
        "select": "cusip,issuer,security_name,security_type,par_value,"
                  "market_value,weight_pct,coupon,maturity_date",
        "order": "par_value.desc.nullslast", "limit": str(MAX_ROWS),
    })
    if not rows:
        return f"{f['ticker']} ({f['fund_name']}): no holdings on record."
    total = _pg_count("holdings_snapshots", at_date)
    for r in rows:  # par is authoritative; market_value is the fallback (CLAUDE.md)
        r["effective_par"] = r["par_value"] if r.get("par_value") is not None else r.get("market_value")
    lines = [
        f"{f['ticker']} — {f['fund_name']} — top {len(rows)} of {total} "
        f"positions by par (as of {latest}):"
    ]
    for r in rows:
        lines.append(
            f"  {_par(r.get('effective_par')):>9}  {_pct(r.get('weight_pct')):>7}  "
            f"{_desc(r)}  cusip={_g(r, 'cusip')}"
        )
    return "\n".join(lines)


def get_type_exposure(ticker: str) -> str:
    """One fund's par broken down by security type (MBS/CMO/UST/...)."""
    f = _resolve_fund(ticker)
    if not f:
        return f"No tracked fund found for ticker {(ticker or '').strip().upper()!r}."
    day = _latest_date_for_funds([f["id"]])
    if not day:
        return f"{f['ticker']} ({f['fund_name']}): no holdings on record."
    rows = _pg_get("v_type_exposure", {
        "fund_id": f"eq.{f['id']}", "as_of_date": f"eq.{day}",
        "select": "security_type,total_par,total_weight_pct,position_count",
    })
    rows.sort(key=lambda r: float(r.get("total_par") or 0), reverse=True)
    lines = [f"{f['ticker']} — {f['fund_name']} — exposure by security type (as of {day}):"]
    for r in rows:
        lines.append(
            f"  {_g(r, 'security_type'):<6} {_par(r.get('total_par')):>9}  "
            f"{_pct(r.get('total_weight_pct')):>7}  ({_g(r, 'position_count')} pos)"
        )
    return "\n".join(lines)


def get_manager_exposure(manager_name: str) -> str:
    """A manager's aggregate par by security type (e.g. 'JPMorgan's MBS exposure')."""
    m = _resolve_manager(manager_name)
    if not m:
        return f"No fund manager matched {manager_name!r} (known: J.P. Morgan, AllSpring, Victory Capital)."
    fund_ids = [r["id"] for r in _pg_get("funds", {"manager_id": f"eq.{m['id']}", "select": "id"})]
    if not fund_ids:
        return f"No funds tracked for {m['canonical_name']}."
    latest = _latest_date_for_funds(fund_ids)
    if not latest:
        return f"No exposure data for {m['canonical_name']}."
    # Filter the exposure view to the latest date only — otherwise it aggregates
    # the manager's ENTIRE history before we'd discard all but the latest date.
    rows = _pg_get("v_manager_type_exposure", {
        "manager_id": f"eq.{m['id']}", "as_of_date": f"eq.{latest}",
        "select": "security_type,total_par,avg_weight_pct,fund_count",
    })
    rows.sort(key=lambda r: float(r.get("total_par") or 0), reverse=True)
    lines = [f"{m['canonical_name']} — aggregate exposure by security type (as of {latest}):"]
    for r in rows:
        lines.append(
            f"  {_g(r, 'security_type'):<6} {_par(r.get('total_par')):>9}  "
            f"avg {_pct(r.get('avg_weight_pct')):>7}  ({_g(r, 'fund_count')} funds)"
        )
    return "\n".join(lines)


def get_manager_changes(manager_name: str, window: str = "30D") -> str:
    """What a manager has been buying/selling over a window (1D/7D/30D/1Y).

    Computed PER FUND — each fund compared against a date on its own snapshot grid
    via compare_snapshots, then aggregated. Funds with <2 snapshots (nothing to
    diff) are skipped, so sparse baseline-only funds don't flood the result with
    spurious ADDs.
    """
    window = (window or "30D").strip().upper()
    if window not in WINDOW_DAYS:
        return f"error: invalid window {window!r}; use one of 1D, 7D, 30D, 1Y."
    m = _resolve_manager(manager_name)
    if not m:
        return f"No fund manager matched {manager_name!r} (known: J.P. Morgan, AllSpring, Victory Capital)."
    funds = _pg_get("funds", {"manager_id": f"eq.{m['id']}", "select": "id,ticker"})
    if not funds:
        return f"No funds tracked for {m['canonical_name']}."
    days = WINDOW_DAYS[window]
    # Anchor each fund to its own grid, then GROUP funds that share the same
    # (from, to) pair so the dense daily ETFs collapse into a single array-RPC
    # instead of one heavy compare_snapshots call each (latency).
    dates_by_fund = _fund_dates_map([fu["id"] for fu in funds])
    groups: dict[tuple[str, str], list[str]] = {}
    covered = 0
    for fu in funds:
        dates = dates_by_fund.get(fu["id"]) or []
        if len(dates) < 2:
            continue
        to_date = dates[0]
        from_date = _window_from_date(dates, to_date, days)
        if not from_date or not _within_window(from_date, to_date, days):
            continue
        covered += 1
        groups.setdefault((from_date, to_date), []).append(fu["id"])
    changes: list[dict] = []
    for (from_date, to_date), ids in groups.items():
        changes.extend(_pg_rpc("compare_snapshots", {
            "p_fund_ids": ids, "p_from_date": from_date, "p_to_date": to_date,
        }))
    if not covered:
        return f"Not enough snapshot history yet to compute {window} changes for {m['canonical_name']}."
    if not changes:
        return f"{m['canonical_name']}: no position changes over ~{window} across {covered} fund(s)."
    buys = sorted((r for r in changes if r.get("change_type") in ("ADDED", "INCREASED")),
                  key=lambda r: float(r.get("par_change") or 0), reverse=True)
    sells = sorted((r for r in changes if r.get("change_type") in ("REMOVED", "DECREASED")),
                   key=lambda r: float(r.get("par_change") or 0))
    cap = max(1, MAX_ROWS // 4)
    lines = [
        f"{m['canonical_name']} — position changes over ~{window} "
        f"(across {covered} fund(s) with history); "
        f"{len(buys)} added/increased, {len(sells)} removed/decreased.",
        "  Top buys (par added):",
    ]
    for r in buys[:cap]:
        lines.append(f"    +{_par(r.get('par_change'))}  {_g(r, 'ticker'):<7} {_desc(r)} [{_g(r, 'change_type')}]")
    lines.append("  Top sells (par reduced):")
    for r in sells[:cap]:
        lines.append(f"    {_par(r.get('par_change'))}  {_g(r, 'ticker'):<7} {_desc(r)} [{_g(r, 'change_type')}]")
    return "\n".join(lines)


def search_holdings_by_cusip(cusip: str) -> str:
    """Which tracked funds currently hold a given CUSIP."""
    cu = (cusip or "").strip().upper()
    if not cu:
        return "error: no cusip provided."
    rows = _pg_get("v_latest_holdings", {
        "cusip": f"eq.{cu}",
        "select": "ticker,issuer,security_name,security_type,effective_par,weight_pct,as_of_date",
        "order": "effective_par.desc", "limit": str(MAX_ROWS),
    })
    if not rows:
        return f"No tracked fund currently holds CUSIP {cu}."
    head = rows[0]
    name = head.get("issuer") or head.get("security_name") or ""
    lines = [f"CUSIP {cu} — {name} ({_g(head, 'security_type')}) — held by {len(rows)} tracked fund(s):"]
    for r in rows:
        lines.append(
            f"  {_g(r, 'ticker'):<7} {_par(r.get('effective_par')):>9}  "
            f"{_pct(r.get('weight_pct')):>7}  (as of {_g(r, 'as_of_date')})"
        )
    return "\n".join(lines)


def list_funds(manager_name: str | None = None) -> str:
    """List the tracked funds, grouped by manager (optionally filtered to one).

    Answers 'what funds are we tracking?' / 'what JP Morgan funds do you cover?'.
    Shows each fund's latest snapshot date so stale/baseline-only funds are visible.
    """
    managers = {m["id"]: m for m in _pg_get("fund_managers", {"select": "id,canonical_name,aliases"})}
    funds = _pg_get("funds", {"select": "id,ticker,fund_name,manager_id,is_hy", "order": "ticker.asc"})
    target = None
    if manager_name:
        target = _resolve_manager(manager_name)
        if not target:
            return f"No fund manager matched {manager_name!r} (known: J.P. Morgan, AllSpring, Victory Capital)."
        funds = [f for f in funds if f.get("manager_id") == target["id"]]
        if not funds:
            return f"No funds tracked for {target['canonical_name']}."
    last = _fund_dates_map([f["id"] for f in funds])
    by_mgr: dict[str, list[dict]] = {}
    for f in funds:
        mgr = (managers.get(f.get("manager_id")) or {}).get("canonical_name", "Unknown")
        by_mgr.setdefault(mgr, []).append(f)
    scope = target["canonical_name"] if target else "all managers"
    lines = [f"Tracked funds ({scope}) — {len(funds)} total:"]
    for mgr in sorted(by_mgr):
        lines.append(f"  {mgr}:")
        for f in by_mgr[mgr]:
            dates = last.get(f["id"]) or []
            asof = dates[0] if dates else "no data"
            hy = " [HY]" if f.get("is_hy") else ""
            lines.append(f"    {_g(f, 'ticker'):<7} {f.get('fund_name') or ''}{hy}  (last: {asof})")
    return "\n".join(lines)


# --- dispatch (name -> callable(args) -> str) --------------------------------

_DISPATCH = {
    "list_funds": lambda a: list_funds(a.get("manager_name")),
    "get_fund_holdings": lambda a: get_fund_holdings(a["ticker"]),
    "get_type_exposure": lambda a: get_type_exposure(a["ticker"]),
    "get_manager_exposure": lambda a: get_manager_exposure(a["manager_name"]),
    "get_manager_changes": lambda a: get_manager_changes(a["manager_name"], a.get("window") or "30D"),
    "search_holdings_by_cusip": lambda a: search_holdings_by_cusip(a["cusip"]),
}

# The set of tool names the tool_execution node recognizes as fund-data tools.
FUND_TOOL_REGISTRY = frozenset(_DISPATCH)


def run_fund_tool(name: str, args: dict | None = None) -> str:
    """Execute a registered fund tool by name; never raises.

    Unlike the vault tools these take no ``user_id`` — fund data is global. An
    unknown tool, a missing required argument, or any backend failure is returned
    as an ``error: ...`` string so the graph keeps running.
    """
    fn = _DISPATCH.get(name)
    if fn is None:
        return f"error: unknown tool {name!r}"
    try:
        return str(fn(args or {}))
    except KeyError as exc:
        return f"error: missing required argument {exc}"
    except Exception as exc:  # never crash the graph on a bad tool call
        return f"error: {type(exc).__name__}: {exc}"
