"""Read-only REIT research report tools for the orchestrator.

Answers questions about REIT research reports produced by the ARR research engine.
The reports live in that engine's ``reit_arr_*`` Supabase tables (forced RLS with
browser roles revoked), so this module reads them via PostgREST with the dedicated
REIT service-role key. Reports are GLOBAL (not per-user), so ``user_id`` is accepted
for a uniform dispatch signature but is not used for scoping.

Strictly read-only: these tools never create, edit, supersede, or trigger a report,
never run the research pipeline, and never mutate a row or Storage object.

Mirrors tools/google_calendar.py: an injectable ``_transport`` test seam, a
name->callable dispatch, and a ``run_reit_tool`` entrypoint that degrades to a
concise ``error: ...`` string (never raises) so a bad tool call never crashes the
graph.

Data contract (source of truth: arr-research-engine models):
  * reit_arr_reports          — id (canonical), issuer_code, portfolio_as_of_date,
                                current_version_id, status ('completed' == current).
  * reit_arr_report_versions  — headline (title), version, markdown (body), status,
                                source_document_id. Superseded revisions have
                                status='superseded' and are never current.
  * reit_arr_source_documents — publication_date.
"""
from __future__ import annotations

import os
import re

import httpx

REITS_SUPABASE_URL_ENV = "REITS_SUPABASE_URL"
REITS_SERVICE_ROLE_ENV = "REITS_SUPABASE_SERVICE_ROLE_KEY"
# The report body (Markdown) lives in the database, so no Storage bucket is required.
# The variable is read only as an optional future hook; absence is not an error.
REITS_STORAGE_BUCKET_ENV = "REITS_SUPABASE_STORAGE_BUCKET"
REITS_REPORT_MAX_CHARS_ENV = "REITS_REPORT_MAX_CHARS"

REQUEST_TIMEOUT_S = 20.0
DEFAULT_REPORT_MAX_CHARS = 50_000
DEFAULT_LIST_LIMIT = 50
MAX_LIST_LIMIT = 200
MAX_REPORTS_SCAN = 500

# Test seam: unit tests set this to an httpx.MockTransport. None -> real network.
_transport: httpx.BaseTransport | None = None


class ReitError(Exception):
    """Config/service error surfaced to the model as ``error: ...``."""


# --- issuer catalog (display names only; the list itself is data-driven) -----

_ISSUER_NAMES = {"ARR": "ARMOUR Residential REIT"}

# Aliases the model / user may use for a known issuer -> canonical symbol. Future
# issuers work without an entry here: a bare symbol that matches _SYMBOL_RE is
# accepted as-is (data-driven), so this map only needs friendly-name aliases.
_ALIASES = {
    "ARR": "ARR",
    "ARMOUR": "ARR",
    "ARMOUR RESIDENTIAL REIT": "ARR",
    "ARMOUR RESIDENTIAL": "ARR",
}
_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.]{0,9}$")
# Word-boundary alias tokens for the supervisor's forced-KB exemption. Kept narrow
# (issuer names, not the generic word "reit") to avoid false positives.
_REFERENCE_RE = re.compile(r"\b(arr|armour)\b", re.IGNORECASE)

_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def looks_like_reit_reference(text: str) -> bool:
    """True if the text clearly references a known REIT issuer (ARR/ARMOUR).

    Used by the supervisor to keep clear REIT-report questions out of the forced
    generic knowledge-base retrieval. Deliberately narrow.
    """
    return bool(_REFERENCE_RE.search(text or ""))


def _issuer_name(code: str) -> str:
    return _ISSUER_NAMES.get(code, code)


def normalize_symbol(raw: str | None) -> str:
    """Normalize a user/model-supplied REIT symbol or alias to a canonical symbol.

    Raises ReitError on anything that is not a known alias or a safe bare symbol —
    so a model can never smuggle a filter/SQL fragment through as a symbol.
    """
    v = (raw or "").strip().upper()
    if not v:
        raise ReitError("a REIT symbol is required (e.g. ARR)")
    if v in _ALIASES:
        return _ALIASES[v]
    if _SYMBOL_RE.match(v):
        return v
    raise ReitError(f"unrecognized REIT symbol {raw!r}")


def _fallback_title(name: str, portfolio_date: str | None) -> str:
    """Deterministic title from issuer + reporting period (never a timestamp)."""
    if not portfolio_date:
        return f"{name} — Monthly Report"
    parts = portfolio_date.split("-")
    if len(parts) < 2:
        return f"{name} — Monthly Report"
    year, month = parts[0], parts[1]
    try:
        label = _MONTHS[int(month) - 1]
    except (ValueError, IndexError):
        return f"{name} — Monthly Report"
    return f"{name} — {label} {year} Monthly Report"


def _title_for(name: str, headline: str | None, portfolio_date: str | None) -> str:
    h = (headline or "").strip()
    return h if h else _fallback_title(name, portfolio_date)


def _report_max_chars() -> int:
    try:
        n = int(os.environ.get(REITS_REPORT_MAX_CHARS_ENV, "") or DEFAULT_REPORT_MAX_CHARS)
    except ValueError:
        return DEFAULT_REPORT_MAX_CHARS
    return n if n > 0 else DEFAULT_REPORT_MAX_CHARS


# --- PostgREST access (REIT service role) -----------------------------------

def _sb_url() -> str:
    url = os.environ.get(REITS_SUPABASE_URL_ENV)
    if not url:
        raise ReitError(f"{REITS_SUPABASE_URL_ENV} is not set")
    return url.rstrip("/")


def _sb_headers() -> dict[str, str]:
    key = os.environ.get(REITS_SERVICE_ROLE_ENV)
    if not key:
        raise ReitError(f"{REITS_SERVICE_ROLE_ENV} is not set")
    return {"apikey": key, "Authorization": f"Bearer {key}", "Accept": "application/json"}


def _http() -> httpx.Client:
    return httpx.Client(timeout=REQUEST_TIMEOUT_S, transport=_transport)


def _get(path: str, params: dict[str, str]) -> list[dict]:
    with _http() as c:
        r = c.get(f"{_sb_url()}/rest/v1/{path}", params=params, headers=_sb_headers())
    r.raise_for_status()
    body = r.json()
    return body if isinstance(body, list) else []


def _in_list(ids: list[str]) -> str:
    return "in.(" + ",".join(ids) + ")"


def _resolve_versions(version_ids: list[str]) -> dict[str, dict]:
    """Fetch the completed report versions for the given ids, keyed by id."""
    if not version_ids:
        return {}
    rows = _get(
        "reit_arr_report_versions",
        {
            "select": "id,headline,version,source_document_id,status",
            "id": _in_list(version_ids),
            "status": "eq.completed",
        },
    )
    return {r["id"]: r for r in rows}


def _resolve_pub_dates(source_ids: list[str]) -> dict[str, str | None]:
    if not source_ids:
        return {}
    rows = _get(
        "reit_arr_source_documents",
        {"select": "id,publication_date", "id": _in_list(source_ids)},
    )
    return {r["id"]: r.get("publication_date") for r in rows}


# --- tools ------------------------------------------------------------------

def list_reit_issuers(user_id: str, args: dict) -> str:
    """List covered REITs: symbol, name, completed-report count, latest date."""
    rows = _get(
        "reit_arr_reports",
        {
            "select": "issuer_code,portfolio_as_of_date",
            "status": "eq.completed",
            "limit": str(MAX_REPORTS_SCAN * 4),
        },
    )
    agg: dict[str, dict] = {}
    for r in rows:
        code = r.get("issuer_code")
        if not code:
            continue
        cur = agg.setdefault(code, {"count": 0, "latest": None})
        cur["count"] += 1
        pd = r.get("portfolio_as_of_date")
        if pd and (cur["latest"] is None or pd > cur["latest"]):
            cur["latest"] = pd
    if not agg:
        return "No REITs with completed reports are available."
    lines = [f"Covered REITs ({len(agg)}):"]
    for code in sorted(agg):
        v = agg[code]
        latest = f", latest {v['latest']}" if v["latest"] else ""
        plural = "s" if v["count"] != 1 else ""
        lines.append(f"• {_issuer_name(code)} ({code}) — {v['count']} report{plural}{latest}")
    return "\n".join(lines)


def list_reit_reports(user_id: str, args: dict) -> str:
    """List completed reports for a REIT (metadata only, newest first)."""
    symbol = normalize_symbol(args.get("reit_symbol"))
    try:
        limit = int(args.get("limit") or DEFAULT_LIST_LIMIT)
    except (TypeError, ValueError):
        limit = DEFAULT_LIST_LIMIT
    limit = max(1, min(limit, MAX_LIST_LIMIT))

    reports = _get(
        "reit_arr_reports",
        {
            "select": "id,issuer_code,portfolio_as_of_date,current_version_id",
            "issuer_code": f"eq.{symbol}",
            "status": "eq.completed",
            "order": "portfolio_as_of_date.desc",
            "limit": str(MAX_REPORTS_SCAN),
        },
    )
    reports = [r for r in reports if r.get("current_version_id")]
    if not reports:
        return f"No completed reports found for {symbol}."

    versions = _resolve_versions([r["current_version_id"] for r in reports])
    pubs = _resolve_pub_dates(
        [v["source_document_id"] for v in versions.values() if v.get("source_document_id")]
    )

    name = _issuer_name(symbol)
    items: list[str] = []
    for r in reports:
        v = versions.get(r["current_version_id"])
        if not v:  # current version not completed -> exclude (no superseded/draft rows)
            continue
        title = _title_for(name, v.get("headline"), r.get("portfolio_as_of_date"))
        pub = pubs.get(v.get("source_document_id"))
        pub_s = f", published {pub}" if pub else ""
        items.append(
            f"• [{r['id']}] {title} — portfolio {r.get('portfolio_as_of_date')}"
            f"{pub_s}, v{v.get('version')}"
        )
        if len(items) >= limit:
            break
    if not items:
        return f"No completed reports found for {symbol}."
    header = f"{name} ({symbol}) — {len(items)} report(s), newest first:"
    return header + "\n" + "\n".join(items)


def _format_detail(report: dict, name: str, version: dict, pub: str | None) -> str:
    body = version.get("markdown") or ""
    cap = _report_max_chars()
    truncated = False
    if len(body) > cap:
        body = body[:cap]
        truncated = True
    title = _title_for(name, version.get("headline"), report.get("portfolio_as_of_date"))
    out = (
        f"Issuer: {name} ({report.get('issuer_code')})\n"
        f"Report ID: {report.get('id')}\n"
        f"Title: {title}\n"
        f"Portfolio date: {report.get('portfolio_as_of_date')}\n"
        f"Publication date: {pub or 'unknown'}\n"
        f"Version: {version.get('version')}\n\n"
        f"Report:\n{body}"
    )
    if truncated:
        out += f"\n\n[... report truncated at {cap} characters; ask for a specific section ...]"
    return out


def _load_current_version(report: dict) -> dict | None:
    vid = report.get("current_version_id")
    if not vid:
        return None
    rows = _get(
        "reit_arr_report_versions",
        {
            "select": "id,headline,version,markdown,source_document_id,status",
            "id": f"eq.{vid}",
            "status": "eq.completed",
        },
    )
    return rows[0] if rows else None


def get_reit_report(user_id: str, args: dict) -> str:
    """Fetch one completed current report (with full body) by report id."""
    report_id = (args.get("report_id") or "").strip()
    if not report_id:
        return "error: get_reit_report needs report_id (find it with list_reit_reports)."
    reports = _get(
        "reit_arr_reports",
        {
            "select": "id,issuer_code,portfolio_as_of_date,current_version_id,status",
            "id": f"eq.{report_id}",
            "status": "eq.completed",
        },
    )
    if not reports:
        return f"No completed report found with id {report_id}."
    report = reports[0]
    version = _load_current_version(report)
    if not version:  # current version superseded/not completed -> treat as not found
        return f"No completed report found with id {report_id}."
    pub = _resolve_pub_dates(
        [version["source_document_id"]] if version.get("source_document_id") else []
    ).get(version.get("source_document_id"))
    return _format_detail(report, _issuer_name(report.get("issuer_code", "")), version, pub)


def get_latest_reit_report(user_id: str, args: dict) -> str:
    """Fetch the newest completed current report for a REIT (with full body)."""
    symbol = normalize_symbol(args.get("reit_symbol"))
    reports = _get(
        "reit_arr_reports",
        {
            "select": "id,issuer_code,portfolio_as_of_date,current_version_id,status",
            "issuer_code": f"eq.{symbol}",
            "status": "eq.completed",
            "order": "portfolio_as_of_date.desc",
            "limit": "1",
        },
    )
    if not reports:
        return f"No completed reports found for {symbol}."
    report = reports[0]
    version = _load_current_version(report)
    if not version:
        return f"No completed reports found for {symbol}."
    pub = _resolve_pub_dates(
        [version["source_document_id"]] if version.get("source_document_id") else []
    ).get(version.get("source_document_id"))
    return _format_detail(report, _issuer_name(symbol), version, pub)


# --- dispatch (name -> callable(user_id, args) -> str) ----------------------

_DISPATCH = {
    "list_reit_issuers": list_reit_issuers,
    "list_reit_reports": list_reit_reports,
    "get_reit_report": get_reit_report,
    "get_latest_reit_report": get_latest_reit_report,
}

REIT_TOOL_REGISTRY = frozenset(_DISPATCH)


def run_reit_tool(name: str, user_id: str, args: dict | None = None) -> str:
    """Execute a registered REIT tool; never raises.

    Missing credentials or any other failure degrade to a concise ``error: ...``
    string so the orchestration graph keeps running.
    """
    fn = _DISPATCH.get(name)
    if fn is None:
        return f"error: unknown tool {name!r}"
    try:
        return fn(user_id, args or {})
    except ReitError as exc:
        return f"error: {exc}"
    except httpx.HTTPStatusError as exc:
        return f"error: REIT research service returned {exc.response.status_code}"
    except httpx.HTTPError as exc:
        return f"error: REIT research service unreachable ({type(exc).__name__})"
    except Exception as exc:  # never crash the graph on a bad tool call
        return f"error: {type(exc).__name__}: {exc}"
