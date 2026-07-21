"""Read-only REIT research report tools for the orchestrator.

Answers questions about REIT research reports produced by the ARR research engine.
The reports are read through that engine's **normalized, server-only reader contract**
— versioned Supabase RPC functions — using the dedicated REIT service-role key. This
module never names or queries the engine's issuer-specific tables; the RPCs own all
schema knowledge, completed/current filtering, and namespacing. Reports are GLOBAL
(not per-user), so ``user_id`` is accepted for a uniform dispatch signature but is not
used for scoping.

Strictly read-only: these tools never create, edit, supersede, or trigger a report,
never run the research pipeline, and never mutate a row or Storage object.

Mirrors tools/google_calendar.py: an injectable ``_transport`` test seam, a
name->callable dispatch, and a ``run_reit_tool`` entrypoint that degrades to a
concise ``error: ...`` string (never raises) so a bad tool call never crashes the
graph.

Reader contract (source of truth: arr-research-engine migration 0005):
  * reit_research_list_issuers_v1()                     — issuers w/ >=1 current report
  * reit_research_list_reports_v1(p_issuer_code, p_limit) — completed/current summaries
  * reit_research_get_report_v1(p_report_id)            — one completed/current report

Report ids are namespaced (``arr:<uuid>`` / ``orc:<uuid>``). A bare UUID is accepted by
the detail RPC as a transitional legacy ARR id; it is never interpreted as ORC.
"""
from __future__ import annotations

import os
import re

import httpx

REITS_SUPABASE_URL_ENV = "REITS_SUPABASE_URL"
REITS_SERVICE_ROLE_ENV = "REITS_SUPABASE_SERVICE_ROLE_KEY"
# The report body (Markdown) is returned by the reader RPC, so no Storage bucket is
# required. The variable is read only as an optional future hook; absence is not an error.
REITS_STORAGE_BUCKET_ENV = "REITS_SUPABASE_STORAGE_BUCKET"
REITS_REPORT_MAX_CHARS_ENV = "REITS_REPORT_MAX_CHARS"

REQUEST_TIMEOUT_S = 20.0
DEFAULT_REPORT_MAX_CHARS = 50_000
DEFAULT_LIST_LIMIT = 50
MAX_LIST_LIMIT = 250  # matches the reader contract's server-side clamp (migration 0006)

# Reader-contract RPC names (versioned). The names are constants — never derived from
# user input — so a caller can never select a different function.
_RPC_LIST_ISSUERS = "reit_research_list_issuers_v1"
_RPC_LIST_REPORTS = "reit_research_list_reports_v1"
_RPC_GET_REPORT = "reit_research_get_report_v1"

# Test seam: unit tests set this to an httpx.MockTransport. None -> real network.
_transport: httpx.BaseTransport | None = None


class ReitError(Exception):
    """Config/service error surfaced to the model as ``error: ...``."""


# --- issuer catalog (display names only; the list itself is data-driven) -----

_ISSUER_NAMES = {
    "ARR": "ARMOUR Residential REIT",
    "ORC": "Orchid Island Capital, Inc.",
}

# Aliases the model / user may use for a known issuer -> canonical issuer code. Future
# issuers work without an entry here: a bare symbol that matches _SYMBOL_RE is accepted
# as-is (data-driven) and passed to the RPC, which returns nothing for an unknown code.
_ALIASES = {
    "ARR": "ARR",
    "ARMOUR": "ARR",
    "ARMOUR RESIDENTIAL REIT": "ARR",
    "ARMOUR RESIDENTIAL": "ARR",
    "ORC": "ORC",
    "ORCHID": "ORC",
    "ORCHID ISLAND": "ORC",
    "ORCHID ISLAND CAPITAL": "ORC",
    "ORCHID ISLAND CAPITAL INC": "ORC",
    "ORCHID ISLAND CAPITAL, INC.": "ORC",
}
_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.]{0,9}$")
# Word-boundary alias tokens for the supervisor's forced-KB exemption. Kept narrow
# (issuer names, not the generic word "reit") to avoid false positives.
_REFERENCE_RE = re.compile(r"\b(arr|armour|orc|orchid)\b", re.IGNORECASE)

# A report id is a namespaced (arr:/orc:) UUID or — transitionally — a bare ARR UUID.
_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_REPORT_ID_RE = re.compile(rf"^(?:(arr|orc):)?({_UUID_RE.pattern})$", re.IGNORECASE)

_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def looks_like_reit_reference(text: str) -> bool:
    """True if the text clearly references a known REIT issuer (ARR/ARMOUR/ORC/Orchid).

    Used by the supervisor to keep clear REIT-report questions out of the forced
    generic knowledge-base retrieval. Deliberately narrow.
    """
    return bool(_REFERENCE_RE.search(text or ""))


def _issuer_name(code: str) -> str:
    return _ISSUER_NAMES.get(code, code)


def normalize_symbol(raw: str | None) -> str:
    """Normalize a user/model-supplied REIT symbol or alias to a canonical issuer code.

    Raises ReitError on anything that is not a known alias or a safe bare symbol — so a
    model can never smuggle a filter/SQL fragment through as a symbol. The result is only
    ever passed as a *value* argument to the reader RPC, never used to build a name.
    """
    v = (raw or "").strip().upper()
    if not v:
        raise ReitError("a REIT symbol is required (e.g. ARR or ORC)")
    if v in _ALIASES:
        return _ALIASES[v]
    if _SYMBOL_RE.match(v):
        return v
    raise ReitError(f"unrecognized REIT symbol {raw!r}")


def normalize_report_id(raw: str | None) -> str:
    """Validate + normalize a report id to ``arr:<uuid>`` / ``orc:<uuid>`` / bare uuid.

    Rejects anything else with ReitError so no malformed id reaches the RPC.
    """
    v = (raw or "").strip()
    if not v:
        raise ReitError("a report id is required (find it with list_reit_reports)")
    m = _REPORT_ID_RE.match(v)
    if not m:
        raise ReitError(f"unrecognized report id {raw!r}")
    prefix, uuid = m.group(1), m.group(2)
    return f"{prefix.lower()}:{uuid.lower()}" if prefix else uuid.lower()


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


# --- reader-contract access (REIT service role, RPC only) --------------------

def _sb_url() -> str:
    url = os.environ.get(REITS_SUPABASE_URL_ENV)
    if not url:
        raise ReitError(f"{REITS_SUPABASE_URL_ENV} is not set")
    return url.rstrip("/")


def _sb_headers() -> dict[str, str]:
    key = os.environ.get(REITS_SERVICE_ROLE_ENV)
    if not key:
        raise ReitError(f"{REITS_SERVICE_ROLE_ENV} is not set")
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _http() -> httpx.Client:
    return httpx.Client(timeout=REQUEST_TIMEOUT_S, transport=_transport)


def _rpc(fn: str, payload: dict) -> list[dict]:
    """POST to a reader-contract RPC and return the row list (set-returning function)."""
    with _http() as c:
        r = c.post(f"{_sb_url()}/rest/v1/rpc/{fn}", json=payload, headers=_sb_headers())
    r.raise_for_status()
    body = r.json()
    return body if isinstance(body, list) else []


# --- tools ------------------------------------------------------------------

def list_reit_issuers(user_id: str, args: dict) -> str:
    """List covered REITs: name, code, completed-report count, latest date."""
    rows = _rpc(_RPC_LIST_ISSUERS, {})
    if not rows:
        return "No REITs with completed reports are available."
    lines = [f"Covered REITs ({len(rows)}):"]
    for r in sorted(rows, key=lambda x: x.get("issuer_code") or ""):
        code = r.get("issuer_code") or "?"
        name = r.get("issuer_name") or _issuer_name(code)
        count = r.get("report_count") or 0
        latest = r.get("latest_portfolio_as_of_date")
        latest_s = f", latest {latest}" if latest else ""
        plural = "s" if count != 1 else ""
        lines.append(f"• {name} ({code}) — {count} report{plural}{latest_s}")
    return "\n".join(lines)


def list_reit_reports(user_id: str, args: dict) -> str:
    """List completed reports for a REIT (metadata only, newest first)."""
    symbol = normalize_symbol(args.get("reit_symbol"))
    try:
        limit = int(args.get("limit") or DEFAULT_LIST_LIMIT)
    except (TypeError, ValueError):
        limit = DEFAULT_LIST_LIMIT
    limit = max(1, min(limit, MAX_LIST_LIMIT))

    rows = _rpc(_RPC_LIST_REPORTS, {"p_issuer_code": symbol, "p_limit": limit})
    if not rows:
        return f"No completed reports found for {symbol}."

    name = _issuer_name(symbol)
    items: list[str] = []
    for r in rows:
        title = _title_for(name, r.get("title"), r.get("portfolio_as_of_date"))
        pub = r.get("publication_date")
        pub_s = f", published {pub}" if pub else ""
        items.append(
            f"• [{r.get('report_id')}] {title} — portfolio {r.get('portfolio_as_of_date')}"
            f"{pub_s}, v{r.get('version')}"
        )
    header = f"{name} ({symbol}) — {len(items)} report(s), newest first:"
    return header + "\n" + "\n".join(items)


def _format_detail(row: dict) -> str:
    name = row.get("issuer_name") or _issuer_name(row.get("issuer_code", ""))
    body = row.get("markdown") or ""
    cap = _report_max_chars()
    truncated = False
    if len(body) > cap:
        body = body[:cap]
        truncated = True
    title = _title_for(name, row.get("title"), row.get("portfolio_as_of_date"))
    out = (
        f"Issuer: {name} ({row.get('issuer_code')})\n"
        f"Report ID: {row.get('report_id')}\n"
        f"Title: {title}\n"
        f"Portfolio date: {row.get('portfolio_as_of_date')}\n"
        f"Publication date: {row.get('publication_date') or 'unknown'}\n"
        f"Version: {row.get('version')}\n\n"
        f"Report:\n{body}"
    )
    if truncated:
        out += f"\n\n[... report truncated at {cap} characters; ask for a specific section ...]"
    return out


def get_reit_report(user_id: str, args: dict) -> str:
    """Fetch one completed current report (with full body) by report id."""
    report_id = normalize_report_id(args.get("report_id"))
    rows = _rpc(_RPC_GET_REPORT, {"p_report_id": report_id})
    if not rows:
        return f"No completed report found with id {report_id}."
    return _format_detail(rows[0])


def get_latest_reit_report(user_id: str, args: dict) -> str:
    """Fetch the newest completed current report for a REIT (with full body)."""
    symbol = normalize_symbol(args.get("reit_symbol"))
    summaries = _rpc(_RPC_LIST_REPORTS, {"p_issuer_code": symbol, "p_limit": 1})
    if not summaries:
        return f"No completed reports found for {symbol}."
    report_id = summaries[0].get("report_id")
    rows = _rpc(_RPC_GET_REPORT, {"p_report_id": report_id})
    if not rows:
        return f"No completed reports found for {symbol}."
    return _format_detail(rows[0])


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
