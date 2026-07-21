"""Unit tests for the read-only REIT report tools (tools/reit_research.py).

No live service: a MockTransport emulates the ARR engine's **reader-contract RPCs**
(reit_research_list_issuers_v1 / _list_reports_v1 / _get_report_v1) over an in-memory
dataset of already-publishable rows (the contract does completed/current filtering
server-side). Covers listing, detail, latest, alias normalization (ARR + ORC),
namespaced ids, legacy bare ARR UUIDs, colliding UUIDs across issuers, malformed ids,
result limits, output cap + truncation marker, and that the service-role key never
leaks into output or errors. The module names no issuer-specific table.
"""
import httpx
import pytest

import tools.reit_research as reit

# A UUID deliberately shared by an ARR report and an ORC report (namespacing must
# disambiguate). Distinct UUIDs for the other rows.
UUID_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
UUID_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
UUID_EXCLUDED = "cccccccc-cccc-cccc-cccc-cccccccccccc"  # superseded/non-current upstream

ARR_A = f"arr:{UUID_A}"
ARR_B = f"arr:{UUID_B}"
ORC_A = f"orc:{UUID_A}"

# Rows exactly as the reader RPCs return them (already completed/current + namespaced).
_PUBLISHABLE = [
    {"report_id": ARR_A, "issuer_code": "ARR", "issuer_name": "ARMOUR Residential REIT",
     "portfolio_as_of_date": "2026-05-31", "publication_date": "2026-06-12",
     "title": "ARR adds $466mm to portfolio in May", "version": 1, "status": "completed",
     "markdown": "# Exec summary\n\nThe portfolio grew by $466mm in May."},
    {"report_id": ARR_B, "issuer_code": "ARR", "issuer_name": "ARMOUR Residential REIT",
     "portfolio_as_of_date": "2026-04-30", "publication_date": "2026-05-14",
     "title": None, "version": 1, "status": "completed", "markdown": "# April body"},
    {"report_id": ORC_A, "issuer_code": "ORC", "issuer_name": "Orchid Island Capital, Inc.",
     "portfolio_as_of_date": "2026-04-30", "publication_date": "2026-05-03",
     "title": "Orchid Island Capital, Inc. (ORC) — RMBS as of April 30, 2026",
     "version": 1, "status": "completed", "markdown": "# ORC body"},
]


def _summary(row):
    return {k: v for k, v in row.items() if k != "markdown"}


def _list_issuers():
    agg = {}
    for r in _PUBLISHABLE:
        a = agg.setdefault(r["issuer_code"], {
            "issuer_code": r["issuer_code"], "issuer_name": r["issuer_name"],
            "report_count": 0, "latest_portfolio_as_of_date": None,
            "latest_publication_date": None,
        })
        a["report_count"] += 1
        if (a["latest_portfolio_as_of_date"] or "") < r["portfolio_as_of_date"]:
            a["latest_portfolio_as_of_date"] = r["portfolio_as_of_date"]
        if (a["latest_publication_date"] or "") < (r["publication_date"] or ""):
            a["latest_publication_date"] = r["publication_date"]
    return list(agg.values())


def _list_reports(code, limit):
    lim = max(1, min(int(limit or 20), 250))  # reader contract clamp (migration 0006)
    rows = [_summary(r) for r in _PUBLISHABLE if r["issuer_code"] == (code or "").upper()]
    rows.sort(key=lambda r: (r["portfolio_as_of_date"], r["publication_date"] or "",
                             r["report_id"]), reverse=True)
    return rows[:lim]


def _get_report(rid):
    low = (rid or "").lower()
    if low.startswith("arr:"):
        issuer, uuid = "ARR", low[4:]
    elif low.startswith("orc:"):
        issuer, uuid = "ORC", low[4:]
    elif ":" not in low:
        issuer, uuid = "ARR", low  # bare UUID -> legacy ARR only
    else:
        return []
    for r in _PUBLISHABLE:
        if r["issuer_code"] == issuer and r["report_id"].split(":", 1)[1] == uuid:
            return [r]
    return []


def _rpc_handler(request: httpx.Request) -> httpx.Response:
    import json

    path = request.url.path
    body = json.loads(request.content or b"{}")
    if path.endswith("/rpc/reit_research_list_issuers_v1"):
        return httpx.Response(200, json=_list_issuers())
    if path.endswith("/rpc/reit_research_list_reports_v1"):
        return httpx.Response(200, json=_list_reports(body.get("p_issuer_code"),
                                                       body.get("p_limit")))
    if path.endswith("/rpc/reit_research_get_report_v1"):
        return httpx.Response(200, json=_get_report(body.get("p_report_id")))
    return httpx.Response(404, json=[])


def _install(monkeypatch, handler=_rpc_handler, *, secret="svc-secret"):
    monkeypatch.setenv("REITS_SUPABASE_URL", "http://reits.local")
    monkeypatch.setenv("REITS_SUPABASE_SERVICE_ROLE_KEY", secret)
    monkeypatch.setattr(reit, "_transport", httpx.MockTransport(handler))


# --- registry + missing env -------------------------------------------------

def test_registry_has_all_four_tools():
    assert set(reit.REIT_TOOL_REGISTRY) == {
        "list_reit_issuers", "list_reit_reports", "get_reit_report", "get_latest_reit_report",
    }


def test_missing_credentials_degrade_safely(monkeypatch):
    monkeypatch.delenv("REITS_SUPABASE_URL", raising=False)
    monkeypatch.delenv("REITS_SUPABASE_SERVICE_ROLE_KEY", raising=False)
    out = reit.run_reit_tool("list_reit_issuers", "u1", {})
    assert out.startswith("error:") and "REITS_SUPABASE_URL" in out


def test_unknown_tool_name():
    assert "unknown tool" in reit.run_reit_tool("nope", "u1", {})


# --- module hides issuer-specific schema ------------------------------------

def test_module_names_no_issuer_specific_tables():
    import inspect

    src = inspect.getsource(reit)
    assert "reit_arr_" not in src
    assert "reit_orc_" not in src


# --- list_reit_issuers ------------------------------------------------------

def test_list_issuers_shows_both_issuers(monkeypatch):
    _install(monkeypatch)
    out = reit.run_reit_tool("list_reit_issuers", "u1", {})
    assert "ARMOUR Residential REIT (ARR)" in out
    assert "Orchid Island Capital, Inc. (ORC)" in out
    assert "2 reports" in out  # ARR: A + B
    assert "1 report" in out and "1 reports" not in out  # ORC: exactly one
    assert "latest 2026-05-31" in out


# --- list_reit_reports ------------------------------------------------------

def test_list_reports_newest_first_metadata_only(monkeypatch):
    _install(monkeypatch)
    out = reit.run_reit_tool("list_reit_reports", "u1", {"reit_symbol": "ARR"})
    assert out.index(f"[{ARR_A}]") < out.index(f"[{ARR_B}]")   # newest first
    assert "# Exec summary" not in out                         # no body in a list
    assert "ARMOUR Residential REIT — April 2026 Monthly Report" in out  # fallback title
    assert "published 2026-06-12" in out


def test_list_reports_orc_newest_first(monkeypatch):
    _install(monkeypatch)
    out = reit.run_reit_tool("list_reit_reports", "u1", {"reit_symbol": "Orchid"})
    assert f"[{ORC_A}]" in out
    assert "Orchid Island Capital, Inc. (ORC)" in out


def test_list_reports_alias_normalization(monkeypatch):
    _install(monkeypatch)
    for alias in ("ARMOUR", "armour residential reit", "arr"):
        assert f"[{ARR_A}]" in reit.run_reit_tool("list_reit_reports", "u1",
                                                  {"reit_symbol": alias})
    for alias in ("ORC", "Orchid", "orchid island", "Orchid Island Capital, Inc."):
        assert f"[{ORC_A}]" in reit.run_reit_tool("list_reit_reports", "u1",
                                                  {"reit_symbol": alias})


def test_list_reports_invalid_symbol(monkeypatch):
    _install(monkeypatch)
    out = reit.run_reit_tool("list_reit_reports", "u1", {"reit_symbol": "'; DROP TABLE"})
    assert out.startswith("error:") and "unrecognized REIT symbol" in out


def test_list_reports_respects_limit(monkeypatch):
    _install(monkeypatch)
    out = reit.run_reit_tool("list_reit_reports", "u1", {"reit_symbol": "ARR", "limit": 1})
    assert out.count("• [") == 1


def test_list_reports_empty_for_unknown_issuer(monkeypatch):
    _install(monkeypatch)
    out = reit.run_reit_tool("list_reit_reports", "u1", {"reit_symbol": "NOPE"})
    assert "No completed reports found for NOPE" in out


# --- get_reit_report --------------------------------------------------------

def test_get_report_namespaced_arr(monkeypatch):
    _install(monkeypatch)
    out = reit.run_reit_tool("get_reit_report", "u1", {"report_id": ARR_A})
    assert "Issuer: ARMOUR Residential REIT (ARR)" in out
    assert f"Report ID: {ARR_A}" in out
    assert "Report:\n# Exec summary" in out


def test_get_report_namespaced_orc(monkeypatch):
    _install(monkeypatch)
    out = reit.run_reit_tool("get_reit_report", "u1", {"report_id": ORC_A})
    assert "Issuer: Orchid Island Capital, Inc. (ORC)" in out
    assert "Report:\n# ORC body" in out


def test_get_report_colliding_uuid_disambiguated_by_namespace(monkeypatch):
    _install(monkeypatch)
    arr = reit.run_reit_tool("get_reit_report", "u1", {"report_id": ARR_A})
    orc = reit.run_reit_tool("get_reit_report", "u1", {"report_id": ORC_A})
    assert "# Exec summary" in arr and "# ORC body" not in arr
    assert "# ORC body" in orc and "# Exec summary" not in orc


def test_get_report_bare_uuid_is_legacy_arr(monkeypatch):
    _install(monkeypatch)
    out = reit.run_reit_tool("get_reit_report", "u1", {"report_id": UUID_A})
    assert "Issuer: ARMOUR Residential REIT (ARR)" in out  # never ORC
    assert "# Exec summary" in out


def test_get_report_malformed_and_missing_id(monkeypatch):
    _install(monkeypatch)
    for bad in ("nope", "xyz:" + UUID_A, "arr:not-a-uuid", "'; DROP"):
        out = reit.run_reit_tool("get_reit_report", "u1", {"report_id": bad})
        assert out.startswith("error:") and "unrecognized report id" in out
    assert reit.run_reit_tool("get_reit_report", "u1", {}).startswith("error:")


def test_get_report_excluded_id_not_found(monkeypatch):
    _install(monkeypatch)
    out = reit.run_reit_tool("get_reit_report", "u1", {"report_id": f"arr:{UUID_EXCLUDED}"})
    assert "No completed report found" in out


# --- get_latest_reit_report -------------------------------------------------

def test_get_latest_returns_newest(monkeypatch):
    _install(monkeypatch)
    out = reit.run_reit_tool("get_latest_reit_report", "u1", {"reit_symbol": "ARR"})
    assert f"Report ID: {ARR_A}" in out and "# Exec summary" in out


def test_get_latest_orc(monkeypatch):
    _install(monkeypatch)
    out = reit.run_reit_tool("get_latest_reit_report", "u1", {"reit_symbol": "Orchid Island"})
    assert f"Report ID: {ORC_A}" in out and "# ORC body" in out


def test_get_latest_unknown_issuer(monkeypatch):
    _install(monkeypatch)
    assert "No completed reports found for ZZ" in reit.run_reit_tool(
        "get_latest_reit_report", "u1", {"reit_symbol": "ZZ"}
    )


# --- output cap + truncation ------------------------------------------------

def test_report_body_is_capped_with_explicit_marker(monkeypatch):
    _install(monkeypatch)
    monkeypatch.setenv("REITS_REPORT_MAX_CHARS", "20")
    out = reit.run_reit_tool("get_reit_report", "u1", {"report_id": ARR_A})
    assert "report truncated at 20 characters" in out
    assert "The portfolio grew by $466mm" not in out


# --- secret hygiene ---------------------------------------------------------

def test_service_role_key_never_appears_in_errors(monkeypatch):
    def _boom(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"message": "internal"})

    _install(monkeypatch, _boom, secret="TOP-SECRET-KEY")
    out = reit.run_reit_tool("list_reit_issuers", "u1", {})
    assert out.startswith("error:")
    assert "TOP-SECRET-KEY" not in out
