"""Unit tests for the read-only REIT report tools (tools/reit_research.py).

No live service: a MockTransport emulates the ARR engine's PostgREST endpoints over
an in-memory dataset that honors the status/issuer/id/order/limit filters the tools
send. Covers listing, detail, latest, alias normalization, invalid input, result
limits, output cap + truncation marker, exclusion of superseded/draft reports, and
that the service-role key never leaks into output or errors.
"""
import httpx
import pytest

import tools.reit_research as reit

# --- in-memory dataset ------------------------------------------------------
# rep-c's current version is superseded (excluded); rep-gen is still generating.
REPORTS = [
    {"id": "rep-a", "issuer_code": "ARR", "portfolio_as_of_date": "2026-05-31", "current_version_id": "v-a", "status": "completed"},
    {"id": "rep-b", "issuer_code": "ARR", "portfolio_as_of_date": "2026-04-30", "current_version_id": "v-b", "status": "completed"},
    {"id": "rep-c", "issuer_code": "ARR", "portfolio_as_of_date": "2026-03-31", "current_version_id": "v-c", "status": "completed"},
    {"id": "rep-gen", "issuer_code": "ARR", "portfolio_as_of_date": "2026-02-28", "current_version_id": "v-gen", "status": "generating"},
    {"id": "rep-x", "issuer_code": "XYZ", "portfolio_as_of_date": "2026-05-31", "current_version_id": "v-x", "status": "completed"},
]
VERSIONS = [
    {"id": "v-a", "headline": "ARR adds $466mm to portfolio in May", "version": 1, "source_document_id": "d-a", "status": "completed", "markdown": "# Exec summary\n\nThe portfolio grew by $466mm in May."},
    {"id": "v-b", "headline": None, "version": 1, "source_document_id": "d-b", "status": "completed", "markdown": "# April body"},
    {"id": "v-c", "headline": "OLD SUPERSEDED", "version": 2, "source_document_id": "d-c", "status": "superseded", "markdown": "old body"},
    {"id": "v-gen", "headline": "GENERATING", "version": 1, "source_document_id": "d-gen", "status": "generating", "markdown": "draft"},
    {"id": "v-x", "headline": "XYZ Q1 report", "version": 1, "source_document_id": "d-x", "status": "completed", "markdown": "# XYZ body"},
]
DOCS = [
    {"id": "d-a", "publication_date": "2026-06-12"},
    {"id": "d-b", "publication_date": "2026-05-14"},
    {"id": "d-c", "publication_date": "2026-04-15"},
    {"id": "d-x", "publication_date": "2026-06-10"},
]


def _apply(rows, params):
    out = list(rows)
    status = params.get("status")
    if status and status.startswith("eq."):
        out = [r for r in out if r.get("status") == status[3:]]
    for col in ("issuer_code", "id"):
        val = params.get(col)
        if not val:
            continue
        if val.startswith("eq."):
            out = [r for r in out if r.get(col) == val[3:]]
        elif val.startswith("in.("):
            ids = val[len("in.("):-1].split(",")
            out = [r for r in out if r.get(col) in ids]
    order = params.get("order", "")
    if order.startswith("portfolio_as_of_date.desc"):
        out = sorted(out, key=lambda r: r.get("portfolio_as_of_date") or "", reverse=True)
    limit = params.get("limit")
    if limit:
        out = out[: int(limit)]
    return out


def _postgrest(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    params = request.url.params
    if path.endswith("/reit_arr_reports"):
        return httpx.Response(200, json=_apply(REPORTS, params))
    if path.endswith("/reit_arr_report_versions"):
        return httpx.Response(200, json=_apply(VERSIONS, params))
    if path.endswith("/reit_arr_source_documents"):
        return httpx.Response(200, json=_apply(DOCS, params))
    return httpx.Response(404, json=[])


def _install(monkeypatch, handler=_postgrest, *, secret="svc-secret"):
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


# --- list_reit_issuers ------------------------------------------------------

def test_list_issuers_is_data_driven(monkeypatch):
    _install(monkeypatch)
    out = reit.run_reit_tool("list_reit_issuers", "u1", {})
    assert "ARMOUR Residential REIT (ARR)" in out
    assert "XYZ (XYZ)" in out  # unknown code -> name falls back to the code
    assert "3 reports" in out  # rep-a/b/c completed; rep-gen (generating) excluded
    assert "latest 2026-05-31" in out


# --- list_reit_reports ------------------------------------------------------

def test_list_reports_newest_first_metadata_only(monkeypatch):
    _install(monkeypatch)
    out = reit.run_reit_tool("list_reit_reports", "u1", {"reit_symbol": "ARR"})
    assert out.index("[rep-a]") < out.index("[rep-b]")           # newest first
    assert "[rep-c]" not in out and "SUPERSEDED" not in out       # superseded excluded
    assert "[rep-gen]" not in out                                 # generating excluded
    assert "# Exec summary" not in out                            # no body in a list
    assert "ARMOUR Residential REIT — April 2026 Monthly Report" in out  # fallback title
    assert "published 2026-06-12" in out


def test_list_reports_alias_normalization(monkeypatch):
    _install(monkeypatch)
    for alias in ("ARMOUR", "armour residential reit", "arr"):
        out = reit.run_reit_tool("list_reit_reports", "u1", {"reit_symbol": alias})
        assert "[rep-a]" in out


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

def test_get_report_labeled_sections_and_body(monkeypatch):
    _install(monkeypatch)
    out = reit.run_reit_tool("get_reit_report", "u1", {"report_id": "rep-a"})
    assert "Issuer: ARMOUR Residential REIT (ARR)" in out
    assert "Report ID: rep-a" in out
    assert "Version: 1" in out
    assert "Publication date: 2026-06-12" in out
    assert "Report:\n# Exec summary" in out


def test_get_report_superseded_current_is_not_found(monkeypatch):
    _install(monkeypatch)
    out = reit.run_reit_tool("get_reit_report", "u1", {"report_id": "rep-c"})
    assert "No completed report found with id rep-c" in out


def test_get_report_unknown_and_missing_id(monkeypatch):
    _install(monkeypatch)
    assert "No completed report found" in reit.run_reit_tool("get_reit_report", "u1", {"report_id": "nope"})
    assert reit.run_reit_tool("get_reit_report", "u1", {}).startswith("error:")


# --- get_latest_reit_report -------------------------------------------------

def test_get_latest_returns_newest(monkeypatch):
    _install(monkeypatch)
    out = reit.run_reit_tool("get_latest_reit_report", "u1", {"reit_symbol": "ARR"})
    assert "Report ID: rep-a" in out and "# Exec summary" in out


def test_get_latest_unknown_issuer(monkeypatch):
    _install(monkeypatch)
    assert "No completed reports found for ZZ" in reit.run_reit_tool(
        "get_latest_reit_report", "u1", {"reit_symbol": "ZZ"}
    )


# --- output cap + truncation ------------------------------------------------

def test_report_body_is_capped_with_explicit_marker(monkeypatch):
    _install(monkeypatch)
    monkeypatch.setenv("REITS_REPORT_MAX_CHARS", "20")
    out = reit.run_reit_tool("get_reit_report", "u1", {"report_id": "rep-a"})
    assert "report truncated at 20 characters" in out
    # Body is cut to the cap (the full sentence must not survive).
    assert "The portfolio grew by $466mm" not in out


# --- secret hygiene ---------------------------------------------------------

def test_service_role_key_never_appears_in_errors(monkeypatch):
    def _boom(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"message": "internal"})

    _install(monkeypatch, _boom, secret="TOP-SECRET-KEY")
    out = reit.run_reit_tool("list_reit_issuers", "u1", {})
    assert out.startswith("error:")
    assert "TOP-SECRET-KEY" not in out
