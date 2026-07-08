"""Tests for the fixed-income fund-holdings tools (tools/fund_holdings.py).

These tools query the shared Supabase fund-tracker tables over PostgREST/RPC with
the service-role key. The HTTP layer is exercised through an injected
``httpx.MockTransport`` (the module's ``_transport`` seam), so the pure query +
formatting + dispatch logic is tested with a small fake dataset and zero network.

Unlike the vault tools there is no per-user isolation to assert — fund data is
global. The emphasis here is: correct formatting, graceful not-found handling,
and that a bad/misconfigured call degrades to an ``error: ...`` string rather
than crashing the graph.
"""

import httpx
import pytest

import orchestrator
import tools.fund_holdings as fh
from tools.fund_holdings import (
    FundDataError,
    get_fund_holdings,
    get_manager_changes,
    get_manager_exposure,
    get_type_exposure,
    list_funds,
    run_fund_tool,
    search_holdings_by_cusip,
)


def _json(body, headers=None):
    return httpx.Response(200, json=body, headers=headers or {})


def _handler(request: httpx.Request) -> httpx.Response:
    """Route requests against a tiny fake fund-tracker dataset."""
    path = request.url.path
    q = dict(request.url.params)

    if path == "/rest/v1/funds":
        if q.get("ticker", "").startswith("eq."):
            tk = q["ticker"][3:]
            if tk == "JMTG":
                return _json([{"id": "fjmtg", "ticker": "JMTG",
                               "fund_name": "JPMorgan Mortgage-Backed Securities ETF"}])
            return _json([])
        if q.get("manager_id", "").startswith("eq."):
            return _json([{"id": "fjmtg", "ticker": "JMTG"},
                          {"id": "fjpie", "ticker": "JPIE"}])
        # unfiltered catalog (list_funds fetches all, then filters client-side)
        return _json([
            {"id": "fjmtg", "ticker": "JMTG", "fund_name": "JPMorgan MBS ETF",
             "manager_id": "mjpm", "is_hy": False},
            {"id": "fjpie", "ticker": "JPIE", "fund_name": "JPMorgan Income ETF",
             "manager_id": "mjpm", "is_hy": False},
            {"id": "fsad", "ticker": "SADAX", "fund_name": "Allspring Ultra Short-Term Income",
             "manager_id": "mall", "is_hy": False},
        ])

    if path == "/rest/v1/fund_managers":
        return _json([{"id": "mjpm", "canonical_name": "J.P. Morgan",
                       "aliases": ["JPMorgan", "JPM"]},
                      {"id": "mall", "canonical_name": "AllSpring",
                       "aliases": ["Allspring"]}])

    if path == "/rest/v1/v_latest_holdings":   # now only search_holdings_by_cusip
        if q.get("cusip", "").startswith("eq."):
            if q["cusip"][3:] == "912828YK0":
                return _json([{"ticker": "JMTG", "issuer": "US TREASURY N/B",
                               "security_name": "UST", "security_type": "UST",
                               "effective_par": 5_000_000, "weight_pct": 3.1,
                               "as_of_date": "2026-07-08"}])
            return _json([])
        return _json([])

    if path == "/rest/v1/v_type_exposure":
        return _json([
            {"security_type": "MBS", "total_par": 13_000_000,
             "total_weight_pct": 80.1, "position_count": 2},
            {"security_type": "UST", "total_par": 2_000_000,
             "total_weight_pct": 10.0, "position_count": 1},
        ])

    if path == "/rest/v1/v_manager_type_exposure":
        return _json([
            {"security_type": "MBS", "total_par": 50_000_000, "avg_weight_pct": 40.0,
             "fund_count": 3, "as_of_date": "2026-07-08"},
            {"security_type": "CORP", "total_par": 20_000_000, "avg_weight_pct": 15.0,
             "fund_count": 2, "as_of_date": "2026-07-08"},
        ])

    if path == "/rest/v1/holdings_snapshots":
        if "count" in request.headers.get("Prefer", ""):
            return _json([{"cusip": "c1"}], headers={"content-range": "0-0/2"})
        if "par_value" in q.get("select", ""):                   # get_fund_holdings rows
            return _json([
                {"cusip": "c1", "issuer": "FNMA", "security_name": "FN pool",
                 "security_type": "MBS", "par_value": 9_000_000, "market_value": None,
                 "weight_pct": 5.2, "coupon": 5.5, "maturity_date": "2055-01-01"},
                {"cusip": "c2", "issuer": "GNMA", "security_name": "GN pool",
                 "security_type": "MBS", "par_value": 4_000_000, "market_value": None,
                 "weight_pct": 2.1, "coupon": 6.0, "maturity_date": "2054-01-01"},
            ])
        if q.get("as_of_date", "").startswith("lte."):
            return _json([{"as_of_date": "2026-06-08"}])          # anchor on/before target
        if q.get("order") == "as_of_date.desc":
            return _json([{"as_of_date": "2026-07-08"}])          # latest
        if q.get("order") == "as_of_date.asc":
            return _json([{"as_of_date": "2025-08-31"}])          # earliest fallback
        return _json([])

    if path == "/rest/v1/rpc/get_fund_snapshot_dates":
        # ≥2 dates so get_manager_changes has something to diff.
        return _json([{"as_of_date": "2026-07-08"}, {"as_of_date": "2026-06-08"},
                      {"as_of_date": "2025-08-31"}])

    if path == "/rest/v1/rpc/compare_snapshots":
        tk = "JMTG" if request.content and b"fjmtg" in request.content else "JPIE"
        return _json([
            {"ticker": tk, "cusip": "c1", "issuer": "FNMA", "security_name": "FN",
             "security_type": "MBS", "change_type": "INCREASED", "from_par": 1_000_000,
             "to_par": 9_000_000, "par_change": 8_000_000, "par_change_pct": 800.0},
            {"ticker": tk, "cusip": "c9", "issuer": "OLDCO", "security_name": "old",
             "security_type": "CORP", "change_type": "REMOVED", "from_par": 3_000_000,
             "to_par": 0, "par_change": -3_000_000, "par_change_pct": -100.0},
        ])

    return httpx.Response(404, json={"error": "unmocked", "path": path})


@pytest.fixture(autouse=True)
def _fake_backend(monkeypatch):
    """Wire env + an httpx.MockTransport so tools run against the fake dataset."""
    monkeypatch.setenv(fh.SUPABASE_URL_ENV, "http://fund-test.local")
    monkeypatch.setenv(fh.SERVICE_ROLE_ENV, "test-service-role-key")
    monkeypatch.setattr(fh, "_transport", httpx.MockTransport(_handler))


# --- per-tool happy paths + not-found ---------------------------------------

def test_list_funds_all_grouped_by_manager():
    out = list_funds()
    assert "3 total" in out
    assert "J.P. Morgan" in out and "AllSpring" in out
    assert "JMTG" in out and "SADAX" in out


def test_list_funds_filtered_to_manager():
    out = list_funds("JPMorgan")
    assert "2 total" in out
    assert "JMTG" in out and "JPIE" in out
    assert "SADAX" not in out  # AllSpring fund excluded


def test_get_fund_holdings_formats():
    out = get_fund_holdings("JMTG")
    assert "JMTG" in out
    assert "of 2 positions" in out       # count came from the Content-Range header
    assert "$9.0M" in out and "MBS" in out
    assert "cusip=c1" in out


def test_get_fund_holdings_is_case_insensitive():
    assert get_fund_holdings("jmtg").startswith("JMTG —")


def test_get_fund_holdings_not_found():
    assert get_fund_holdings("ZZZZ").startswith("No tracked fund found")


def test_get_type_exposure_sorted_by_par():
    out = get_type_exposure("JMTG")
    assert "exposure by security type" in out
    # MBS ($13.0M) must sort above UST ($2M)
    assert out.index("MBS") < out.index("UST")


def test_get_manager_exposure_resolves_alias():
    out = get_manager_exposure("JPMorgan")   # alias of canonical "J.P. Morgan"
    assert "J.P. Morgan" in out
    assert "$50.0M" in out and "MBS" in out


def test_get_manager_exposure_no_match():
    assert get_manager_exposure("Nonexistent Advisors").startswith("No fund manager matched")


def test_get_manager_changes_splits_buys_and_sells():
    out = get_manager_changes("JPMorgan", "30D")
    assert "over ~30D (across 2 fund(s) with history)" in out
    assert "Top buys" in out and "+$8.0M" in out and "INCREASED" in out
    assert "Top sells" in out and "-$3.0M" in out and "REMOVED" in out


def test_get_manager_changes_defaults_window():
    # No window -> defaults to 30D (dispatch supplies it).
    assert "~30D" in run_fund_tool("get_manager_changes", {"manager_name": "JPMorgan"})


def test_get_manager_changes_invalid_window():
    assert get_manager_changes("JPMorgan", "5D").startswith("error: invalid window")


def test_search_by_cusip_found():
    out = search_holdings_by_cusip("912828YK0")
    assert "held by 1 tracked fund" in out
    assert "JMTG" in out and "US TREASURY" in out


def test_search_by_cusip_none():
    assert search_holdings_by_cusip("000000000").startswith("No tracked fund currently holds")


# --- dispatch + degrade-never-crash -----------------------------------------

def test_run_fund_tool_unknown_tool():
    assert run_fund_tool("does_not_exist", {}).startswith("error: unknown tool")


def test_run_fund_tool_missing_required_arg():
    assert run_fund_tool("get_fund_holdings", {}).startswith("error: missing required argument")


def test_run_fund_tool_missing_env_degrades(monkeypatch):
    monkeypatch.delenv(fh.SUPABASE_URL_ENV, raising=False)
    out = run_fund_tool("get_fund_holdings", {"ticker": "JMTG"})
    assert out.startswith("error:") and "SUPABASE_URL" in out


def test_backend_error_raises_in_pure_helper(monkeypatch):
    # A 5xx surfaces as an HTTPStatusError in the pure helper (run_fund_tool wraps it).
    monkeypatch.setattr(fh, "_transport",
                        httpx.MockTransport(lambda r: httpx.Response(500, text="boom")))
    with pytest.raises(httpx.HTTPStatusError):
        get_fund_holdings("JMTG")
    assert run_fund_tool("get_fund_holdings", {"ticker": "JMTG"}).startswith("error:")


# --- orchestrator integration: tool_execution routes fund tools --------------

def _tool_state(request):
    return {
        "user_id": "anyone", "tool_request": request, "messages": [],
        "usage": None, "visited": [], "step": 1, "next": "", "status": "",
    }


def test_node_dispatches_fund_tool():
    update = orchestrator.tool_execution(
        _tool_state({"name": "get_fund_holdings", "args": {"ticker": "JMTG"}})
    )
    content = update["messages"][0].content
    assert content.startswith("[tool:get_fund_holdings] JMTG —")
    assert update["visited"] == ["tool_execution"]


def test_node_fund_tool_error_does_not_crash(monkeypatch):
    monkeypatch.setattr(fh, "_transport",
                        httpx.MockTransport(lambda r: httpx.Response(500, text="boom")))
    update = orchestrator.tool_execution(
        _tool_state({"name": "get_manager_exposure", "args": {"manager_name": "JPMorgan"}})
    )
    assert update["messages"][0].content.startswith("[tool:get_manager_exposure] error:")


def test_fund_tools_registered_in_schema():
    # The routing schema + model enum must advertise every dispatchable fund tool.
    enum = orchestrator.ROUTING_JSON_SCHEMA["properties"]["tool_name"]["enum"]
    assert fh.FUND_TOOL_REGISTRY.issubset(set(enum))
