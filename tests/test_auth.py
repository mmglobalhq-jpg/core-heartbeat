"""Tests for the Bearer-token → user_id resolution (Phase 1 multi-user bridge).

Covers auth.extract_bearer_token / resolve_user_id parsing and sandbox fallback,
documents the currently-disabled verify_supabase_jwt hook, proves the resolve
seam returns a verified subject once verification is wired, and checks the
end-to-end wiring that the resolved user_id reaches the orchestrator via the
gateway.
"""

import pytest
from starlette.datastructures import Headers
from starlette.testclient import TestClient

import auth
import router
from auth import (
    SANDBOX_USER_ID,
    extract_bearer_token,
    resolve_user_id,
    verify_supabase_jwt,
)
from main import create_app
from models import OrchestrationOutcome, TokenUsage


class _Req:
    """Minimal stand-in for starlette.Request: only .headers is read by auth."""

    def __init__(self, headers=None):
        self.headers = Headers(headers or {})


# --- extract_bearer_token() : header parsing --------------------------------

@pytest.mark.parametrize(
    "header,expected",
    [
        (None, None),                       # no Authorization header
        ("Bearer abc.def.ghi", "abc.def.ghi"),
        ("bearer abc.def.ghi", "abc.def.ghi"),  # scheme is case-insensitive
        ("BEARER tok", "tok"),
        ("Bearer    tok   ", "tok"),        # surrounding whitespace trimmed
        ("Basic abc", None),                # wrong scheme
        ("Bearer", None),                   # scheme only, no credential
        ("Bearer   ", None),                # empty credential
        ("abc.def.ghi", None),              # no scheme
    ],
)
def test_extract_bearer_token(header, expected):
    headers = {"Authorization": header} if header is not None else {}
    assert extract_bearer_token(_Req(headers)) == expected


# --- resolve_user_id() : sandbox fallback + verification seam ----------------

def test_resolve_no_token_is_sandbox_user():
    assert resolve_user_id(_Req()) == SANDBOX_USER_ID


def test_resolve_unverified_token_falls_back_to_sandbox():
    # Verification is a disabled hook today, so a present-but-unverifiable token
    # is IGNORED (never trusted), resolving to the sandbox user.
    assert resolve_user_id(_Req({"Authorization": "Bearer forged"})) == SANDBOX_USER_ID


def test_verify_hook_is_disabled():
    # Explicitly documents the security posture: no signing creds → always None.
    assert verify_supabase_jwt("any.jwt.here") is None


def test_resolve_returns_verified_subject_when_verification_wired(monkeypatch):
    # Proves the seam: once verify_supabase_jwt returns a subject, resolve_user_id
    # surfaces it verbatim. This is the test that flips to real behavior when
    # HS256/JWKS verification lands.
    monkeypatch.setattr(auth, "verify_supabase_jwt", lambda token: "user-123")
    assert resolve_user_id(_Req({"Authorization": "Bearer good"})) == "user-123"


# --- end-to-end wiring: resolved user_id reaches the orchestrator -----------

_INTENT_BODY = {
    "intent": "chat",
    "confidence": 0.95,  # clears the default 0.5 threshold → accepted
    "raw_input": "hi",
    "source": "test",
}


def _stub_outcome():
    return OrchestrationOutcome(
        status="completed", nodes_executed=[], messages=[], usage=TokenUsage(), steps=0
    )


@pytest.mark.parametrize(
    "headers,expected_user_id",
    [
        ({}, SANDBOX_USER_ID),                                  # no token
        ({"Authorization": "Bearer forged"}, SANDBOX_USER_ID),  # unverified → sandbox
    ],
)
def test_intent_threads_user_id_into_orchestration(monkeypatch, headers, expected_user_id):
    captured = {}

    async def fake_run(payload, user_id=SANDBOX_USER_ID):
        captured["user_id"] = user_id
        return _stub_outcome()

    monkeypatch.setattr(router, "run_orchestration", fake_run)
    client = TestClient(create_app())
    resp = client.post("/intent", json=_INTENT_BODY, headers=headers)

    assert resp.status_code == 200
    assert captured["user_id"] == expected_user_id
