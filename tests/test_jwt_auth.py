"""Tests for local HS256 Supabase JWT verification (auth.verify_supabase_jwt).

Covers the production crypto path: valid UUID extraction, expiry, bad/absent
signatures, algorithm/audience enforcement, the graceful missing-secret
fallback, and the resolve_user_id boundary (verified UUID vs. sandbox). Tokens
are minted in-test with PyJWT against a throwaway secret injected via
monkeypatch.setenv — no real Supabase project or network involved.
"""

import time
import uuid

import jwt
import pytest
from starlette.datastructures import Headers

import auth
from auth import SANDBOX_USER_ID, resolve_user_id, verify_supabase_jwt

SECRET = "test-hs256-signing-secret-000000000000"


class _Req:
    """Minimal starlette.Request stand-in: only .headers is read by auth."""

    def __init__(self, headers=None):
        self.headers = Headers(headers or {})


def _make_token(secret=SECRET, sub=None, aud="authenticated", exp_delta=3600, alg="HS256", key=None, iss=None):
    """Mint a JWT. `key` overrides the signing key (for wrong-secret cases)."""
    sub = sub or str(uuid.uuid4())
    now = int(time.time())
    payload = {"sub": sub, "iat": now, "exp": now + exp_delta}
    if aud is not None:
        payload["aud"] = aud
    if iss is not None:
        payload["iss"] = iss
    token = jwt.encode(payload, key if key is not None else secret, algorithm=alg)
    return token, sub


@pytest.fixture
def secret_set(monkeypatch):
    """Configure the signing secret for the verification path."""
    monkeypatch.setenv(auth.SUPABASE_JWT_SECRET_ENV, SECRET)


# --- valid UUID extraction --------------------------------------------------

def test_valid_token_returns_sub_uuid(secret_set):
    token, sub = _make_token(sub="6f3b7c2a-1e4d-4a9b-8c2f-abcdef012345")
    assert verify_supabase_jwt(token) == sub
    # sanity: it really is the UUID we signed, returned verbatim
    assert verify_supabase_jwt(token) == "6f3b7c2a-1e4d-4a9b-8c2f-abcdef012345"


def test_anon_audience_is_accepted(secret_set):
    token, sub = _make_token(aud="anon")
    assert verify_supabase_jwt(token) == sub


def test_authenticated_audience_is_accepted(secret_set):
    token, sub = _make_token(aud="authenticated")
    assert verify_supabase_jwt(token) == sub


# --- expiry -----------------------------------------------------------------

def test_expired_token_is_rejected(secret_set):
    token, _ = _make_token(exp_delta=-60)  # expired a minute ago
    assert verify_supabase_jwt(token) is None


# --- signatures / algorithm -------------------------------------------------

def test_wrong_secret_signature_is_rejected(secret_set):
    # signed with a different key than the configured SUPABASE_JWT_SECRET
    token, _ = _make_token(key="a-totally-different-secret-999999999999")
    assert verify_supabase_jwt(token) is None


def test_unsigned_alg_none_is_rejected(secret_set):
    # alg=none (no signature) must not be accepted when we require HS256
    token, _ = _make_token(alg="none", key="")
    assert verify_supabase_jwt(token) is None


def test_alg_outside_hs256_allowlist_is_rejected(secret_set):
    # A token signed with a valid-but-disallowed algorithm (HS512) — even using the
    # correct secret — must be rejected: only HS256 is on the allowlist.
    now = int(time.time())
    payload = {"sub": str(uuid.uuid4()), "aud": "authenticated", "iat": now, "exp": now + 60}
    token = jwt.encode(payload, SECRET, algorithm="HS512")
    assert verify_supabase_jwt(token) is None


# --- audience ---------------------------------------------------------------

def test_wrong_audience_is_rejected(secret_set):
    token, _ = _make_token(aud="some-other-service")
    assert verify_supabase_jwt(token) is None


def test_missing_audience_is_rejected(secret_set):
    token, _ = _make_token(aud=None)  # no aud claim, but we require one
    assert verify_supabase_jwt(token) is None


# --- malformed / missing sub ------------------------------------------------

def test_malformed_token_is_rejected(secret_set):
    assert verify_supabase_jwt("not-a-jwt") is None
    assert verify_supabase_jwt("a.b.c") is None


def test_missing_sub_is_rejected(secret_set):
    now = int(time.time())
    token = jwt.encode({"aud": "authenticated", "iat": now, "exp": now + 60}, SECRET, algorithm="HS256")
    assert verify_supabase_jwt(token) is None


# --- graceful missing-secret fallback ---------------------------------------

def test_missing_secret_returns_none_without_crashing(monkeypatch):
    monkeypatch.delenv(auth.SUPABASE_JWT_SECRET_ENV, raising=False)
    token, _ = _make_token()  # a perfectly valid token...
    assert verify_supabase_jwt(token) is None  # ...but no secret configured -> None


def test_blank_secret_is_treated_as_missing(monkeypatch):
    monkeypatch.setenv(auth.SUPABASE_JWT_SECRET_ENV, "   ")
    token, _ = _make_token()
    assert verify_supabase_jwt(token) is None


# --- resolve_user_id integration --------------------------------------------

def test_resolve_returns_real_uuid_for_valid_bearer(secret_set):
    token, sub = _make_token(sub="11111111-2222-3333-4444-555555555555")
    assert resolve_user_id(_Req({"Authorization": f"Bearer {token}"})) == sub


def test_resolve_expired_bearer_falls_back_to_sandbox(secret_set):
    token, _ = _make_token(exp_delta=-10)
    assert resolve_user_id(_Req({"Authorization": f"Bearer {token}"})) == SANDBOX_USER_ID


def test_resolve_no_header_is_sandbox(secret_set):
    assert resolve_user_id(_Req()) == SANDBOX_USER_ID


def test_resolve_cf_access_service_token_is_sandbox(secret_set):
    # Automation authenticated at the CF edge (no Supabase Bearer) → sandbox user.
    headers = {
        "CF-Access-Client-Id": "5ea74e27c23920f2d43791bddacc75a0.access",
        "CF-Access-Client-Secret": "deadbeef" * 8,
    }
    assert resolve_user_id(_Req(headers)) == SANDBOX_USER_ID


def test_resolve_valid_bearer_beats_sandbox_even_with_cf_headers(secret_set):
    # A verifiable Supabase token wins even if CF Access headers are also present.
    token, sub = _make_token(sub="99999999-8888-7777-6666-555555555555")
    headers = {
        "Authorization": f"Bearer {token}",
        "CF-Access-Client-Id": "5ea74e27c23920f2d43791bddacc75a0.access",
        "CF-Access-Client-Secret": "deadbeef" * 8,
    }
    assert resolve_user_id(_Req(headers)) == sub


# --- asymmetric (ES256) verification via JWKS -------------------------------

from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402


def _es256_token(pub_kid="07b5af1d-01c7-4046-99be-c376352df247", sub=None, aud="authenticated", exp_delta=3600):
    """Mint an ES256 token with a fresh EC key; return (token, sub, public_key)."""
    priv = ec.generate_private_key(ec.SECP256R1())
    sub = sub or str(uuid.uuid4())
    now = int(time.time())
    payload = {"sub": sub, "aud": aud, "iat": now, "exp": now + exp_delta}
    token = jwt.encode(payload, priv, algorithm="ES256", headers={"kid": pub_kid})
    return token, sub, priv.public_key()


def test_es256_token_verified_via_jwks(monkeypatch):
    token, sub, pub = _es256_token(sub="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    # JWKS returns the matching public key (no network).
    monkeypatch.setattr(auth, "_asymmetric_signing_key", lambda t: pub)
    assert verify_supabase_jwt(token) == sub


def test_es256_expired_is_rejected(monkeypatch):
    token, _, pub = _es256_token(exp_delta=-30)
    monkeypatch.setattr(auth, "_asymmetric_signing_key", lambda t: pub)
    assert verify_supabase_jwt(token) is None


def test_es256_wrong_key_is_rejected(monkeypatch):
    token, _, _ = _es256_token()
    other_pub = ec.generate_private_key(ec.SECP256R1()).public_key()
    monkeypatch.setattr(auth, "_asymmetric_signing_key", lambda t: other_pub)
    assert verify_supabase_jwt(token) is None


def test_es256_no_jwks_match_falls_back_to_hs256(monkeypatch):
    # No JWKS key for this kid -> fall back to HS256. An ES256 token can't pass
    # HS256 (alg mismatch), so the result is a safe None even with a secret set.
    monkeypatch.setenv(auth.SUPABASE_JWT_SECRET_ENV, SECRET)
    token, _, _ = _es256_token()
    monkeypatch.setattr(auth, "_asymmetric_signing_key", lambda t: None)
    assert verify_supabase_jwt(token) is None


def test_es256_no_jwks_and_no_secret_is_none(monkeypatch):
    monkeypatch.delenv(auth.SUPABASE_JWT_SECRET_ENV, raising=False)
    token, _, _ = _es256_token()
    monkeypatch.setattr(auth, "_asymmetric_signing_key", lambda t: None)
    assert verify_supabase_jwt(token) is None


def test_resolve_es256_bearer_returns_uuid(monkeypatch):
    token, sub, pub = _es256_token(sub="12121212-3434-5656-7878-909090909090")
    monkeypatch.setattr(auth, "_asymmetric_signing_key", lambda t: pub)
    assert resolve_user_id(_Req({"Authorization": f"Bearer {token}"})) == sub


def test_jwks_url_default_is_wellknown_not_rpc():
    # Guard against regressing to the non-existent /rest/v1/rpc/get_jwks path.
    assert auth.DEFAULT_SUPABASE_JWKS_URL.endswith("/auth/v1/.well-known/jwks.json")
    assert "rpc/get_jwks" not in auth.DEFAULT_SUPABASE_JWKS_URL


# --- optional issuer pinning (L1) -------------------------------------------

ISSUER = "https://ulzhtdnjwikcadtskzgi.supabase.co/auth/v1"


def test_issuer_unset_does_not_enforce_iss(secret_set, monkeypatch):
    # Default: no SUPABASE_JWT_ISSUER -> issuer is not checked (backward compatible).
    monkeypatch.delenv(auth.SUPABASE_JWT_ISSUER_ENV, raising=False)
    token, sub = _make_token(iss="https://some-other-project.supabase.co/auth/v1")
    assert verify_supabase_jwt(token) == sub
    token2, sub2 = _make_token(iss=None)  # no iss claim at all
    assert verify_supabase_jwt(token2) == sub2


def test_matching_issuer_is_accepted(secret_set, monkeypatch):
    monkeypatch.setenv(auth.SUPABASE_JWT_ISSUER_ENV, ISSUER)
    token, sub = _make_token(iss=ISSUER)
    assert verify_supabase_jwt(token) == sub


def test_wrong_issuer_is_rejected(secret_set, monkeypatch):
    monkeypatch.setenv(auth.SUPABASE_JWT_ISSUER_ENV, ISSUER)
    token, _ = _make_token(iss="https://attacker-project.supabase.co/auth/v1")
    assert verify_supabase_jwt(token) is None
