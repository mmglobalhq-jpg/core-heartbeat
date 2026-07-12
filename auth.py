"""Bearer-token → user identity resolution for core-heartbeat.

The multi-user memory bridge. The frontend (core-chat) forwards a Supabase
session JWT as an ``Authorization: Bearer <token>`` header; this module extracts
it, **cryptographically verifies it locally** (HS256 against the project's JWT
secret), and resolves the token's ``sub`` claim — the caller's authentic Supabase
user UUID — which the orchestrator threads through GraphState.

SECURITY posture:

* Verification is real and local (:func:`verify_supabase_jwt`): the signature is
  checked with the project's ``SUPABASE_JWT_SECRET`` (HS256) and the ``aud`` claim
  must be ``authenticated`` or ``anon``. Expiry is enforced. An unverifiable token
  is never trusted — it is ignored and the caller falls back to the sandbox user.
* If ``SUPABASE_JWT_SECRET`` is not configured, verification degrades gracefully
  to ``None`` (with a descriptive warning) rather than crashing — the service
  stays bootable and every caller resolves to the sandbox user.
* The sandbox boundary (:data:`SANDBOX_USER_ID`) is preserved for callers with no
  Authorization header at all (local/dev, uptime probes, the test suite) and for
  automation that authenticated at the Cloudflare edge via the Access service
  token (``CF-Access-Client-Id`` / ``CF-Access-Client-Secret``) — Access already
  gate-kept the request, so inside the trust boundary it runs as the shared
  sandbox user.

Signing algorithms: both are supported, selected by the token's own ``alg`` header.
An **asymmetric** token (RS256/ES256 — how current Supabase projects sign) is
verified against the project's published **JWKS** (public keys fetched once and
cached in memory). A **symmetric** token (HS256 — legacy shared secret) is verified
against ``SUPABASE_JWT_SECRET``; this is also the fallback if the JWKS has no key
matching an asymmetric token's ``kid``.

Note: the JWKS URL defaults to the standard Supabase location
``/auth/v1/.well-known/jwks.json`` (override with ``SUPABASE_JWKS_URL``). The
``/rest/v1/rpc/get_jwks`` RPC path is NOT a real endpoint (no such DB function) —
do not point this at it.
"""

from __future__ import annotations

import logging
import os

import jwt
from fastapi import Request

logger = logging.getLogger(__name__)

# Identity used when a caller presents no (trustworthy) token. Kept deliberately
# distinct from any real Supabase user id so sandbox data never collides with a
# verified user's memory.
SANDBOX_USER_ID = "sandbox-user"

_BEARER_PREFIX = "bearer "

# HS256 verification config (env-read at call time so deployments/tests can set it
# without a rebuild, matching the rest of the codebase).
SUPABASE_JWT_SECRET_ENV = "SUPABASE_JWT_SECRET"
JWT_ALGORITHMS = ["HS256"]
# Asymmetric algorithms verified via JWKS (public keys). Selected by the token's
# own alg header; the private keys never leave Supabase.
ASYMMETRIC_ALGORITHMS = ["RS256", "ES256"]
# A Supabase access token's aud is "authenticated" (signed-in) or "anon".
JWT_AUDIENCES = ["authenticated", "anon"]

# JWKS endpoint for asymmetric verification. Defaults to the standard Supabase
# well-known location (env-overridable). NOT the /rest/v1/rpc/get_jwks RPC path,
# which does not exist on the project.
SUPABASE_JWKS_URL_ENV = "SUPABASE_JWKS_URL"
DEFAULT_SUPABASE_JWKS_URL = (
    "https://ulzhtdnjwikcadtskzgi.supabase.co/auth/v1/.well-known/jwks.json"
)

# Optional issuer pinning (hardening). When SUPABASE_JWT_ISSUER is set, the token's
# `iss` claim must match it exactly, so a token minted by a *different* Supabase
# project that happens to share the JWKS/secret is rejected. Left unset by default
# so existing verification behavior is unchanged (opt-in, no breakage). For this
# deployment set it to: https://ulzhtdnjwikcadtskzgi.supabase.co/auth/v1
SUPABASE_JWT_ISSUER_ENV = "SUPABASE_JWT_ISSUER"

# Cloudflare Access service-token headers — automation authenticated at the edge.
_CF_ACCESS_CLIENT_ID_HEADER = "cf-access-client-id"
_CF_ACCESS_CLIENT_SECRET_HEADER = "cf-access-client-secret"


def extract_bearer_token(request: Request) -> str | None:
    """Return the bearer credential from the Authorization header, or ``None``.

    Tolerant of the usual shapes: missing header, non-Bearer scheme, or an empty
    credential all yield ``None`` (the caller then falls back to the sandbox
    user). The scheme match is case-insensitive per RFC 7235.
    """
    header = request.headers.get("authorization")
    if not header or not header.lower().startswith(_BEARER_PREFIX):
        return None
    token = header[len(_BEARER_PREFIX) :].strip()
    return token or None


# Lazily-built, in-memory-caching JWKS client (PyJWKClient caches fetched keys).
_jwks_client: jwt.PyJWKClient | None = None


def _get_jwks_client() -> jwt.PyJWKClient | None:
    """Return the memoized JWKS client, or ``None`` if it can't be constructed.

    Built once from ``SUPABASE_JWKS_URL`` (default: the well-known endpoint) and
    reused, so public keys are fetched at most once and cached in memory.
    """
    global _jwks_client
    if _jwks_client is None:
        url = os.environ.get(SUPABASE_JWKS_URL_ENV) or DEFAULT_SUPABASE_JWKS_URL
        try:
            _jwks_client = jwt.PyJWKClient(url, cache_keys=True)
        except Exception as exc:  # never crash the auth path on a bad URL
            logger.warning("could not initialize JWKS client (%s): %s", url, exc)
            return None
    return _jwks_client


def _asymmetric_signing_key(token: str):
    """Public signing key for an asymmetric ``token`` from the cached JWKS.

    Returns the key object, or ``None`` when the JWKS is unreachable or has no key
    matching the token's ``kid`` (the caller then falls back to HS256). Never
    raises.
    """
    client = _get_jwks_client()
    if client is None:
        return None
    try:
        return client.get_signing_key_from_jwt(token).key
    except Exception as exc:  # unreachable JWKS / unknown kid / parse error
        logger.info("no JWKS signing key for token: %s: %s", type(exc).__name__, exc)
        return None


def _decode_and_extract_sub(token: str, key, algorithms) -> str | None:
    """Verify ``token`` with ``key``/``algorithms`` and return its ``sub``, or ``None``.

    Enforces expiry and the ``authenticated``/``anon`` audience. Never raises.
    """
    issuer = os.environ.get(SUPABASE_JWT_ISSUER_ENV) or None  # None => not enforced
    try:
        payload = jwt.decode(
            token, key, algorithms=algorithms, audience=JWT_AUDIENCES, issuer=issuer
        )
    except jwt.PyJWTError as exc:  # expired, bad signature, wrong alg/aud/iss, malformed
        logger.info("JWT verification failed: %s: %s", type(exc).__name__, exc)
        return None
    sub = payload.get("sub")
    if not isinstance(sub, str) or not sub:
        logger.info("verified JWT carries no usable 'sub' claim; rejecting.")
        return None
    return sub


def _verify_hs256(token: str) -> str | None:
    """Verify a symmetric (HS256) token against ``SUPABASE_JWT_SECRET``.

    Missing/blank secret degrades gracefully to ``None`` with a warning.
    """
    secret = os.environ.get(SUPABASE_JWT_SECRET_ENV)
    if not secret or not secret.strip():
        logger.warning(
            "%s is not set; cannot verify this HS256 JWT — treating the caller as "
            "unverified (sandbox fallback).",
            SUPABASE_JWT_SECRET_ENV,
        )
        return None
    return _decode_and_extract_sub(token, secret, JWT_ALGORITHMS)


def verify_supabase_jwt(token: str) -> str | None:
    """Verify a Supabase JWT locally and return its ``sub`` (user UUID), or ``None``.

    The token's own ``alg`` header selects the path: an asymmetric token
    (RS256/ES256) is verified against the project's JWKS public keys (fetched once,
    cached in memory); a symmetric token (HS256) is verified against
    ``SUPABASE_JWT_SECRET``. If an asymmetric token's ``kid`` isn't found in the
    JWKS (or the JWKS is unreachable), verification falls back to the HS256 secret.

    Never raises — every failure mode (unreadable/malformed header, bad/absent
    signature, expired, wrong/missing audience, unknown key, missing ``sub``,
    unconfigured secret) maps to ``None`` so callers fall back to the sandbox user
    rather than trusting an unverified token.
    """
    try:
        alg = jwt.get_unverified_header(token).get("alg")
    except jwt.PyJWTError as exc:  # not a well-formed JWT
        logger.info("could not read JWT header: %s: %s", type(exc).__name__, exc)
        return None

    if alg in ASYMMETRIC_ALGORITHMS:
        key = _asymmetric_signing_key(token)
        if key is not None:
            return _decode_and_extract_sub(token, key, [alg])
        # No JWKS key matched this kid — fall back to the HS256 shared secret.

    return _verify_hs256(token)


def _has_cf_access_service_token(request: Request) -> bool:
    """True when the request carries both Cloudflare Access service-token headers."""
    headers = request.headers
    return bool(
        headers.get(_CF_ACCESS_CLIENT_ID_HEADER)
        and headers.get(_CF_ACCESS_CLIENT_SECRET_HEADER)
    )


def resolve_user_id(request: Request) -> str:
    """FastAPI dependency: resolve the caller's ``user_id`` for this request.

    A present, verifiable Supabase Bearer token resolves to its ``sub`` (the real
    user UUID). Any other case preserves the sandbox boundary: an unverifiable
    token is ignored (never trusted), no token falls back, and automation that
    reached us through the Cloudflare Access service token runs as the shared
    sandbox user. Never raises.
    """
    token = extract_bearer_token(request)
    user_id = SANDBOX_USER_ID
    if token is not None:
        verified = verify_supabase_jwt(token)
        if verified:
            user_id = verified
        # Token present but unverifiable → ignore it and fall through to sandbox.

    # No verified Supabase identity. Sandbox boundary (see module docstring):
    # unauthenticated local/dev + test calls, and CF Access service-token
    # automation that Access already gate-kept at the edge.
    if user_id == SANDBOX_USER_ID and _has_cf_access_service_token(request):
        logger.debug("CF Access service-token request resolved to the sandbox user.")

    # Don't log the resolved user id at INFO — it's identity (PII-adjacent) in every
    # access-log line. Debug only, and just whether we resolved a real user.
    logger.debug("resolved active identity: %s", "sandbox" if user_id == SANDBOX_USER_ID else "user")
    return user_id
