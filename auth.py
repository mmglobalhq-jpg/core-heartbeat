"""Bearer-token → user identity resolution for core-heartbeat.

Phase 1 of the multi-user memory bridge. The frontend (core-chat) will forward a
Supabase session JWT as an ``Authorization: Bearer <token>`` header; this module
extracts it and resolves a stable ``user_id`` that the orchestrator threads
through GraphState. When no token is present (local/dev calls, uptime probes) the
resolution falls back to :data:`SANDBOX_USER_ID` so existing single-user flows
keep working unchanged.

SECURITY — signature verification is intentionally NOT implemented yet. This
project has no Supabase project or JWT signing credentials wired up, so
:func:`verify_supabase_jwt` is a disabled hook that always returns ``None``. We
never decode-and-trust an unverified token: a request that carries a token but
hits the unconfigured verifier resolves to the sandbox user (the token is
ignored, not trusted). When credentials exist, implement the signature check
inside :func:`verify_supabase_jwt` — either HS256 with the project's JWT secret
or, for current Supabase projects, an asymmetric key (ES256/RS256) verified
against the project's JWKS endpoint — and return the token's ``sub`` claim.
"""

from __future__ import annotations

from fastapi import Request

# Identity used when a caller presents no (trustworthy) token. Kept deliberately
# distinct from any real Supabase user id so sandbox data never collides with a
# verified user's memory once multi-user is live.
SANDBOX_USER_ID = "sandbox-user"

_BEARER_PREFIX = "bearer "


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


def verify_supabase_jwt(token: str) -> str | None:
    """DISABLED verification hook: return the verified ``user_id`` or ``None``.

    Not yet implemented — no Supabase signing credentials are configured, so this
    returns ``None`` and callers fall back to the sandbox user rather than
    trusting an unverified token. Wire real verification here once credentials
    exist:

      * HS256 — ``jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"],
        audience="authenticated")`` then return ``claims["sub"]``.
      * Asymmetric (current Supabase) — fetch/cache the project JWKS and verify
        the ES256/RS256 signature, then return ``claims["sub"]``.

    Deliberately NOT doing ``jwt.decode(..., options={"verify_signature": False})``
    and trusting the result — an attacker could forge any ``sub``.
    """
    return None


def resolve_user_id(request: Request) -> str:
    """FastAPI dependency: resolve the caller's ``user_id`` for this request.

    No token → sandbox user. Token present → its verified subject, or the sandbox
    user when verification is unconfigured/fails. Never trusts an unverified
    token. Synchronous today; when JWKS verification lands this can become
    ``async`` and FastAPI will await it transparently.
    """
    token = extract_bearer_token(request)
    if token is None:
        return SANDBOX_USER_ID
    return verify_supabase_jwt(token) or SANDBOX_USER_ID
