"""Read a secret from a mounted file, falling back to the environment.

WHY
Every secret used to arrive as an environment variable, which meant its value was
visible in `docker inspect`, in `printenv` inside the container, and inherited by
every child process. That is not a privilege boundary on this host — the only
docker-group member is the sole operator, and docker group membership is
root-equivalent — but it is a real *accidental exposure* surface, and it has
already bitten: on 2026-08-06 a Supabase service-role key reached a log file via a
Python traceback.

So the high-blast-radius secrets (the three Supabase service-role keys, which
bypass RLS entirely, and the JWT secret, which could forge logins) are now mounted
as read-only files and blanked in the container environment.

WHAT THIS DOES NOT DO
Anyone who can `docker exec` can still read the mounted file, and code already
holding the value can still leak it in a traceback. This narrows the surface; it
does not make the secret unreachable.

CONTRACT
`secret("FOO")` returns, in order of preference:
  1. the contents of the file at `$FOO_FILE`, if that variable is set and the file
     is readable and non-empty
  2. `$FOO`, if set and non-empty
  3. `default`

An EMPTY environment variable is treated as absent, which matters: Docker Compose
cannot unset a variable inherited from `env_file:`, only override it — so the
compose file sets these to "" and the file takes over. Without that rule the empty
string would win and every Supabase call would fail authentication.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Read once per path. These files do not change while the process lives, and
# re-reading on every Supabase call would put a filesystem hit in the hot path.
_cache: dict[str, str] = {}


def secret(name: str, default: str | None = None) -> str | None:
    """Resolve a secret from ``$NAME_FILE``, then ``$NAME``, then ``default``."""
    path = (os.environ.get(f"{name}_FILE") or "").strip()
    if path:
        cached = _cache.get(path)
        if cached is not None:
            return cached
        try:
            value = Path(path).read_text(encoding="utf-8").strip()
        except OSError as exc:
            # Fall through to the environment rather than raising: a missing mount
            # should degrade to the old behaviour, not take the service down.
            logger.warning("secret %s: cannot read %s (%s); falling back to env",
                           name, path, type(exc).__name__)
        else:
            if value:
                _cache[path] = value
                return value
            logger.warning("secret %s: file %s is empty; falling back to env", name, path)

    env = os.environ.get(name)
    return env if env else default


def reset_cache() -> None:
    """Drop the cached file contents. For tests, and after rotating a secret."""
    _cache.clear()
