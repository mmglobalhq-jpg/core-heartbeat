#!/usr/bin/env python3
"""Run the briefing job in development with the repo's .env loaded.

WHY THIS EXISTS RATHER THAN `env FOO=... python -m briefing.run`
Putting a key on a command line puts it in the process list and in shell history.
This reads the env file inside the process instead, so nothing sensitive is ever
an argument. Values are never printed — only the NAMES that were loaded.

    scripts/run_briefing_dev.py --user <uuid> --repo postgres --deliver file

Everything after this script's own flags is passed to briefing.run.
"""

from __future__ import annotations

import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The Resend key lives in the platform-wide env file, not the service's own.
ENV_FILES = (ROOT / ".env", ROOT.parent / ".env")

# Only these are loaded. A blanket load would pull production Supabase URLs and
# service-role keys into a development run, which is exactly the accident this
# build is meant not to have.
ALLOWED = {
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "WEB_SEARCH_MODEL",
    # Live-send testing only. Reading it here keeps the key out of argv, out of
    # shell history, and out of the service's own environment — the briefing
    # service still has no production mail credential, which is the documented
    # state in doc 30 §9.
    "RESEND_API_KEY",
}

# NOT loaded, deliberately: OLLAMA_URL. The .env value is the container-internal
# hostname, which does not resolve from the host — loading it silently took the
# local model offline for a whole run, and every write-up fell back to the
# publisher's summary while the run still reported success.


def load_env(path: pathlib.Path) -> list[str]:
    loaded: list[str] = []
    if not path.is_file():
        return loaded
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        if name not in ALLOWED or os.environ.get(name):
            continue
        os.environ[name] = value.strip().strip("'\"")
        loaded.append(name)
    return loaded


if __name__ == "__main__":
    names: list[str] = []
    for env_file in ENV_FILES:
        names.extend(load_env(env_file))
    # Names only. Never the values, never a length, never a prefix.
    print(f"[dev] loaded from .env: {', '.join(sorted(names)) or '(nothing)'}", file=sys.stderr)

    from briefing.run import main

    sys.exit(main(sys.argv[1:]))
