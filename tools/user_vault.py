"""User-isolated Obsidian-vault filesystem tools for the LangGraph supervisor.

These are the tools the tool_execution node exposes so the graph can read,
search, and write the caller's Markdown notes — the vault localized to
``/tmp/vaults/<user_id>/`` by ``services.storage_sync`` (Phase 2). The same vault
root is reused here so the tools operate on exactly what was synced.

SECURITY — strict per-user isolation is the whole point of this module:

  * The active ``user_id`` is resolved from the LangGraph **state** via
    :class:`~langgraph.prebuilt.InjectedState`, never from a tool argument. It is
    stripped from the schema the model sees, so the LLM (and, transitively, the
    end user) cannot supply, spoof, or override it. Every filesystem operation is
    scoped to that state-resolved identity's directory.
  * Every path is resolved with :func:`os.path.realpath` and then checked to be
    contained within the user's vault directory. This defeats ``..`` traversal,
    absolute-path override (``/etc/passwd``), and symlink escapes alike — a
    resolved path that lands outside the user's folder raises
    :class:`VaultAccessError` and the operation is refused.
  * A ``user_id`` that itself contains a path separator or ``..`` is rejected, so
    even a malformed identity cannot widen the boundary (defense in depth).

The pure ``read_note``/``search_vault``/``write_note`` helpers hold the security
logic and raise on violation (clean to unit-test). The ``@tool`` wrappers inject
the state ``user_id`` and degrade to an ``error: ...`` string rather than raising,
so a bad tool call never crashes the graph — consistent with the rest of the
codebase's "degrade, never crash" philosophy.
"""

from __future__ import annotations

import os
import re
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

# Reuse storage_sync's vault root so these tools read exactly what was synced.
from services.storage_sync import DEFAULT_VAULT_SYNC_ROOT, VAULT_SYNC_ROOT_ENV

VAULT_SUFFIX = ".md"
# Cap search output so a huge vault cannot flood the model context / logs.
MAX_SEARCH_MATCHES = 200


class VaultAccessError(Exception):
    """Raised when a path would escape the caller's isolated vault directory."""


# --- path resolution (the isolation boundary) -------------------------------

def _vault_root() -> str:
    """Local root holding every user's vault (env-overridable, read at call time)."""
    return os.environ.get(VAULT_SYNC_ROOT_ENV) or DEFAULT_VAULT_SYNC_ROOT


def _user_dir(user_id: str) -> str:
    """Absolute, real path of ``user_id``'s vault directory.

    Rejects a ``user_id`` that could itself widen the boundary (empty, contains a
    path separator or NUL, or is a ``.``/``..`` component). The result is
    realpath-resolved so it is a stable base for containment checks.
    """
    if (
        not user_id
        or user_id in (".", "..")
        or "\x00" in user_id
        or os.sep in user_id
        or (os.altsep and os.altsep in user_id)
    ):
        raise VaultAccessError(f"invalid user_id: {user_id!r}")
    return os.path.realpath(os.path.join(_vault_root(), user_id))


def _resolve_within_vault(user_id: str, filename: str) -> str:
    """Resolve ``filename`` against the user's vault and prove it stays inside.

    Returns the real absolute path. Raises :class:`VaultAccessError` if the
    resolved path escapes the user's directory by any means (``..``, an absolute
    path, or a symlink pointing out). ``os.path.join`` lets an absolute
    ``filename`` override the base, and ``realpath`` collapses ``..`` and follows
    symlinks — so the single containment check below covers all three vectors.
    """
    if not filename or "\x00" in filename:
        raise VaultAccessError(f"invalid filename: {filename!r}")
    base = _user_dir(user_id)
    target = os.path.realpath(os.path.join(base, filename))
    if target != base and not target.startswith(base + os.sep):
        raise VaultAccessError(
            f"path {filename!r} escapes vault for user {user_id!r}"
        )
    return target


# --- pure operations (raise on violation; unit-tested directly) -------------

def read_note(user_id: str, filename: str) -> str:
    """Return the text content of ``filename`` from ``user_id``'s vault."""
    path = _resolve_within_vault(user_id, filename)
    if os.path.isdir(path):
        raise IsADirectoryError(f"{filename!r} is a directory")
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def search_vault(user_id: str, query: str) -> list[dict]:
    """Case-insensitive text/regex search across the user's ``.md`` files.

    ``query`` is compiled as a regex (case-insensitive); an invalid pattern falls
    back to a literal substring match so a stray metacharacter never errors.
    Returns up to :data:`MAX_SEARCH_MATCHES` ``{"file", "line_no", "line"}`` hits,
    where ``file`` is the path relative to the user's vault. Only files that stay
    inside the vault are searched (symlinks out are skipped).
    """
    try:
        pattern = re.compile(query, re.IGNORECASE)
    except re.error:
        pattern = re.compile(re.escape(query), re.IGNORECASE)

    base = _user_dir(user_id)
    if not os.path.isdir(base):
        return []

    matches: list[dict] = []
    for dirpath, _dirnames, filenames in os.walk(base, followlinks=False):
        for name in filenames:
            if not name.endswith(VAULT_SUFFIX):
                continue
            full = os.path.realpath(os.path.join(dirpath, name))
            # Skip anything that (via a symlink) resolves outside the vault.
            if full != base and not full.startswith(base + os.sep):
                continue
            try:
                with open(full, encoding="utf-8", errors="replace") as fh:
                    for line_no, line in enumerate(fh, start=1):
                        if pattern.search(line):
                            matches.append({
                                "file": os.path.relpath(full, base),
                                "line_no": line_no,
                                "line": line.rstrip("\n"),
                            })
                            if len(matches) >= MAX_SEARCH_MATCHES:
                                return matches
            except OSError:
                continue
    return matches


def write_note(user_id: str, filename: str, content: str) -> str:
    """Create or overwrite ``filename`` in ``user_id``'s vault; return its path."""
    path = _resolve_within_vault(user_id, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


def _format_matches(matches: list[dict]) -> str:
    """Render search hits for a tool result string."""
    if not matches:
        return "no matches"
    return "\n".join(f"{m['file']}:{m['line_no']}: {m['line']}" for m in matches)


# --- LangGraph tool wrappers (user_id injected from state) -------------------

@tool
def read_user_note(filename: str, state: Annotated[dict, InjectedState]) -> str:
    """Read the full content of a note from the current user's vault.

    ``filename`` is relative to the user's own vault; paths that escape it are
    refused. Returns the file content, or an ``error: ...`` message.
    """
    try:
        return read_note(state["user_id"], filename)
    except Exception as exc:  # never crash the graph on a bad tool call
        return f"error: {type(exc).__name__}: {exc}"


@tool
def search_user_vault(query: str, state: Annotated[dict, InjectedState]) -> str:
    """Case-insensitive text/regex search across the current user's ``.md`` notes.

    Returns ``file:line: text`` hits from the user's own vault only, or
    ``no matches`` / an ``error: ...`` message.
    """
    try:
        return _format_matches(search_vault(state["user_id"], query))
    except Exception as exc:
        return f"error: {type(exc).__name__}: {exc}"


@tool
def write_user_note(
    filename: str, content: str, state: Annotated[dict, InjectedState]
) -> str:
    """Create or update a note in the current user's vault.

    ``filename`` is relative to the user's own vault; paths that escape it are
    refused. Returns a confirmation with the written path, or an ``error: ...``.
    """
    try:
        path = write_note(state["user_id"], filename, content)
        return f"wrote {os.path.basename(path)}"
    except Exception as exc:
        return f"error: {type(exc).__name__}: {exc}"


# The tool set exposed to the orchestrator / tool_execution node.
USER_VAULT_TOOLS = [read_user_note, search_user_vault, write_user_note]

# Direct-dispatch table (name -> callable(user_id, args) -> str). Used by the
# tool_execution node so the user_id ALWAYS comes from graph state, never args.
_DISPATCH = {
    "read_user_note": lambda uid, a: read_note(uid, a["filename"]),
    "search_user_vault": lambda uid, a: _format_matches(search_vault(uid, a["query"])),
    "write_user_note": lambda uid, a: write_note(uid, a["filename"], a["content"]),
}


def run_vault_tool(name: str, user_id: str, args: dict | None = None) -> str:
    """Execute a registered vault tool by name with a state-resolved ``user_id``.

    The node passes ``user_id`` straight from graph state, so no tool argument can
    redirect the operation to another user's folder. Never raises: an unknown tool
    or a refused/failed operation is returned as an ``error: ...`` string.
    """
    fn = _DISPATCH.get(name)
    if fn is None:
        return f"error: unknown tool {name!r}"
    try:
        return str(fn(user_id, args or {}))
    except Exception as exc:
        return f"error: {type(exc).__name__}: {exc}"
