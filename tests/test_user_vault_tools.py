"""Tests for the user-isolated vault filesystem tools (tools/user_vault.py).

The emphasis is STRICT per-user isolation: user A can never read, search, or
write into user B's ``/tmp/vaults/<B>/`` folder, by any path vector — ``..``
traversal, an absolute path, a symlink pointing out, or a malformed user_id. The
vault root is redirected into ``tmp_path`` via env so nothing touches the real
``/tmp/vaults``.
"""

import os

import pytest

import orchestrator
import tools.user_vault as uv
from tools.user_vault import (
    VaultAccessError,
    read_note,
    read_user_note,
    run_vault_tool,
    search_user_vault,
    search_vault,
    write_note,
    write_user_note,
)


@pytest.fixture(autouse=True)
def _redirect_vault_root(monkeypatch, tmp_path):
    """Point the vault root at tmp_path for every test in this module."""
    root = tmp_path / "vaults"
    monkeypatch.setenv(uv.VAULT_SYNC_ROOT_ENV, str(root))
    return root


def _seed(root, user_id, rel, content):
    """Create <root>/<user_id>/<rel> with content; return its path."""
    path = root / user_id / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# --- happy path: a user operates within their own vault ---------------------

def test_read_own_note(_redirect_vault_root):
    _seed(_redirect_vault_root, "userA", "note.md", "content A")
    assert read_note("userA", "note.md") == "content A"


def test_write_then_read_roundtrip(_redirect_vault_root):
    path = write_note("userA", "new.md", "fresh")
    assert path == str(_redirect_vault_root / "userA" / "new.md")
    assert read_note("userA", "new.md") == "fresh"


def test_write_creates_nested_subdir_within_vault(_redirect_vault_root):
    write_note("userA", "projects/plan.md", "nested body")
    assert (_redirect_vault_root / "userA" / "projects" / "plan.md").read_text() == "nested body"


def test_search_is_case_insensitive_and_regex(_redirect_vault_root):
    _seed(_redirect_vault_root, "userA", "a.md", "Alpha Beta\nGamma")
    _seed(_redirect_vault_root, "userA", "b.md", "delta BETA line")
    hits = search_vault("userA", r"be.a")  # regex, case-insensitive -> matches "Beta"/"BETA"
    files = sorted({h["file"] for h in hits})
    assert files == ["a.md", "b.md"]


def test_search_missing_vault_returns_empty(_redirect_vault_root):
    assert search_vault("ghost-user", "anything") == []


def test_search_invalid_regex_falls_back_to_literal(_redirect_vault_root):
    _seed(_redirect_vault_root, "userA", "a.md", "cost is 50% off")
    hits = search_vault("userA", "50%")  # '%' is fine, but trailing metachars would error
    assert any(h["file"] == "a.md" for h in hits)


# --- ISOLATION: A must never reach B's folder -------------------------------

def test_read_cannot_traverse_to_other_user(_redirect_vault_root):
    _seed(_redirect_vault_root, "userB", "secret.md", "B's secret")
    with pytest.raises(VaultAccessError):
        read_note("userA", "../userB/secret.md")


def test_read_absolute_path_is_refused(_redirect_vault_root):
    with pytest.raises(VaultAccessError):
        read_note("userA", "/etc/passwd")


def test_read_deep_traversal_is_refused(_redirect_vault_root):
    with pytest.raises(VaultAccessError):
        read_note("userA", "../../../../etc/hosts")


def test_write_cannot_traverse_into_other_user(_redirect_vault_root):
    _seed(_redirect_vault_root, "userB", "keep.md", "original")
    with pytest.raises(VaultAccessError):
        write_note("userA", "../userB/keep.md", "HACKED")
    # B's file is untouched.
    assert (_redirect_vault_root / "userB" / "keep.md").read_text() == "original"


def test_write_absolute_path_is_refused(_redirect_vault_root, tmp_path):
    outside = tmp_path / "outside.md"
    with pytest.raises(VaultAccessError):
        write_note("userA", str(outside), "nope")
    assert not outside.exists()


def test_search_only_sees_own_vault(_redirect_vault_root):
    _seed(_redirect_vault_root, "userA", "mine.md", "shared-keyword here")
    _seed(_redirect_vault_root, "userB", "theirs.md", "shared-keyword there")
    hits = search_vault("userA", "shared-keyword")
    assert [h["file"] for h in hits] == ["mine.md"]  # B's match never appears


def test_symlink_escape_is_refused(_redirect_vault_root, tmp_path):
    # A symlink inside A's vault pointing at B's secret must not yield B's content.
    _seed(_redirect_vault_root, "userB", "secret.md", "B's secret")
    (_redirect_vault_root / "userA").mkdir(parents=True, exist_ok=True)
    link = _redirect_vault_root / "userA" / "escape.md"
    try:
        os.symlink(_redirect_vault_root / "userB" / "secret.md", link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform")
    with pytest.raises(VaultAccessError):
        read_note("userA", "escape.md")


def test_symlinked_file_excluded_from_search(_redirect_vault_root):
    _seed(_redirect_vault_root, "userB", "secret.md", "needle in B")
    (_redirect_vault_root / "userA").mkdir(parents=True, exist_ok=True)
    link = _redirect_vault_root / "userA" / "escape.md"
    try:
        os.symlink(_redirect_vault_root / "userB" / "secret.md", link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform")
    # A searching its own vault must not surface B's content via the symlink.
    assert search_vault("userA", "needle") == []


def test_malformed_user_id_is_refused(_redirect_vault_root):
    _seed(_redirect_vault_root, "userB", "secret.md", "B's secret")
    # A user_id carrying a separator/traversal cannot widen the boundary.
    for bad in ["../userB", "userB/..", "..", "a/b"]:
        with pytest.raises(VaultAccessError):
            read_note(bad, "secret.md")


# --- tool wrappers resolve user_id from injected state ----------------------

def test_tool_reads_from_injected_state(_redirect_vault_root):
    _seed(_redirect_vault_root, "userA", "note.md", "state-scoped content")
    out = read_user_note.invoke({"filename": "note.md", "state": {"user_id": "userA"}})
    assert out == "state-scoped content"


def test_tool_schema_hides_user_id():
    # The model can supply ONLY the documented args — never user_id/state.
    assert set(read_user_note.args) == {"filename"}
    assert set(search_user_vault.args) == {"query"}
    assert set(write_user_note.args) == {"filename", "content"}


def test_tool_traversal_returns_error_not_crash(_redirect_vault_root):
    _seed(_redirect_vault_root, "userB", "secret.md", "B's secret")
    out = read_user_note.invoke(
        {"filename": "../userB/secret.md", "state": {"user_id": "userA"}}
    )
    assert out.startswith("error:")
    assert "B's secret" not in out


def test_write_tool_scoped_to_state_user(_redirect_vault_root):
    out = write_user_note.invoke(
        {"filename": "n.md", "content": "hi", "state": {"user_id": "userA"}}
    )
    assert out == "wrote n.md"
    assert (_redirect_vault_root / "userA" / "n.md").read_text() == "hi"


# --- orchestrator integration: tool_execution enforces isolation ------------

def _tool_state(user_id, request):
    return {
        "user_id": user_id,
        "tool_request": request,
        "messages": [],
        "usage": None,
        "visited": [],
        "step": 1,
        "next": "",
        "status": "",
    }


def test_node_dispatches_tool_with_state_user_id(_redirect_vault_root):
    _seed(_redirect_vault_root, "userA", "note.md", "node content")
    update = orchestrator.tool_execution(
        _tool_state("userA", {"name": "read_user_note", "args": {"filename": "note.md"}})
    )
    assert update["messages"][0].content == "[tool:read_user_note] node content"
    assert update["visited"] == ["tool_execution"]


def test_node_isolation_blocks_cross_user_request(_redirect_vault_root):
    _seed(_redirect_vault_root, "userB", "secret.md", "B's secret")
    # Even a crafted request cannot read B while the run's user_id is A.
    update = orchestrator.tool_execution(
        _tool_state("userA", {"name": "read_user_note", "args": {"filename": "../userB/secret.md"}})
    )
    content = update["messages"][0].content
    assert "error:" in content
    assert "B's secret" not in content


def test_node_without_request_keeps_stub_behavior(_redirect_vault_root):
    update = orchestrator.tool_execution(_tool_state("userA", None))
    assert update["messages"][0].content == "[stub] tool executed"


def test_run_vault_tool_unknown_tool(_redirect_vault_root):
    assert run_vault_tool("does_not_exist", "userA", {}).startswith("error: unknown tool")
