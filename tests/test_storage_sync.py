"""Tests for per-user vault sync (services/storage_sync.py, Phase 2).

Focus is path routing: the sandbox user reads a local mock folder and NEVER
touches S3, while a real user id downloads over the S3 endpoint under its
``<user_id>/`` prefix. All S3 access goes through a fake injected at the
``build_s3_client`` seam — NO real network, NO credentials, no spend. The sync
root and mock root are redirected into ``tmp_path`` via env so nothing writes to
the real ``/tmp/vaults`` or the repo's ``mock_vaults`` fixture.
"""

import asyncio
import os

import pytest

from auth import SANDBOX_USER_ID
import services.storage_sync as storage_sync
from services.storage_sync import sync_user_vault


# --- fake S3 client (no network) --------------------------------------------

class _FakeS3:
    """Minimal stand-in for a boto3 S3 client.

    ``list_objects_v2`` returns the scripted keys (optionally in truncated pages
    to exercise pagination); ``download_file`` records the call and writes a
    placeholder file so callers can assert on-disk results.
    """

    def __init__(self, pages):
        # pages: list of (keys, next_token|None) — one list_objects_v2 response each.
        self._pages = list(pages)
        self.list_calls: list[dict] = []
        self.downloaded: list[tuple[str, str, str]] = []
        self.uploaded: list[tuple[str, str, str]] = []  # (bucket, key, content)

    def list_objects_v2(self, **kwargs):
        self.list_calls.append(kwargs)
        keys, next_token = self._pages.pop(0)
        resp = {"Contents": [{"Key": k} for k in keys]}
        if next_token is not None:
            resp["IsTruncated"] = True
            resp["NextContinuationToken"] = next_token
        return resp

    def download_file(self, bucket, key, path):
        self.downloaded.append((bucket, key, path))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"content of {key}")

    def upload_file(self, filename, bucket, key):
        with open(filename, encoding="utf-8") as fh:
            self.uploaded.append((bucket, key, fh.read()))


def _single_page(keys):
    return _FakeS3([(keys, None)])


def _install_s3(monkeypatch, fake):
    monkeypatch.setattr(storage_sync, "build_s3_client", lambda: fake)


def _redirect_roots(monkeypatch, tmp_path):
    """Point sync + mock roots at tmp_path so tests never touch real dirs."""
    sync_root = tmp_path / "vaults"
    mock_root = tmp_path / "mock_vaults"
    monkeypatch.setenv(storage_sync.VAULT_SYNC_ROOT_ENV, str(sync_root))
    monkeypatch.setenv(storage_sync.VAULT_MOCK_ROOT_ENV, str(mock_root))
    return sync_root, mock_root


def _run(user_id):
    return asyncio.run(sync_user_vault(user_id))


def _listdir_md(path):
    return sorted(p for p in os.listdir(path) if p.endswith(".md"))


# --- sandbox path: local mock, never S3 -------------------------------------

def test_sandbox_reads_mock_folder(monkeypatch, tmp_path):
    sync_root, mock_root = _redirect_roots(monkeypatch, tmp_path)
    src = mock_root / SANDBOX_USER_ID
    src.mkdir(parents=True)
    (src / "note.md").write_text("hello vault", encoding="utf-8")

    dest = _run(SANDBOX_USER_ID)

    assert dest == str(sync_root / SANDBOX_USER_ID)
    assert _listdir_md(dest) == ["note.md"]
    assert (sync_root / SANDBOX_USER_ID / "note.md").read_text(encoding="utf-8") == "hello vault"


def test_sandbox_never_calls_s3(monkeypatch, tmp_path):
    _redirect_roots(monkeypatch, tmp_path)

    def _boom():
        raise AssertionError("sandbox path must not build an S3 client")

    monkeypatch.setattr(storage_sync, "build_s3_client", _boom)
    # No mock folder created — must still succeed (empty vault), never hit S3.
    dest = _run(SANDBOX_USER_ID)
    assert os.path.isdir(dest)
    assert _listdir_md(dest) == []


def test_sandbox_missing_mock_is_empty_not_error(monkeypatch, tmp_path):
    sync_root, _ = _redirect_roots(monkeypatch, tmp_path)
    dest = _run(SANDBOX_USER_ID)  # mock root has no sandbox-user/ subdir
    assert dest == str(sync_root / SANDBOX_USER_ID)
    assert os.listdir(dest) == []


def test_sandbox_copies_nested_markdown(monkeypatch, tmp_path):
    _, mock_root = _redirect_roots(monkeypatch, tmp_path)
    src = mock_root / SANDBOX_USER_ID / "projects"
    src.mkdir(parents=True)
    (src / "plan.md").write_text("nested", encoding="utf-8")

    dest = _run(SANDBOX_USER_ID)
    assert os.path.isfile(os.path.join(dest, "projects", "plan.md"))


# --- real-user path: S3 under the <user_id>/ prefix -------------------------

def test_real_user_downloads_under_prefix(monkeypatch, tmp_path):
    sync_root, _ = _redirect_roots(monkeypatch, tmp_path)
    fake = _single_page(["user-123/a.md", "user-123/b.md"])
    _install_s3(monkeypatch, fake)

    dest = _run("user-123")

    assert dest == str(sync_root / "user-123")
    # Listed with the caller's prefix, and both notes landed locally.
    assert fake.list_calls[0]["Prefix"] == "user-123/"
    assert _listdir_md(dest) == ["a.md", "b.md"]


def test_real_user_skips_non_markdown(monkeypatch, tmp_path):
    _redirect_roots(monkeypatch, tmp_path)
    fake = _single_page(["user-123/keep.md", "user-123/skip.txt", "user-123/image.png"])
    _install_s3(monkeypatch, fake)

    dest = _run("user-123")

    assert _listdir_md(dest) == ["keep.md"]
    # Only the .md object was ever downloaded.
    assert [key for _, key, _ in fake.downloaded] == ["user-123/keep.md"]


def test_real_user_skips_prefix_placeholder(monkeypatch, tmp_path):
    _redirect_roots(monkeypatch, tmp_path)
    # A bare "<prefix>/" key is the folder marker, not a file — must be skipped.
    fake = _single_page(["user-123/", "user-123/real.md"])
    _install_s3(monkeypatch, fake)

    dest = _run("user-123")
    assert _listdir_md(dest) == ["real.md"]


def test_real_user_preserves_nested_keys(monkeypatch, tmp_path):
    _redirect_roots(monkeypatch, tmp_path)
    fake = _single_page(["user-123/sub/deep.md"])
    _install_s3(monkeypatch, fake)

    dest = _run("user-123")
    assert os.path.isfile(os.path.join(dest, "sub", "deep.md"))


def test_real_user_follows_pagination(monkeypatch, tmp_path):
    _redirect_roots(monkeypatch, tmp_path)
    fake = _FakeS3([(["user-123/p1.md"], "tok"), (["user-123/p2.md"], None)])
    _install_s3(monkeypatch, fake)

    dest = _run("user-123")

    assert _listdir_md(dest) == ["p1.md", "p2.md"]
    # Second page requested with the continuation token from the first.
    assert fake.list_calls[1]["ContinuationToken"] == "tok"


# --- snapshot semantics: dest cleared each sync -----------------------------

def test_sync_preserves_user_preferences_across_wipe(monkeypatch, tmp_path):
    # The locally-generated profile must survive the clean-snapshot wipe, while
    # ordinary stale notes are still cleared and mock/remote notes still sync in.
    sync_root, mock_root = _redirect_roots(monkeypatch, tmp_path)
    user = SANDBOX_USER_ID
    (sync_root / user).mkdir(parents=True)
    (sync_root / user / "user_preferences.md").write_text(
        "# User Preferences\n- keep me\n", encoding="utf-8"
    )
    (sync_root / user / "stale.md").write_text("old", encoding="utf-8")
    (mock_root / user).mkdir(parents=True)
    (mock_root / user / "fresh.md").write_text("new", encoding="utf-8")

    dest = _run(user)
    names = sorted(os.listdir(dest))
    assert "user_preferences.md" in names   # preserved across the wipe
    assert "fresh.md" in names              # synced from the mock source
    assert "stale.md" not in names          # ordinary stale note cleared
    assert (
        (sync_root / user / "user_preferences.md").read_text(encoding="utf-8")
        == "# User Preferences\n- keep me\n"
    )


def test_sync_clears_stale_files(monkeypatch, tmp_path):
    sync_root, mock_root = _redirect_roots(monkeypatch, tmp_path)
    dest_dir = sync_root / SANDBOX_USER_ID
    dest_dir.mkdir(parents=True)
    (dest_dir / "stale.md").write_text("old", encoding="utf-8")

    src = mock_root / SANDBOX_USER_ID
    src.mkdir(parents=True)
    (src / "fresh.md").write_text("new", encoding="utf-8")

    dest = _run(SANDBOX_USER_ID)
    assert _listdir_md(dest) == ["fresh.md"]  # stale.md was cleared


# --- S3 write-back (durability) ---------------------------------------------

def test_upload_user_file_puts_to_bucket(monkeypatch, tmp_path):
    sync_root, _ = _redirect_roots(monkeypatch, tmp_path)
    (sync_root / "user-123").mkdir(parents=True)
    (sync_root / "user-123" / "user_preferences.md").write_text(
        "# User Preferences\n- red\n", encoding="utf-8"
    )
    fake = _single_page([])
    _install_s3(monkeypatch, fake)

    storage_sync.upload_user_file("user-123", "user_preferences.md")

    assert fake.uploaded == [
        ("user-vaults", "user-123/user_preferences.md", "# User Preferences\n- red\n")
    ]


def test_upload_user_file_sandbox_is_noop(monkeypatch, tmp_path):
    _redirect_roots(monkeypatch, tmp_path)

    def _boom():
        raise AssertionError("sandbox write-back must not build an S3 client")

    monkeypatch.setattr(storage_sync, "build_s3_client", _boom)
    storage_sync.upload_user_file(SANDBOX_USER_ID, "user_preferences.md")  # no raise


def test_upload_user_file_missing_local_is_noop(monkeypatch, tmp_path):
    _redirect_roots(monkeypatch, tmp_path)
    fake = _single_page([])
    _install_s3(monkeypatch, fake)
    storage_sync.upload_user_file("user-123", "user_preferences.md")  # file absent
    assert fake.uploaded == []


def test_fresh_sync_restores_profile_from_s3(monkeypatch, tmp_path):
    # Step 3: a fresh container (empty local root) downloads the profile back.
    sync_root, _ = _redirect_roots(monkeypatch, tmp_path)
    fake = _single_page(["user-123/user_preferences.md", "user-123/note.md"])
    _install_s3(monkeypatch, fake)

    dest = _run("user-123")
    assert "user_preferences.md" in _listdir_md(dest)  # restored on startup
    assert (sync_root / "user-123" / "user_preferences.md").is_file()
