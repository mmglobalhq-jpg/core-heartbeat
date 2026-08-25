"""Route + service tests for Settings -> File Upload (POST/GET/DELETE /uploads).

The admin gate is the whole security boundary for this feature (core-chat proxies
to it from an edge with no Cloudflare Access), so it is tested first and on every
verb. The rest of the file is about the target being a Windows directory: a name
that is merely POSIX-safe can still be permanently unopenable in Explorer.
"""
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from auth import SANDBOX_USER_ID, resolve_user_id
from main import create_app
import services.kb as kb
import services.uploads as up

USER = "11111111-1111-1111-1111-111111111111"


@pytest.fixture
def root(tmp_path, monkeypatch):
    """An upload root that looks mounted (sentinel present)."""
    d = tmp_path / "Uploads"
    d.mkdir()
    (d / up.SENTINEL).write_text("test root")
    monkeypatch.setenv(up.UPLOAD_ROOT_ENV, str(d))
    return d


@pytest.fixture
def client(root, monkeypatch):
    app = create_app()
    app.dependency_overrides[resolve_user_id] = lambda: USER
    _admin(monkeypatch, True)
    return TestClient(app)


def _admin(monkeypatch, value: bool):
    async def _is_admin(uid):
        return value
    monkeypatch.setattr(kb, "is_admin", _is_admin)


def _put(client, name: str, body: bytes = b"hello"):
    return client.post("/uploads", content=body, headers={"X-Upload-Filename": name})


# --- the admin gate ---------------------------------------------------------


def test_upload_requires_authentication(root, monkeypatch):
    app = create_app()
    app.dependency_overrides[resolve_user_id] = lambda: SANDBOX_USER_ID
    _admin(monkeypatch, True)  # even if the profile lookup would say yes
    c = TestClient(app)
    assert _put(c, "x.txt").status_code == 401
    assert c.get("/uploads").status_code == 401
    assert c.delete("/uploads/x.txt").status_code == 401


def test_upload_requires_admin_on_every_verb(root, monkeypatch):
    app = create_app()
    app.dependency_overrides[resolve_user_id] = lambda: USER
    _admin(monkeypatch, False)
    c = TestClient(app)
    assert _put(c, "x.txt").status_code == 403
    assert c.get("/uploads").status_code == 403
    assert c.delete("/uploads/x.txt").status_code == 403
    assert list(root.iterdir()) == [root / up.SENTINEL]


def test_admin_check_fails_closed_when_profiles_unreachable(root, monkeypatch):
    async def _boom(uid):
        raise RuntimeError("supabase down")
    monkeypatch.setattr(kb, "is_admin", _boom)
    app = create_app()
    app.dependency_overrides[resolve_user_id] = lambda: USER
    assert _put(TestClient(app), "x.txt").status_code == 403


# --- the happy path ---------------------------------------------------------


def test_upload_stores_the_bytes(client, root):
    r = _put(client, "notes.txt", b"the quick brown fox")
    assert r.status_code == 200
    body = r.json()
    assert body["stored_as"] == "notes.txt"
    assert body["size_bytes"] == 19
    assert body["rewritten"] is False
    assert body["windows_path"].endswith("\\notes.txt")
    assert (root / "notes.txt").read_bytes() == b"the quick brown fox"


def test_no_part_file_survives_a_successful_upload(client, root):
    _put(client, "notes.txt")
    assert [p.name for p in root.iterdir() if p.name.endswith(up.PART_SUFFIX)] == []


# --- path containment -------------------------------------------------------


@pytest.mark.parametrize(
    "attack",
    ["../../etc/passwd", "/etc/passwd", "..\\..\\windows\\system32\\evil.dll", "....//x.txt"],
)
def test_traversal_attempts_stay_inside_the_root(client, root, attack):
    r = _put(client, attack)
    assert r.status_code == 200
    stored = root / r.json()["stored_as"]
    assert stored.parent.resolve() == root.resolve()
    # and nothing was created anywhere above the root
    assert not (root.parent / "etc").exists()


def test_nul_and_control_characters_are_stripped(client, root):
    r = _put(client, "ev\x00il\x1fname.txt")
    assert r.status_code == 200
    assert "\x00" not in r.json()["stored_as"]


def test_filename_header_is_required(client):
    r = client.post("/uploads", content=b"x", headers={"X-Upload-Filename": "   "})
    assert r.status_code == 400


# --- Windows-specific name hardening ---------------------------------------
#
# Verified on the box 2026-08-25: 9p lets Linux create CON.txt on the NTFS mount,
# Windows then LISTS it but cannot open it (ItemNotFoundException). A file that
# uploads "fine" and is permanently unopenable is the failure this prevents.


@pytest.mark.parametrize("name", ["CON.txt", "con.pdf", "NUL", "com1.log", "LPT9.dat"])
def test_reserved_device_names_are_rewritten(client, root, name):
    r = _put(client, name)
    assert r.status_code == 200
    stored = r.json()["stored_as"]
    assert stored != name
    assert r.json()["rewritten"] is True
    assert stored.partition(".")[0].lower() not in up._RESERVED_STEMS


def test_trailing_dots_and_spaces_are_stripped(client, root):
    # NTFS silently drops these on create, so the file we think we wrote would not
    # be the file that exists.
    r = _put(client, "report. ")
    assert r.json()["stored_as"] == "report"


@pytest.mark.parametrize("ch", ['<', '>', ':', '"', '|', '?', '*'])
def test_windows_illegal_characters_are_replaced(client, ch):
    r = _put(client, f"a{ch}b.txt")
    assert r.status_code == 200
    assert ch not in r.json()["stored_as"]


def test_leading_dots_are_stripped_so_the_sentinel_cannot_be_targeted(client, root):
    r = _put(client, up.SENTINEL, b"overwritten!")
    assert r.status_code == 200
    assert r.json()["stored_as"] != up.SENTINEL
    assert (root / up.SENTINEL).read_text() == "test root"


def test_long_names_are_capped_but_keep_their_extension(client):
    r = _put(client, "a" * 400 + ".pdf")
    stored = r.json()["stored_as"]
    assert len(stored.encode()) <= up.MAX_NAME_BYTES
    assert stored.endswith(".pdf")


# --- collisions -------------------------------------------------------------


def test_second_upload_of_a_name_does_not_overwrite(client, root):
    _put(client, "report.pdf", b"first")
    r = _put(client, "report.pdf", b"second")
    assert r.json()["stored_as"] == "report (2).pdf"
    assert (root / "report.pdf").read_bytes() == b"first"
    assert (root / "report (2).pdf").read_bytes() == b"second"
    assert r.json()["rewritten"] is True


def test_collision_detection_is_case_insensitive_on_a_case_insensitive_mount(client, root):
    """DrvFs (the real target) is case-insensitive, so Report.pdf collides with
    report.pdf exactly as it would in Explorer. tmp_path is usually ext4, which is
    not — so this asserts the real behaviour only where the filesystem can show it,
    rather than passing green on a filesystem that does not match production."""
    _put(client, "report.pdf", b"first")
    if not (root / "REPORT.PDF").exists():
        pytest.skip("test filesystem is case-sensitive; verified on DrvFs instead")
    r = _put(client, "Report.pdf", b"second")
    assert r.json()["stored_as"] != "Report.pdf"


# --- limits -----------------------------------------------------------------


def test_oversized_upload_is_refused_and_leaves_no_partial_file(client, root, monkeypatch):
    monkeypatch.setenv(up.MAX_UPLOAD_BYTES_ENV, "16")
    r = _put(client, "big.bin", b"x" * 4096)
    assert r.status_code == 413
    assert [p.name for p in root.iterdir() if p.name != up.SENTINEL] == []


def test_empty_upload_is_refused(client, root):
    r = _put(client, "empty.txt", b"")
    assert r.status_code == 400
    assert [p.name for p in root.iterdir() if p.name != up.SENTINEL] == []


def test_upload_refused_when_the_folder_is_over_budget(client, root, monkeypatch):
    (root / "existing.bin").write_bytes(b"x" * 1024)
    monkeypatch.setenv(up.UPLOAD_TOTAL_BUDGET_BYTES_ENV, "512")
    assert _put(client, "another.txt").status_code == 507


# --- the missing-mount guard ------------------------------------------------


def test_every_verb_refuses_when_the_sentinel_is_missing(client, root):
    """Docker creates a bind-mount source that isn't there, so a WSL boot without
    /mnt/c would silently redirect uploads into an ext4 directory Windows cannot
    see. Failing loudly is the entire point of the sentinel."""
    (root / up.SENTINEL).unlink()
    assert _put(client, "x.txt").status_code == 503
    assert client.get("/uploads").status_code == 503
    assert client.delete("/uploads/x.txt").status_code == 503
    assert list(root.iterdir()) == []


# --- listing ----------------------------------------------------------------


def test_listing_includes_files_dropped_in_from_windows(client, root):
    (root / "dropped-by-hand.xlsx").write_bytes(b"xx")
    _put(client, "uploaded.txt", b"yyy")
    files = {f["name"]: f for f in client.get("/uploads").json()["files"]}
    assert "dropped-by-hand.xlsx" in files
    assert files["uploaded.txt"]["size_bytes"] == 3


def test_listing_hides_the_sentinel_and_partial_uploads(client, root):
    (root / ("half.bin" + up.PART_SUFFIX)).write_bytes(b"x")
    names = [f["name"] for f in client.get("/uploads").json()["files"]]
    assert up.SENTINEL not in names
    assert not any(n.endswith(up.PART_SUFFIX) for n in names)


def test_listing_is_newest_first(client, root):
    for n, t in (("old.txt", 1_000_000), ("new.txt", 2_000_000)):
        p = root / n
        p.write_bytes(b"x")
        os.utime(p, (t, t))
    assert [f["name"] for f in client.get("/uploads").json()["files"]][:2] == ["new.txt", "old.txt"]


# --- delete -----------------------------------------------------------------


def test_delete_removes_the_file(client, root):
    _put(client, "gone.txt")
    assert client.delete("/uploads/gone.txt").status_code == 200
    assert not (root / "gone.txt").exists()


def test_delete_refuses_to_escape_the_root(client, root, tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("do not delete me")
    r = client.delete("/uploads/%2e%2e%2fvictim.txt")
    assert r.status_code in (400, 404)
    assert victim.exists()


def test_delete_refuses_the_sentinel(client, root):
    assert client.delete(f"/uploads/{up.SENTINEL}").status_code == 400
    assert (root / up.SENTINEL).exists()


def test_delete_of_a_missing_file_is_404(client):
    assert client.delete("/uploads/nope.txt").status_code == 404
