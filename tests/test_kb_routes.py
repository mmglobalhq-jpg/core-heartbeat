"""Route tests for the KB gateway endpoints — admin gating + scope resolution.

Uses dependency overrides for resolve_user_id and monkeypatches services.kb /
services.documents so no live KB service or storage is needed.
"""
import pytest
from fastapi.testclient import TestClient

from auth import SANDBOX_USER_ID, resolve_user_id
from main import create_app
import services.kb as kb
import services.documents as docs

USER = "11111111-1111-1111-1111-111111111111"


@pytest.fixture
def client(monkeypatch):
    app = create_app()
    app.dependency_overrides[resolve_user_id] = lambda: USER
    monkeypatch.setattr(docs, "fetch_original", lambda uid, did: b"hello world")

    async def _ingest(owner, filename, content):
        return {"job_id": "job-1", "status": "pending", "_owner": owner}

    async def _list(owner):
        return {"documents": [{"id": "d1", "title": "t", "scope": "global", "_owner": owner}]}

    async def _delete(doc_id, owner):
        return True

    monkeypatch.setattr(kb, "ingest", _ingest)
    monkeypatch.setattr(kb, "list_documents", _list)
    monkeypatch.setattr(kb, "delete_document", _delete)
    return TestClient(app)


def _admin(monkeypatch, value: bool):
    async def _is_admin(uid):
        return value
    monkeypatch.setattr(kb, "is_admin", _is_admin)


def test_global_ingest_blocked_for_non_admin(client, monkeypatch):
    _admin(monkeypatch, False)
    r = client.post("/kb/ingest", json={"doc_id": "d1", "filename": "f.txt", "scope": "global"})
    assert r.status_code == 403


def test_global_ingest_allowed_for_admin_stamps_global(client, monkeypatch):
    _admin(monkeypatch, True)
    r = client.post("/kb/ingest", json={"doc_id": "d1", "filename": "f.txt", "scope": "global"})
    assert r.status_code == 200
    assert r.json()["_owner"] == "global"


def test_private_ingest_uses_user_owner(client, monkeypatch):
    _admin(monkeypatch, False)  # irrelevant for private scope
    r = client.post("/kb/ingest", json={"doc_id": "d1", "filename": "f.txt"})
    assert r.status_code == 200
    assert r.json()["_owner"] == USER


def test_documents_list_proxied_with_user_owner(client):
    r = client.get("/kb/documents")
    assert r.status_code == 200
    assert r.json()["documents"][0]["_owner"] == USER


def test_delete_global_blocked_for_non_admin(client, monkeypatch):
    _admin(monkeypatch, False)
    assert client.delete("/kb/documents/d1?scope=global").status_code == 403


def test_delete_global_allowed_for_admin_uses_global_owner(client, monkeypatch):
    _admin(monkeypatch, True)
    seen = {}

    async def _del(doc_id, owner):
        seen["owner"] = owner
        return True

    monkeypatch.setattr(kb, "delete_document", _del)
    r = client.delete("/kb/documents/d1?scope=global")
    assert r.status_code == 200
    assert seen["owner"] == "global"


def test_delete_own_uses_user_owner(client, monkeypatch):
    seen = {}

    async def _del(doc_id, owner):
        seen["owner"] = owner
        return True

    monkeypatch.setattr(kb, "delete_document", _del)
    r = client.delete("/kb/documents/d1")  # scope defaults to private
    assert r.status_code == 200
    assert seen["owner"] == USER


def test_delete_missing_returns_404(client, monkeypatch):
    async def _del(doc_id, owner):
        return False

    monkeypatch.setattr(kb, "delete_document", _del)
    assert client.delete("/kb/documents/nope").status_code == 404


def test_sandbox_user_rejected():
    app = create_app()
    app.dependency_overrides[resolve_user_id] = lambda: SANDBOX_USER_ID
    c = TestClient(app)
    assert c.post("/kb/ingest", json={"doc_id": "d1", "filename": "f.txt"}).status_code == 401
    assert c.get("/kb/documents").status_code == 401
