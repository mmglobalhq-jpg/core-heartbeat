"""Tests for document upload/parse (docling) + injection into the answer prompt.

Docling is never invoked here — the fast path handles textual files and the docling
worker is monkeypatched — so these run with or without docling installed, no network.
"""

import asyncio

from starlette.testclient import TestClient

import orchestrator
import router
import services.documents as docstore
from auth import resolve_user_id
from main import create_app
from models import IntentPayload
from services.document_parser import parse_document


# --- parser -----------------------------------------------------------------

def test_fast_path_txt_decodes_directly():
    assert asyncio.run(parse_document(b"hello world", "notes.txt", "text/plain")) == "hello world"


def test_fast_path_csv():
    out = asyncio.run(parse_document(b"a,b\n1,2", "data.csv", "text/csv"))
    assert "a,b" in out


def test_non_text_routes_to_docling(monkeypatch):
    import services.document_parser as dp

    monkeypatch.setattr(dp, "_docling_extract", lambda data, filename: "# Parsed")
    out = asyncio.run(parse_document(b"%PDF-1.4 ...", "doc.pdf", "application/pdf"))
    assert out == "# Parsed"


# --- POST /documents/parse --------------------------------------------------

def test_parse_endpoint_ready(monkeypatch):
    monkeypatch.setattr(docstore, "fetch_original", lambda uid, did: b"raw bytes")
    stored = {}
    monkeypatch.setattr(
        docstore, "store_extracted", lambda uid, did, text: stored.update(text=text)
    )

    async def fake_parse(data, filename, content_type=None):
        return "extracted text"

    monkeypatch.setattr(router, "parse_document", fake_parse)

    app = create_app()
    app.dependency_overrides[resolve_user_id] = lambda: "real-user-123"
    client = TestClient(app)
    r = client.post("/documents/parse", json={"doc_id": "d1", "filename": "doc.pdf"})
    assert r.status_code == 200
    assert r.json() == {"status": "ready", "char_count": 14, "error": None}
    assert stored["text"] == "extracted text"


def test_parse_endpoint_requires_auth():
    client = TestClient(create_app())
    r = client.post("/documents/parse", json={"doc_id": "d1", "filename": "doc.pdf"})
    assert r.status_code == 401  # no JWT -> sandbox -> rejected (docs are per-user)


def test_parse_endpoint_error_on_missing_original(monkeypatch):
    def boom(uid, did):
        raise FileNotFoundError

    monkeypatch.setattr(docstore, "fetch_original", boom)
    app = create_app()
    app.dependency_overrides[resolve_user_id] = lambda: "real-user-123"
    r = TestClient(app).post("/documents/parse", json={"doc_id": "x", "filename": "a.pdf"})
    assert r.status_code == 200
    assert r.json()["status"] == "error"


# --- orchestrator injection -------------------------------------------------

def test_load_documents_concatenates(monkeypatch):
    monkeypatch.setattr(
        docstore, "fetch_extracted", lambda uid, did: {"d1": "AAA", "d2": "BBB"}.get(did, "")
    )
    out = asyncio.run(orchestrator._load_documents("u", ["d1", "d2"]))
    assert "AAA" in out and "BBB" in out


def test_load_documents_empty_returns_blank():
    assert asyncio.run(orchestrator._load_documents("u", [])) == ""


def test_load_documents_truncates_to_budget(monkeypatch):
    big = "x" * (orchestrator.DOC_CHAR_BUDGET + 5000)
    monkeypatch.setattr(docstore, "fetch_extracted", lambda uid, did: big)
    out = asyncio.run(orchestrator._load_documents("u", ["d1"]))
    assert len(out) <= orchestrator.DOC_CHAR_BUDGET + 100  # capped (+ truncation note)


def test_local_prompt_includes_attached_documents():
    state = {
        "intent": IntentPayload(intent="chat", confidence=0.9, raw_input="summarize", source="t"),
        "messages": [],
        "prior_context": [],
        "documents": "DOC TEXT HERE",
        "user_id": "sandbox-user",
    }
    prompt = orchestrator._build_local_prompt(state)
    assert "DOC TEXT HERE" in prompt
    assert "ATTACHED DOCUMENTS" in prompt


def test_local_prompt_no_documents_block_when_empty():
    state = {
        "intent": IntentPayload(intent="chat", confidence=0.9, raw_input="hi", source="t"),
        "messages": [],
        "prior_context": [],
        "documents": "",
        "user_id": "sandbox-user",
    }
    assert "ATTACHED DOCUMENTS" not in orchestrator._build_local_prompt(state)
