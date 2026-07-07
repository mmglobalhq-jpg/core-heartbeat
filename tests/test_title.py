"""Tests for local-LLM chat title generation (POST /title).

All Ollama access goes through httpx.MockTransport — no network, no daemon.
Covers the _clean_title sanitizer, generate_title's failure-tolerance, and the
endpoint contract (200 {title} on success/refusal, 422 on empty messages).
"""

import asyncio

import httpx
from starlette.testclient import TestClient

import orchestrator
from main import create_app
from models import HistoryTurn
from orchestrator import _clean_title, generate_title


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _ollama_json(body):
    def _h(request):
        return httpx.Response(200, json=body)
    return _h


TURNS = [
    HistoryTurn(role="user", content="Any good museums to see in DC?"),
    HistoryTurn(role="assistant", content="The Smithsonian museums are all free and excellent."),
]


# --- _clean_title -----------------------------------------------------------

def test_clean_title_strips_quotes_and_punctuation():
    assert _clean_title('"DC Museums".') == "DC Museums"
    assert _clean_title("`DC Museums`") == "DC Museums"
    assert _clean_title("DC Museums!!!") == "DC Museums"


def test_clean_title_takes_first_nonempty_line_and_collapses_space():
    assert _clean_title("\n\n  DC   Museums \nextra") == "DC Museums"


def test_clean_title_rejects_refusals_and_sentences():
    assert _clean_title("Sure, here is a title: DC Museums") is None
    assert _clean_title("I can't create a title for that.") is None
    assert _clean_title("Here's a good one") is None


def test_clean_title_rejects_empty_and_overlong():
    assert _clean_title("") is None
    assert _clean_title(None) is None
    assert _clean_title("x" * 80) is None  # a long single line is a sentence


def test_clean_title_caps_length():
    # 11 words (~54 chars) — under the 60-char "that's a sentence" reject line,
    # but over the 48-char cap, so it exercises truncation.
    out = _clean_title("Word " * 11)
    assert out is not None and len(out) <= 48


# --- generate_title (mocked Ollama) -----------------------------------------

def test_generate_title_returns_clean_label():
    client = _mock_client(_ollama_json({"response": '  "DC Museums"\n'}))
    title = asyncio.run(generate_title(TURNS, client))
    assert title == "DC Museums"


def test_generate_title_none_on_refusal():
    client = _mock_client(_ollama_json({"response": "I'm sorry, I cannot do that."}))
    assert asyncio.run(generate_title(TURNS, client)) is None


def test_generate_title_none_on_http_error():
    client = _mock_client(lambda req: httpx.Response(500, json={}))
    assert asyncio.run(generate_title(TURNS, client)) is None


def test_generate_title_none_on_transport_error():
    def boom(req):
        raise httpx.ConnectError("refused")
    assert asyncio.run(generate_title(TURNS, _mock_client(boom))) is None


def test_generate_title_none_on_empty_turns():
    client = _mock_client(_ollama_json({"response": "DC Museums"}))
    assert asyncio.run(generate_title([], client)) is None


# --- POST /title endpoint ---------------------------------------------------

def test_title_endpoint_returns_label(monkeypatch):
    # Stub generate_title on the router's namespace so the endpoint does no
    # network (build_ollama_client still opens/closes a client but never calls).
    async def fake_title(turns, client):
        return "DC Museums"
    monkeypatch.setattr("router.generate_title", fake_title)
    client = TestClient(create_app())
    r = client.post("/title", json={"messages": [{"role": "user", "content": "museums in DC?"}]})
    assert r.status_code == 200
    assert r.json() == {"title": "DC Museums"}


def test_title_endpoint_422_on_empty_messages():
    client = TestClient(create_app())
    r = client.post("/title", json={"messages": []})
    assert r.status_code == 422


def test_title_endpoint_null_title_passthrough(monkeypatch):
    async def fake_title(turns, client):
        return None
    monkeypatch.setattr("router.generate_title", fake_title)
    client = TestClient(create_app())
    r = client.post("/title", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert r.json() == {"title": None}
