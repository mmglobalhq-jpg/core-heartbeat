"""Every model attempt the orchestrator makes reaches the token ledger.

Heartbeat was the worst case the 2026-09-15 audit found: it read token counts for the
router and the answer, then threw them away, and never counted memory extraction, web
search grounding, image re-reads or titles at all. Nothing downstream could tell
"spent nothing" from "measured nothing".

These tests read the spool file the wrapper actually writes, not a mock: that file is
the interface to the host drain, so a row that never reaches disk never reaches the
ledger (platform doc 07 §3a).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import orchestrator
from models import HistoryTurn
from services import llm_ledger

from .test_tool_catalog import _native_state

SERVICE = "heartbeat-test"


@pytest.fixture(autouse=True)
def _service_name(monkeypatch: pytest.MonkeyPatch) -> None:
    # In production LLM_LEDGER_SERVICE=heartbeat comes from compose; the tests pin their
    # own so the rows land somewhere predictable under the isolated spool.
    monkeypatch.setattr(llm_ledger, "SERVICE", SERVICE)


def _rows() -> list[dict[str, Any]]:
    d = Path(llm_ledger.SPOOL_DIR) / SERVICE
    if not d.exists():
        return []
    return [
        json.loads(line)
        for f in sorted(d.iterdir())
        for line in f.read_text().splitlines()
        if line.strip()
    ]


class _Usage:
    def __init__(self, inp: int, out: int) -> None:
        self.input_tokens = inp
        self.output_tokens = out


class _Block:
    type = "tool_use"
    input: dict[str, Any] = {}


class _AnthropicResponse:
    def __init__(self) -> None:
        self.usage = _Usage(1200, 40)
        self.content = [_Block()]
        self.model = "claude-haiku-4-5"
        self.id = "msg_test"


class _FakeAnthropic:
    """Minimal stand-in for the Anthropic client the router is handed."""

    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc
        self.messages = self

    def create(self, **_kwargs: Any) -> _AnthropicResponse:
        if self._exc is not None:
            raise self._exc
        return _AnthropicResponse()


class _LangChainResponse:
    """LangChain's provider-agnostic usage_metadata — a third spelling.

    input_tokens is the SUM of all input types, so the cached and cache-written tokens
    are already inside it; the raw Anthropic shape means the opposite.
    """

    usage_metadata = {
        "input_tokens": 5000,
        "output_tokens": 900,
        "total_tokens": 5900,
        "input_token_details": {"cache_read": 3500, "cache_creation": 500},
        "output_token_details": {"reasoning": 200},
    }
    tool_calls: list[dict[str, Any]] = []


class _FakeBound:
    def invoke(self, _messages: Any) -> _LangChainResponse:
        return _LangChainResponse()


def test_anthropic_router_records_the_attempt() -> None:
    orchestrator._decide_anthropic(_native_state(), _FakeAnthropic(), "claude-haiku-4-5")

    rows = _rows()
    assert len(rows) == 1
    assert rows[0]["operation"] == "heartbeat.router.anthropic"
    assert rows[0]["provider"] == "anthropic"
    assert rows[0]["status"] == "ok"
    assert rows[0]["usage_source"] == "provider"
    assert rows[0]["input_uncached"] == 1200
    assert rows[0]["output_billable"] == 40
    assert rows[0]["model_served"] == "claude-haiku-4-5"


def test_a_response_the_router_cannot_parse_is_still_recorded_as_spend() -> None:
    # The fake returns a tool_use block with an empty input, so RoutingDecision
    # validation fails and the turn degrades — but the tokens were still billed.
    decision, failure, _usage = orchestrator._decide_anthropic(
        _native_state(), _FakeAnthropic(), "claude-haiku-4-5"
    )
    assert decision is None and failure is not None

    rows = _rows()
    assert len(rows) == 1
    assert rows[0]["usage_source"] == "provider"
    assert rows[0]["input_uncached"] == 1200


def test_a_failed_call_records_with_no_usage_rather_than_zeros() -> None:
    orchestrator._decide_anthropic(
        _native_state(), _FakeAnthropic(RuntimeError("upstream exploded")), "claude-haiku-4-5"
    )

    rows = _rows()
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert rows[0]["usage_source"] == "missing"
    assert rows[0]["input_uncached"] is None
    assert rows[0]["error_type"] == "RuntimeError"


def test_native_router_uses_the_langchain_spelling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orchestrator, "_native_chat_model", lambda _p: _FakeBound())

    orchestrator._decide_native(_native_state(), "claude-haiku-4-5")

    rows = _rows()
    assert len(rows) == 1
    assert rows[0]["operation"] == "heartbeat.router.native"
    # 5000 total input MINUS 3500 cache reads MINUS 500 cache writes = 1000 uncached.
    # Reading input_tokens as "uncached" here would overstate the bill five-fold.
    assert rows[0]["input_uncached"] == 1000
    assert rows[0]["cache_read"] == 3500
    assert rows[0]["cache_write_5m"] == 500
    assert rows[0]["output_billable"] == 900
    assert rows[0]["thinking_tokens"] == 200


def test_title_generation_is_recorded_even_though_it_is_free() -> None:
    # Local inference costs nothing, but an empty ledger for ollama must mean
    # "not called", never "not recorded".
    class _Resp:
        status_code = 200

        @staticmethod
        def json() -> dict[str, Any]:
            return {"response": "A title", "prompt_eval_count": 42, "eval_count": 7}

    class _Client:
        async def post(self, *_a: Any, **_k: Any) -> _Resp:
            return _Resp()

    import asyncio

    turns = [HistoryTurn(role="user", content="hi")]
    asyncio.run(orchestrator.generate_title(turns, _Client()))

    rows = _rows()
    assert len(rows) == 1
    assert rows[0]["provider"] == "ollama"
    assert rows[0]["operation"] == "heartbeat.title"
    assert rows[0]["input_uncached"] == 42
    assert rows[0]["output_billable"] == 7
