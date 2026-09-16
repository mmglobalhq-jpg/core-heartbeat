"""Shared fixtures.

The only thing here is spool isolation for the token ledger: the orchestrator records
one row per model attempt (platform doc 07 §3a), including in unit tests against fake
clients. Without this, a test run would try to write into the real host spool
(`/var/spool/llm-ledger/heartbeat`) — absent on a dev box, and REAL on the mini PC,
where it would put test rows into the production ledger.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _ledger_spool(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    from services import llm_ledger

    monkeypatch.setattr(llm_ledger, "SPOOL_DIR", str(tmp_path_factory.mktemp("llm-ledger")))
    yield
